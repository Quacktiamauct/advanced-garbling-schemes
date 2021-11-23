import hashlib
import secrets
import itertools

translate = {
    "O-": 0,
    "O+": 1,
    "A-": 2,
    "A+": 3,
    "B-": 4,
    "B+": 5,
    "AB-": 6,
    "AB+": 7,
}
bloodtypeBits = {
    "O-": [0, 0, 0],
    "O+": [0, 0, 1],
    "A-": [0, 1, 0],
    "A+": [0, 1, 1],
    "B-": [1, 0, 0],
    "B+": [1, 0, 1],
    "AB-": [1, 1, 0],
    "AB+": [1, 1, 1],
}
bloodtypes = [
    "O-", "O+",
    "A-", "A+",
    "B-", "B+",
    "AB-", "AB+"
]

def truth_table_lookup(donor, receiver):
    compat = [
            [True, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, False, True, False, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, False, False, False, True, False, False, False],
            [True, True, False, False, True, True, False, False],
            [True, False, True, False, True, False, True, False],
            [True, True, True, True, True, True, True, True],
        ]
    # We need to encrypt a number, these are arbitrarily chosen to represent true and false
    if compat[translate[receiver]][translate[donor]]:
        return True
    else:
        return False

# OT functions
p = 3727
q = int((p - 1) / 2)
g = 113
def gen():
    sk = secrets.randbelow(q - 1) + 1
    h = pow(g, sk, p)
    return sk, h

def ogen():
    r = secrets.randbelow(q - 1) + 1
    h = pow(r, 2, p)
    return r, h

def enc_one(pk, m):
    r = secrets.randbelow(q - 1) + 1
    return pow(g, r, p), (m * pow(pk, r, p)) % p

def dec_one(sk, c):
    c1, c2 = c
    return c2 * pow(c1, -sk, p) % p

def enc(pk, data):
    cipher = list()
    for b in data:
        cipher.append(enc_one(pk, b + 255))
    return cipher

def dec(sk, data):
    result = list()
    for c in data:
        result.append(dec_one(sk, c) - 255)
    return bytes(result)

# Convinience functions
def to_num(data):
    return int.from_bytes(data, byteorder='big')

def to_bytes(num):
    return int.to_bytes(num, (num.bit_length() + 7), byteorder='big', signed=False)

def zero(a):
    for b in a:
        if b != 0:
            return False
    return True

def xor(bytes1, bytes2):
    return bytes(b1 ^ b2 for (b1, b2) in zip(bytes1, bytes2))

# Garble functions
def G(A, B, i):
    hash = hashlib.sha256(A + B + to_bytes(i)).digest()
    left = hash[:16]
    right = hash[16:]
    return left, right    

def pick_random_pair():
    rnd = secrets.token_bytes(32)
    left = rnd[:16]
    right = rnd[16:]
    return [left, right]

# Circuit:
# aA bA aB bB  aR bR  <- wire 0, 1, 2, 3, 4, 5
#  | |   | |    |  |
#  | !   | !    |  !  <- wire 6, 7, 8
#  \ /   \ /    \  /
#   &     &       &   <- wire 9, 10, 11 (NAND Gates)
#   !     !       !
#   \    /        |
#    \  /         |
#      &          |   <- wire 12
#       \         /
#        \       /
#         \     /
#          \   /
#           \ /
#            &        <- wire 13
# 6  input wires
# 10 intermediate wires
# 1  output wire
T = 14
n = 6
outputWire = T - 1

lParents = {
    # NOT gates
    6: 1,
    7: 3,
    8: 5,
    # NAND Gates
    9:  0,
    10: 2,
    11: 4,
    # AND Gates
    12: 9,
    13: 12
}
def L(i):
    return lParents[i]

rParents = {
    # NOT gates (note should use left)
    6: 1,
    7: 3,
    8: 5,
    # NAND Gates
    9:  6,
    10: 7,
    11: 8,
    # AND Gates
    12: 10,
    13: 11
}
def R(i):
    return rParents[i]

# Truth tables for gates
values = {
    # NOT Gates
    6: [1,-1,-1,0],
    7: [1,-1,-1,0],
    8: [1,-1,-1,0],
    # NAND Gates
    9:  [1,1,1,0],
    10: [1,1,1,0],
    11: [1,1,1,0],
    # AND Gates
    12: [0,0,0,1],
    13: [0,0,0,1]
}
pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
indices = [0, 1, 2, 3]

# Input wires for alice and bob
bobInputs = [1, 3, 5]
aliceInputs = [0, 2, 4]

class Alice:
    def __init__(self, bloodtype):
        self.bloodtype = bloodtype

    def garble(self):
        K = [[0,0]] * T
        F = [[0,0,0,0]] * T

        for i in range(0, T):
            K[i] = pick_random_pair()

        for i in range(n, T):
            C = [0, 0, 0, 0]
            kL = K[L(i)]
            kR = K[R(i)]
            for j in range(len(pairs)):
                gate = values[i][j]
                kGate = K[i][gate]
                a, b = pairs[j]
                gL, gR = G(kL[a], kR[b], i)
                C[j] =  (xor(gL, kGate), gR)

            permutation = secrets.choice(list(itertools.permutations(indices)))
            F[i] = [C[j] for j in permutation]
        
        e = K[:n]
        d = K[outputWire]
        return F, e, d
    
    def encode(self, E):
        X = list()
        bits = bloodtypeBits[self.bloodtype]
        for i,j in enumerate(aliceInputs):
            k0, k1 = E[j]
            X.append(k1 if bits[i] else k0)
        return X
    
    def transfer(self, keys, E):
        messages = list()
        for i, key in enumerate(keys):
            k0, k1 = key
            v0, v1 = E[bobInputs[i]]
            m0 = enc(k0, v0)
            m1 = enc(k1, v1)
            messages.append((m0, m1))
        
        return messages

class Bob:
    def __init__(self, bloodtype):
        self.bloodtype = bloodtype
    
    def start_transfer(self):
        pks = list()
        sks = list()

        bits = bloodtypeBits[self.bloodtype]
        for bit in bits:
            sk, pk = gen()
            _, opk = ogen()

            sks.append(sk)
            if bit == 1:
                pks.append((opk, pk))
            else:
                pks.append((pk, opk))

        return pks, sks

    def end_transfer(self, messages, sks):
        bits = bloodtypeBits[self.bloodtype]
        X = list()
        for i, msg in enumerate(messages):
            key = sks[i]
            bit = bits[i]

            m0, m1 = msg
            if bit == 1:
                X.append(dec(key, m1))
            else:
                X.append(dec(key, m0))

        return X
            
    def evaluate(self, F, aliceX, bobX):
        X = [to_bytes(0)] * T
        for i, j in enumerate(aliceInputs):
            X[j] = aliceX[i]

        for i, j in enumerate(bobInputs):
            X[j] = bobX[i]

        for i in range(n, T):
            # From the notes it is not clear whether "Recover (C_1^i, ...) from F" means we have to de-permute or not however, the permutation
            # should not matter at this point (and it does indeed seem to work with and without de-permuting), so we don't (if we did we would
            # need to also keep track of the permutation)
            C = F[i]

            parentL = L(i)
            parentR = R(i)
            kL = X[parentL]
            kR = X[parentR]
            gL, gR = G(kL, kR, i)
            for j in range(4):
                cL, cR = C[j]
                k = xor(gL, cL)
                t = xor(gR, cR)
                if zero(t):
                    X[i] = k
            
            # If everything works and is done correctly this should not happen
            if zero(X[i]):
                print("ABORT SOMETHING IS WRONG WITH WIRE " + str(i) + " !!!")
                input()

        return X[outputWire]
    
    def decode(self, Z, d):
        z0, z1 = d
        if z0 == Z:
            return False
        elif z1 == Z:
            return True
        else:
            # If this happens someone cheated or did something bad!
            print("Something done gone goofed! Someone might be cheating?")
            return False

def protocol(donor, recipient):
    alice = Alice(donor)
    bob = Bob(recipient)

    # Note that F contains some 0 entries for the corresponding input wires, these should remain 0 and not be used,
    # but are kept in F to make indexing into easier, as all indices are defined according to the diagram above.
    F, e, d = alice.garble()

    aliceX = alice.encode(e)

    # Oblivious transfer between alice and bob
    pks, sks = bob.start_transfer()
    messages = alice.transfer(pks, e)
    bobXTransfer = bob.end_transfer(messages, sks)

    Z = bob.evaluate(F, aliceX, bobXTransfer)
    result = bob.decode(Z, d)

    return result

total = 0
correct = 0
for donor in ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]:
    for recipient in ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]:
        total += 1
        correct += protocol(donor, recipient) == truth_table_lookup(donor, recipient)

p = correct / total
print("Agrees with truth table " + str(p * 100) + "% percent of the time")

# Alternatively to check for just one pair use this code:
#donor = "AB+"
#recipient = "O-"

# result = protocol(donor, recipient)
# print("Compatible? " + str(result))

# truth_table = truth_table_lookup(donor, recipient)
# print("Sanity check, truth table says: " + str(truth_table))