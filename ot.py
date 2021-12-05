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

