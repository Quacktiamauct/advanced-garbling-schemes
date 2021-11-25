import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from circuits import Circuit
import secrets
import hashlib

SIZE = 8*16

class GarbledCircuit:
    num_inputs = 0
    num_wires = 0
    wires = []
    table = [bitarray()]
    e = []
    d = []

    def __init__(self):
        pass

    def eval(self, X):
        wires = self.wires[:] # copy wires
        wires[:self.num_inputs] = X
        for i in range(self.num_inputs, self.num_wires):
            for j in range(4):
                res = G(wires[0], wires[1], i) ^ self.table[j]
                out = res[:SIZE]
                tau = res[SIZE:]
                if not tau.any():
                    wires[i] = out
                    break
                elif j == 3: # we shouldn't get here if everything works
                    raise Exception("No valid wire found")


def G(A : bitarray, B : bitarray, i : int):
    """
        returns a 256 bitarray of with the left and right halves of the garbled wire
    """
    num = int2ba(i, 64, endian='little')
    food = A + B + num
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray()
    arr.frombytes(hash)
    return arr

def pick_random_pair():
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]


def garble(c : Circuit) -> GarbledCircuit:
    gc = GarbledCircuit()
    gc.num_inputs = c.num_inputs
    gc.num_wires = c.num_wires
    gc.wires = []
    for i in range(c.num_wires):
        garbled_wire = pick_random_pair()
        gc.wires.append(garbled_wire)
    gc.d = gc.wires[-1]
    gc.e = gc.wires[:c.num_inputs]
    gc.table = [[0,0,0,0]] * (c.num_wires - c.num_inputs)
    for i in range(c.num_inputs, c.num_wires):
        for j, (a,b) in enumerate([[0,0], [0,1], [1,0], [1,1]]):
            wire = gc.wires[i]
            print(c.gates[i]) # TODO: Find out what to actually use here
            outwire = c.gates[i].op(a,b)
            g = G(wire[a], wire[b], i)
            gc.table[i][j] = g[:SIZE] ^ gc.wires[i][outwire], g[SIZE:]
        permute = secrets.choice(list(itertools.permutations([0,1,2,3])))
        gc.table[i] = [gc.table[i][k] for k in permute]
    return gc

def encode(e, x):
    X = [e[i][x[i]] for i, _ in enumerate(x)]
    return bitarray(X)


if __name__ == "__main__":
    f = open("./adder64.txt")
    raw = f.read()
    c = Circuit(raw)
    gc = garble(c)
    a = int2ba(5, 64, endian='little')
    b = int2ba(7, 64, endian='little')
    e = gc.e
    enc_a = encode(e, a)
    enc_b = encode(e, b)
    ab = c.eval(enc_a, enc_b)
    print(ba2int(ab))

