import itertools
from bitarray import bitarray
from circuits import Circuit
import secrets
import hashlib

class GarbledCircuit:
    num_inputs = 0
    num_wires = 0
    wires = []
    table = [[]]
    e = []
    d = []

    def __init__(self):
        pass

    def eval(self, X):
        wires = self.wires[:]
        wires[:self.num_inputs] = X


def to_bytes(num):
    return int.to_bytes(num, (num.bit_length() + 7), byteorder='big', signed=False)

def G(A, B, i):
    hash = hashlib.sha256(A + B + to_bytes(i)).digest()
    arr = bitarray()
    arr.frombytes(hash)
    left = arr[:16*8]
    right = arr[8*16:]
    return left, right

def pick_random_pair():
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    left = rnd[:16*8]
    right = rnd[8*16:]
    return left, right


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
            outwire = c.gates[i].op(a,b)
            gleft, gright = G(wire[a], wire[b], i)
            gc.table[i][j] = gleft ^ gc.wires[i][outwire], gright
        permute = secrets.choice(list(itertools.permutations([0,1,2,3])))
        gc.table[i] = gc.table[i][permute]
    return gc

def encode(e, x):
    X = [e[i][x[i]] for i, _ in enumerate(x)]
    return bitarray(X)
