import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from circuits import Circuit
import secrets
import hashlib

SIZE = 8*16 # underlying primitive size k

class GarbledCircuit:
    num_inputs: int
    num_wires: int
    input_sizes: list
    output_sizes: list
    wires: list
    table: list
    e: list
    d: list

    def __init__(self):
        pass

    def eval(self, *X) -> bitarray:
        wires = self.wires[:] # copy wires
        wires[len(X):] = X
        for i in range(self.num_inputs, self.num_wires):
            for j in range(4):
                res = G(wires[i][0], wires[i][1], i) ^ self.table[i][j]
                out = res[:SIZE]
                tau = res[SIZE:]
                if not tau.any():
                    wires[i] = out
                    break
                elif j == 3: # we shouldn't get here if everything works
                    raise Exception("No valid wire found")
        return wires[-sum(self.output_sizes)]


def G(A : bitarray, B : bitarray, i : int) -> bitarray:
    """
        input: A and B are bitarrays of size SIZE
        returns a 2*SIZE bitarray
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
    gc.input_sizes = c.input_sizes
    gc.output_sizes = c.output_sizes
    gc.num_wires = c.num_wires
    gc.wires = []
    for i in range(c.num_wires):
        garbled_wire = pick_random_pair()
        gc.wires.append(garbled_wire)
    gc.d = gc.wires[-sum(gc.output_sizes)]
    sum_input_sizes = sum(gc.input_sizes)
    gc.e = gc.wires[:sum_input_sizes]
    gc.table = [[0,0,0,0]] * gc.num_wires
    for gate in c.gates:
        for j, (a,b) in enumerate([[0,0], [0,1], [1,0], [1,1]]):
            left = gate.input_wires[0] # assume two inputs for now
            right = gate.input_wires[1]
            i = gate.output_wires[0] # assume one output for now
            tmp1 = G(gc.wires[left][a], gc.wires[right][b], i)
            tmp2 = gc.wires[i][gate.op(a,b)]
            gc.table[i][j] = (tmp1[:SIZE] ^ tmp2) + tmp1[SIZE:]
        # do the permutations
        permute = secrets.choice(list(itertools.permutations([0,1,2,3])))
        gc.table[i] = [gc.table[i][k] for k in permute]
    return gc

def encode(e, x):
    X = [e[i][x[i]] for i, _ in enumerate(x)]
    return X

def decode(d, Z):
    X = [d[i][Z[i]] for i, _ in enumerate(Z)]
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
    ab = gc.eval(enc_a, enc_b)
    print(ba2int(ab))

