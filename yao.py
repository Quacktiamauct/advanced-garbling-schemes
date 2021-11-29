import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from circuits import Circuit, Gate, Operation
from typing import Tuple, List
import secrets
import hashlib

SIZE = 8 * 16  # underlying primitive size k
MAGIC = 2863311530 # 101010101010101010... in binary


def make_bitarray(i: int):
    return int2ba(i, SIZE, endian='little')

def G(A: bitarray, B: bitarray, i: int) -> Tuple[bitarray, bitarray]:
    """
        input: A and B are bitarrays of size SIZE
        returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian='little')
    food = A + B + ia
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray()
    arr.frombytes(hash)
    return arr[SIZE:], arr[:SIZE]


def pick_random_pair():
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]

class GarbledGate:
    C: List[Tuple[bitarray, bitarray]]
    # K: Tuple[bitarray, bitarray]
    left: int
    right: int
    output: int

    def __init__(self, gate : Gate):
        self.left = gate.left
        self.right = gate.right
        self.output = gate.output
        self.C = [(make_bitarray(MAGIC), make_bitarray(MAGIC))] * 4
        # self.K = make_bitarray(MAGIC), make_bitarray(MAGIC)

    def permute(self):
        permute = secrets.choice(list(itertools.permutations([0, 1, 2, 3])))
        self.C = [self.C[k] for k in permute]

    def __str__(self):
        return f"{self.left} {self.right} {self.output}"


class GarbledCircuit:
    num_inputs: int
    num_wires: int
    input_sizes: list
    output_sizes: list
    gates: List[GarbledGate]
    K: List[Tuple[bitarray, bitarray]]
    e: List[Tuple[bitarray, bitarray]]
    d: List[Tuple[bitarray, bitarray]]

    def __init__(self, circuit : Circuit):
        self.num_inputs = circuit.num_inputs
        self.input_sizes = circuit.input_sizes
        self.output_sizes = circuit.output_sizes
        self.num_wires = circuit.num_wires
        self.K = []
        for _ in range(c.num_wires):
            garbled_wire = pick_random_pair()
            self.K.append(garbled_wire)
        self.e = self.K[:sum(self.input_sizes)]
        self.d = self.K[-sum(self.output_sizes):]
        self.gates = []

    def eval(self, *X) -> bitarray:
        args = []
        for x in X:
            args.extend(x)
        wires = [bitarray()] * self.num_wires
        for i, a in enumerate(args):
            wires[i] = a
        for gg in self.gates:
            found = False
            gL, gR = G(wires[gg.left], wires[gg.right], gg.output)
            for i in range(4):
                cL, cR = gg.C[i]
                k = gL ^ cL
                t = gR ^ cR
                if not t.any(): # check if all zero
                    print(f"gate {gg} works")
                    wires[gg.output] = k
                    found = True
                    break
            if not found:
                raise Exception("Error at gate: " + str(gg))
        return wires[-sum(self.output_sizes)]



def garble(c: Circuit) -> GarbledCircuit:
    gc = GarbledCircuit(c)
    # Iterate over all gates
    for gate in c.gates:
        gg = GarbledGate(gate)
        for j, (leftValue, rightValue) in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
            gateValue = gate.op(leftValue, rightValue)
            gL, gR = G(gc.K[gg.left][leftValue], gc.K[gg.right][rightValue], gg.output)
            garbledValue = gc.K[gg.output][gateValue]
            gg.C[j] = gL ^ garbledValue, gR
        # do the permutations
        gg.permute()
        gc.gates.append(gg)
    return gc


def encode(e, xs):
    X = [ e[i][x] for i, x in enumerate(xs) ]
    return X


def decode(d, Z : List[bitarray]):
    x = [-1] * len(Z)
    for i, z in enumerate(Z):
        Z0, Z1 = d[i]
        if z == Z0:
            x[i] = 0
        elif z == Z1:
            x[i] = 1
        else:
            raise Exception("Error at decode")
    return bitarray(x)


if __name__ == "__main__":
    # f = open("./adder64.txt")
    # raw = f.read()
    # c = Circuit(raw)
    # gc = garble(c)
    # a = int2ba(7, 64, endian='little')
    # b = int2ba(5, 64, endian='little')
    # e = gc.e
    # enc_a = encode(e, a)
    # enc_b = encode(e, b)
    # ab = gc.eval(enc_a, enc_b)
    # dec_ab = decode(gc.d, ab)
    # print(ba2int(dec_ab))
    andGate = Gate(Operation.AND, 0, 1, 4)
    xorGate = Gate(Operation.XOR, 2, 3, 5)
    orGate = Gate(Operation.OR, 4, 5, 6)
    steps = [andGate, xorGate, orGate]
    c = Circuit(1, [4], 1, [1], 7, steps)
    res = c.eval(bitarray([1,0,1,1]))
    print(res)
    gc = garble(c)
    e = gc.e
    res = gc.eval(encode(e, bitarray([1,0,1,1])))
    res_dec = decode(gc.d, [res])
    print(res_dec)
