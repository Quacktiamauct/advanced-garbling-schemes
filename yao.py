import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from circuits import Circuit, Gate, Operation
from typing import Tuple, List
import secrets
import hashlib

SIZE = 8 * 16  # underlying primitive size k
MAGIC = 2863311530  # 101010101010101010... in binary


def make_bitarray(i: int):
    """
    returns a bitarray representing 'i' of size SIZE
    """
    return int2ba(i, SIZE, endian="little")


def G(A: bitarray, B: bitarray, i: int) -> Tuple[bitarray, bitarray]:
    """
    input: A and B are bitarrays of size SIZE
    returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian="little")
    food = A + B + ia
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray(endian='little')
    arr.frombytes(hash)
    return arr[SIZE:], arr[:SIZE]


def pick_random_pair():
    """
    returns a random pair of bitarrays of size SIZE
    """
    rnd = secrets.token_bytes(32)
    arr = bitarray(endian='little')
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]


class GarbledGate:
    C: List[Tuple[bitarray, bitarray]]
    left: int
    right: int
    output: int

    def __init__(self, gate: Gate):
        self.left = gate.left
        self.right = gate.right
        self.output = gate.output
        self.C = [(make_bitarray(MAGIC), make_bitarray(MAGIC))] * 4

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
    e: List[Tuple[bitarray, bitarray]]
    d: List[Tuple[bitarray, bitarray]]
    num_in_wires: int
    num_out_wires: int

    def __init__(self, circuit: Circuit):
        self.num_inputs = circuit.num_inputs
        self.input_sizes = circuit.input_sizes
        self.output_sizes = circuit.output_sizes
        self.num_wires = circuit.num_wires
        self.num_in_wires = sum(self.input_sizes)
        self.num_out_wires = sum(self.output_sizes)

    def eval(self, *args: bitarray):
        """
        args: unencoded input
        returns: unencoded output
        """
        arg = bitarray(endian="little")
        for a in args:
            arg += a
        return self.decode(self.eval_enc(self.encode(arg)))

    def eval_enc(self, X: List[bitarray]) -> List[bitarray]:
        """
        X is a list of bitarrays of size SIZE (being an encoded)
        returns a list of bitarrays of size SIZE
        """
        wires = [e[0] for e in self.e]  # default all wires to zero
        extra = [
            bitarray(endian="little") for _ in range(self.num_wires - self.num_in_wires)
        ]
        wires.extend(extra)  # extend with rest of wires not used for input
        for i, x in enumerate(X):
            wires[i] = x
        for gg in self.gates:
            found = False
            gL, gR = G(wires[gg.left], wires[gg.right], gg.output)
            for i in range(4):
                cL, cR = gg.C[i]
                k = gL ^ cL
                t = gR ^ cR
                if not t.any():  # check if all zero
                    wires[gg.output] = k
                    found = True
                    break
            if not found:
                raise Exception("Error at gate: " + str(gg))
        for wire in wires:
            if len(wire) == 0:
                print("num of wires:", len(wires))
                raise Exception("Error: wire is empty. All wires has to be evaluated")
        return wires[-sum(self.output_sizes) :]

    def encode(self, xs):
        if len(self.e) != len(xs):
            raise Exception(f"Encoding error: {len(self.e)} != {len(xs)}")
        z = [self.e[i][x] for i, x in enumerate(xs)]
        return z

    def decode(self, zs: List[bitarray]):
        x = [-1] * len(zs)
        for i, z in enumerate(zs):
            Z0, Z1 = self.d[i]
            if z == Z0:
                x[i] = 0
            elif z == Z1:
                x[i] = 1
            else:
                raise Exception("Error at decode, no valid Z")
        return bitarray(x, endian="little")


def garble(c: Circuit) -> GarbledCircuit:
    """
    returns a garbled circuit
    """
    gc = GarbledCircuit(c)
    K = []
    for _ in range(c.num_wires):
        garbled_wire = pick_random_pair()
        K.append(garbled_wire)
    gc.e = K[: sum(gc.input_sizes)]
    gc.d = K[-sum(gc.output_sizes) :]
    gc.gates = []
    # Iterate over all gates
    for gate in c.gates:
        gg = GarbledGate(gate)
        for j, (leftValue, rightValue) in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
            gateValue = gate.op(leftValue, rightValue)
            garbledValue = K[gg.output][gateValue]
            gL, gR = G(K[gg.left][leftValue], K[gg.right][rightValue], gg.output)
            gg.C[j] = gL ^ garbledValue, gR
        gg.permute()

        # do the permutations
        gc.gates.append(gg)
    return gc


if __name__ == "__main__":
    f = open("./bristol/sub64.txt")
    c = Circuit(f.read())
    a = int2ba(7, 64, "little")
    b = int2ba(5, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    print(f"{ba2int(a)} - {ba2int(b)} = {ba2int(res)}")
    f = open("./bristol/mult64.txt")
    c = Circuit(f.read())
    a = int2ba(5, 64, "little")
    b = int2ba(3, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    print(f"{ba2int(a)} * {ba2int(b)} = {ba2int(res)}")
