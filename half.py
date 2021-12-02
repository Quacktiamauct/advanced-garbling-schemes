import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int, zeros
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
    arr = bitarray(endian="little")
    arr.frombytes(hash)
    return arr[SIZE:], arr[:SIZE]


def pick_random_pair():
    """
    returns a random pair of bitarrays of size SIZE
    """
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]


class GarbledGate:
    left: int
    right: int
    output: int
    F: Tuple[bitarray, bitarray]
    operation: Operation

    def __init__(self, gate: Gate):
        self.left = gate.left
        self.right = gate.right
        self.output = gate.output
        self.operation = gate.operation
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
    e: List[bitarray]
    d: bitarray
    num_in_wires: int
    num_out_wires: int
    R: bitarray

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
        j = 0
        j_prime = 0
        wires = [bitarray()] * c.num_wires
        zero = zeros(SIZE)
        for i, x in enumerate(X):
            wires[i] = x
        for gate in self.gates:
            a = gate.left
            b = gate.right
            i = gate.output
            if gate.operation is Operation.XOR:
                wires[i] = wires[a] ^ wires[b]
            else:
                sa = wires[a][-1]
                sb = wires[b][-1]
                j += 1
                j_prime += 1
                tg, te = gate.F
                wg = H(wires[a], j) ^ (tg if sa else zero)
                we = H(wires[b], j_prime) ^ (te ^ wires[a] if sb else zero)
                wires[i] = wg ^ we
        return wires[-self.num_out_wires :]

    def encode(self, xs):
        if len(self.e) != len(xs):
            raise Exception(f"Encoding error: {len(self.e)} != {len(xs)}")
        zero = zeros(SIZE)
        z = [self.e[i] ^ (self.R if x else zero) for i, x in enumerate(xs)]
        return z

    def decode(self, zs: List[bitarray]):
        if len(self.d) != len(zs):
            raise Exception(f"Decoding error: {len(self.d)} != {len(zs)}")
        x = [d ^ zs[i][-1] for i, d in enumerate(self.d)]
        # return self.d ^ zs[:][-1]
        return bitarray(x, endian="little")


def rnd_bitarray(size):
    i = secrets.randbits(size)
    return int2ba(i, size)


def H(A: bitarray, i: int):
    """
    input: A and B are bitarrays of size SIZE
    returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian="little")
    food = A + ia
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray()
    arr.frombytes(hash)
    return arr[:SIZE]


def garble(c: Circuit) -> GarbledCircuit:
    """
    returns a garbled circuit
    """
    gc = GarbledCircuit(c)
    gc.gates = []
    R = rnd_bitarray(SIZE)
    R[-1] = 1
    gc.R = R
    W = [[bitarray(), bitarray()]] * c.num_wires
    for i in range(c.num_in_wires):
        W[i][0] = rnd_bitarray(SIZE) # FIX: lack of randomness
        W[i][1] = W[i][0] ^ R

    for w in W[:c.num_in_wires]:
        print(w[0])
        print(w[1])
    gc.e = [w[0] for w in W[: c.num_in_wires]]
    j = 0
    j_prime = 0
    # Iterate over all gates
    zero = zeros(SIZE)
    for gate in c.gates:
        garbled = GarbledGate(gate)
        i = gate.output
        a = gate.left
        b = gate.right
        if gate.operation == Operation.XOR:
            W[i][0] = W[a][0] ^ W[b][0]
        else:  # TODO: NEG gate?
            p_a = bitarray(W[a][0][-1])  # lsb
            p_b = bitarray(W[b][0][-1])  # lsb
            j += 1  # TODO: NextIndex? find out what that actually is
            j_prime += 1
            # first half-gate
            tg = H(W[a][0], j) ^ H(W[a][1], j) ^ (R if p_b else zero)
            wg = H(W[a][0], j) ^ (tg if p_a else zero)
            # second half-gate
            te = H(W[b][0], j) ^ H(W[b][1], j_prime) ^ W[a][0]
            we = H(W[b][0], j) ^ (te ^ W[a][0] if p_b else zero)
            # combine halves
            W[i][0] = wg ^ we
            garbled.F = tg, te
        gc.gates.append(garbled)
    gc.d = bitarray([w[0][-1] for w in W[-c.num_out_wires :]])
    return gc


if __name__ == "__main__":
    f = open("./bristol/adder64.txt")
    c = Circuit(f.read())
    a = int2ba(7, 64, "little")
    b = int2ba(5, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    print(f"{ba2int(a)} + {ba2int(b)} = {ba2int(res)}")
    # f = open("./bristol/mult64.txt")
    # c = Circuit(f.read())
    # a = int2ba(5, 64, "little")
    # b = int2ba(3, 64, "little")
    # gc = garble(c)
    # res = gc.eval(a, b)
    # print(f"{ba2int(a)} * {ba2int(b)} = {ba2int(res)}")
