import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int, zeros
from circuits import Circuit, Gate, Operation
from typing import Tuple, List
import secrets
import hashlib

SIZE = 128
MAGIC = 0
LSB_INDEX = 0

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
        wires = [bitarray()] * c.num_wires
        zero = zeros(SIZE, endian='little')

        # Inputs
        for i, x in enumerate(X):
            wires[i] = x

        # Gates
        j = 0
        jp = 0
        for gate in self.gates:
            a = gate.left
            b = gate.right
            i = gate.output

            # XOR
            if gate.operation is Operation.XOR:
                wires[i] = wires[a] ^ wires[b]
            # AND
            else:
                sa = wires[a][LSB_INDEX]
                sb = wires[b][LSB_INDEX]

                j += 1
                jp += 1

                tg, te = gate.F
                wg = H(wires[a], j) ^ (tg if sa else zero)
                we = H(wires[b], jp) ^ (te ^ wires[a] if sb else zero)

                wires[i] = wg ^ we

        return wires[-self.num_out_wires:]

    def encode(self, xs):
        if len(self.e) != len(xs):
            raise Exception(f"Encoding error: {len(self.e)} != {len(xs)}")

        zero = zeros(SIZE, endian='little')
        z = [self.e[i] ^ (self.R if x else zero) for i, x in enumerate(xs)]
        return z

    def decode(self, zs: List[bitarray]):
        if len(self.d) != len(zs):
            raise Exception(f"Decoding error: {len(self.d)} != {len(zs)}")

        x = [d ^ zs[i][LSB_INDEX] for i, d in enumerate(self.d)]
        return bitarray(x, endian="little")


def rnd_bitarray(size):
    i = secrets.randbits(size)
    return int2ba(i, size, endian='little')


def H(A: bitarray, i: int):
    """
    input: A and B are bitarrays of size SIZE
    returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian="little")
    food = A + ia
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray(endian='little')
    arr.frombytes(hash)
    return arr[:SIZE]


def garble(c: Circuit) -> GarbledCircuit:
    """
    returns a garbled circuit
    """
    gc = GarbledCircuit(c)
    gc.gates = []

    # Setup R
    R = rnd_bitarray(SIZE)
    R[LSB_INDEX] = 1
    gc.R = R

    # Inputs
    W = [[bitarray(), bitarray()]] * c.num_wires
    for i in range(c.num_in_wires):
        W[i][0] = rnd_bitarray(SIZE) # TODO: FIX: lack of randomness
        W[i][1] = W[i][0] ^ R
    gc.e = [w[0] for w in W[: c.num_in_wires]]

    # Iterate over all gates
    zero = zeros(SIZE, endian='little')
    j = 0
    jp = 0
    for gate in c.gates:
        garbled = GarbledGate(gate)

        i = gate.output
        a = gate.left
        b = gate.right

        # XOR
        if gate.operation == Operation.XOR:
            W[i][0] = W[a][0] ^ W[b][0]
        # AND
        else:
            p_a = bitarray(W[a][0][LSB_INDEX])
            p_b = bitarray(W[b][0][LSB_INDEX])

            j += 1
            jp += 1

            # first half-gate
            tg = H(W[a][0], j) ^ H(W[a][1], j) ^ (R if p_b else zero)
            wg = H(W[a][0], j) ^ (tg if p_a else zero)

            # second half-gate
            te = H(W[b][0], jp) ^ H(W[b][1], jp) ^ W[a][0]
            we = H(W[b][0], jp) ^ (te ^ W[a][0] if p_b else zero)

            # combine halves
            W[i][0] = wg ^ we
            garbled.F = tg, te

        W[i][1] = W[i][0] ^ R
        gc.gates.append(garbled)

    # Outputs
    gc.d = bitarray([w[0][LSB_INDEX] for w in W[-c.num_out_wires:]])

    return gc


if __name__ == "__main__":
    num1 = 64 + 4865
    num2 = 64 + 123
    a = int2ba(num1, 64, "little")
    b = int2ba(num2, 64, "little")
    expected = int2ba(num1 + num2, 64, "little")
    failed = 0
    passed = 0
    for something in range(100):
        f = open("./bristol/adder64.txt")
        c = Circuit(f.read())
        gc = garble(c)
        res = gc.eval(a, b)

        delta = res ^ expected
        if expected == res:
            passed += 1
            if delta[0] == 0:
                print("\33[32mPASS\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[32mPASS\33[0m -- \33[32m LSB FAIL\33[0m")
        else:
            failed += 1

            print("a         = " + str(a) + " (" + str(ba2int(a)) + ")")
            print("b         = " + str(b) + " (" + str(ba2int(b)) + ")")
            print("res       = " + str(res) + " (" + str(ba2int(res)) + ")")
            print("expecting = " + str(expected) + " (" + str(ba2int(expected)) + ")")
            print("delta     = " + str(delta) + " (" + str(ba2int(delta)) + ")")

            if delta[0] == 0:
                print("\33[31mFAIL\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[31mFAIL\33[0m -- \33[32m LSB FAIL\33[0m")

    print("")

    total = failed + passed
    if passed == total:
        print("\33[32mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    elif passed == 0:
        print("\33[31mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    else:
        print("\33[33mPasses " + str(passed) + " / " + str(total) + "\33[0m")

    if True:
        print("\33[34m--------------------------------------------------\33[0m")
        # XOR Gate
        f = open("./bristol/simple_xor.txt")
        c = Circuit(f.read())

        xorPass = 0
        xorFail = 0
        for num in range(2500):
            gc = garble(c)
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                res = gc.eval(int2ba(a, 1, "little"), int2ba(b, 1, "little"))
                r = ba2int(res)
                if r == (a ^ b):
                    xorPass += 1
                else:
                    xorFail += 1

        xorTotal = xorFail + xorPass
        if xorTotal == xorPass:
            print("\33[32mXOR PASS\33[0m --> " + str(xorPass) + " / " + str(xorTotal))
        else:
            print("\33[31mXOR FAIL\33[0m --> " + str(xorPass) + " / " + str(xorTotal))

        # AND Gate
        f = open("./bristol/simple_and.txt")
        c = Circuit(f.read())

        andPass = 0
        andFail = 0
        for num in range(2500):
            gc = garble(c)
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                res = gc.eval(int2ba(a, 1, "little"), int2ba(b, 1, "little"))
                r = ba2int(res)
                if r == (a & b):
                    andPass += 1
                else:
                    andFail += 1

        andTotal = andFail + andPass
        if xorTotal == andPass:
            print("\33[32mAND PASS\33[0m --> " + str(andPass) + " / " + str(andTotal))
        else:
            print("\33[31mAND FAIL\33[0m --> " + str(andPass) + " / " + str(andTotal))