import itertools
from bitarray import bitarray
from bitarray.util import int2ba
from circuits import Circuit, Gate, Operation
from typing import Tuple, List
import secrets
import hashlib

# Size of the underlying primitive (i.e. no of bits used, n)
SIZE = 128
EXTENDED_SIZE = SIZE + 1

# "Magic" number used to initialize bitarrays before use (easy to recoqnize)
MAGIC = 0


def F(left: bitarray, index: int, bit: int) -> bitarray:
    """
    F is our pseudo random function, here backed by SHA256 it takes 3 parameters:
    left is an array of bits, index and perm are an index and permutation bits which are combined
    into a bitarray, both of these are then passed to the hash function
    """
    right = int2ba(index, SIZE, endian="little")
    right.append(bit)

    food = left + right
    arr = bitarray(endian="little")
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())

    return arr[:EXTENDED_SIZE]


def F2(left: bitarray, index: int, bit1: int, bit2: int) -> bitarray:
    """
    F2 does the same as F, but is used as a helper by the AND gates as their input depends on not only some bitarray
    and index of the gate/output wire, but also two additional bytes as opposed to one.
    """
    right = int2ba(index, SIZE, endian="little")
    right.append(bit1)
    right.append(bit2)

    food = left + right
    arr = bitarray(endian="little")
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())

    return arr[:EXTENDED_SIZE]


def pick_random_pair():
    """
    returns a random pair of bitarrays of size SIZE
    """
    rnd = secrets.token_bytes(SIZE)
    arr = bitarray(endian="little")
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:(SIZE + SIZE)]


def pick_random():
    """
    returns a random bitarray of size SIZE
    """
    rnd = secrets.token_bytes(SIZE // 2)
    arr = bitarray(endian="little")
    arr.frombytes(rnd)
    return arr[:SIZE]


class GarbledGate:
    """
    Class representing a gate after it has been garbled, keeps track of left/right input indices, output index, operator,
    and the truth table/auxiliary information generated during garbling (which is required for evaluation)
    """
    C: List
    left: int
    right: int
    output: int
    t: List
    operation: Operation

    def __init__(self, gate: Gate):
        self.left = gate.left
        self.right = gate.right
        self.output = gate.output
        self.C = []
        self.t = None
        self.operation = gate.operation

    def __str__(self):
        return f"{self.left} {self.right} {self.output}"


class GarbledCircuit:
    """
    Class representing a garbled circuit, keeps track of all the gates, wires and intermediate information of the
    garbled circuit.
    """
    num_inputs: int
    num_wires: int
    input_sizes: list
    output_sizes: list
    gates: List[GarbledGate]
    e: List[Tuple[bitarray, bitarray]]
    d: List[Tuple[bitarray, bitarray]]
    num_in_wires: int
    num_out_wires: int

    def __init__(self, circuit: Circuit, improvedAnd: bool, improvedXor: bool):
        self.num_inputs = circuit.num_inputs
        self.input_sizes = circuit.input_sizes
        self.output_sizes = circuit.output_sizes
        self.num_wires = circuit.num_wires
        self.num_in_wires = sum(self.input_sizes)
        self.num_out_wires = sum(self.output_sizes)
        self.improvedAnd = improvedAnd
        self.improvedXor = improvedXor

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
        signal = [-1] * self.num_wires
        wires = [bitarray(endian="big")] * self.num_wires
        for i, x in enumerate(X):
            wires[i] = x[:SIZE]
            signal[i] = x[SIZE]

        for gate in self.gates:
            l = gate.left
            r = gate.right

            # AND Gate
            if gate.operation == Operation.AND and not self.improvedAnd:
                if (signal[l] + signal[r]) == 0:
                    key = F2(wires[l], gate.output, signal[l], signal[r]) ^ F2(wires[r], gate.output, signal[l], signal[r])
                else:
                    T = gate.C[(signal[l] * 2 + signal[r]) - 1]
                    key = T ^ F2(wires[l], gate.output, signal[l], signal[r]) ^ F2(wires[r], gate.output, signal[l], signal[r])

                wires[gate.output] = key[:SIZE]
                signal[gate.output] = key[SIZE]
            # XOR Gate
            elif gate.operation == Operation.XOR and not self.improvedXor:
                vLeft = F(wires[l], gate.output, signal[l])[:SIZE]
                vRight = F(wires[r], gate.output, signal[r])[:SIZE]
                if signal[r] == 1:
                    k = vLeft ^ vRight ^ gate.C[0]
                else:
                    k = vLeft ^ vRight

                signal[gate.output] = signal[l] ^ signal[r]
                wires[gate.output] = k
            # NOT Gate
            elif gate.operation == Operation.INV:
                signal[gate.output] = 1 ^ signal[l]
                wires[gate.output] = wires[l]
            # Improved AND Gate
            elif gate.operation == Operation.AND and self.improvedAnd:
                KF = F2(wires[l], gate.output, signal[l], signal[r]) ^ F2(wires[r], gate.output, signal[l], signal[r])
                k, m = [KF[:SIZE], KF[SIZE]]

                signals = 2 * signal[l] + signal[r]

                if signals == 0:
                    wires[gate.output] = k
                elif signals == 1:
                    wires[gate.output] = k ^ gate.C[0]
                elif signals == 2:
                    wires[gate.output] = k ^ gate.C[1]
                else:
                    wires[gate.output] = k ^ gate.C[0] ^ gate.C[1]
                signal[gate.output] = m ^ gate.t[signals]
            # Improved XOR Gate
            elif gate.operation == Operation.XOR and self.improvedXor:
                if signal[r] == 0:
                    k = F(wires[l], gate.output, signal[l])[:SIZE] ^ wires[r]
                else:
                    vLeft = F(wires[l], gate.output, signal[l])[:SIZE]
                    vRight = F(wires[r], gate.output, 1)[:SIZE]
                    k = vLeft ^ vRight ^ gate.C[0]

                wires[gate.output] = k
                signal[gate.output] = signal[l] ^ signal[r]

        # Run outputs through F
        for i in range(self.num_wires - self.num_out_wires, self.num_wires):
            wires[i] = F(wires[i], i, signal[i])

        for i, wire in enumerate(wires):
            if len(wire) == 0:
                print("num of wires:", len(wires))
                raise Exception(f"Error: wire {i} is empty. All wires have to be evaluated")

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


def garble(c: Circuit, improvedAnd = True, improvedXor = True) -> GarbledCircuit:
    """
    returns a garbled circuit
    """
    gc = GarbledCircuit(c, improvedAnd, improvedXor)

    # 1. Setup input wires
    K = [[]] * c.num_wires
    permutation = [-1] * c.num_wires
    gc.e = []
    for i in range(c.num_in_wires):
        # a) choose random keys
        k0, k1 = pick_random_pair()
        K[i] = k0, k1

        # b) choose permutation bit
        perm = secrets.randbits(1)
        permutation[i] = perm

        # c) Prepare encoding information
        e0 = k0.copy()
        e0.append(perm)
        e1 = k1.copy()
        e1.append(1 ^ perm)
        gc.e.append((e0, e1))

    gc.gates = []
    # Iterate over all gates
    for gate in c.gates:
        garbled = GarbledGate(gate)
        l = gate.left
        r = gate.right

        # AND Gate
        if gate.operation == Operation.AND and not improvedAnd:
            # 1. Compute K0
            K0 = F2(K[l][permutation[l]], gate.output, 0, 0) ^ F2(K[r][permutation[r]], gate.output, 0, 0)

            # 2. Set output wires and permutation bit
            if permutation[l] == permutation[r] and permutation[l] == 1:
                k0 = pick_random()
                perm = 1 ^ K0[SIZE]
                k1 = K0[:SIZE]
            else:
                k0 = K0[:SIZE]
                perm = K0[SIZE]
                k1 = pick_random()

            kl0 = k0.copy()
            kl0.append(perm)
            kl1 = k1.copy()
            kl1.append(1 ^ perm)
            k = [kl0, kl1]

            # 3. Compute gate ciphertexts
            T1 = (F2(K[l][permutation[l]], gate.output, 0, 1)     ^ F2(K[r][1 ^ permutation[r]], gate.output, 0, 1) ^ k[permutation[l]       & (1 ^ permutation[r])])
            T2 = (F2(K[l][1 ^ permutation[l]], gate.output, 1, 0) ^ F2(K[r][permutation[r]], gate.output, 1, 0)     ^ k[(1 ^ permutation[l]) & permutation[r]])
            T3 = (F2(K[l][1 ^ permutation[l]], gate.output, 1, 1) ^ F2(K[r][1 ^ permutation[r]], gate.output, 1, 1) ^ k[(1 ^ permutation[l]) & (1 ^ permutation[r])])

            # 4. Set the values
            K[gate.output] = [k0, k1]
            permutation[gate.output] = perm
            garbled.C = [T1, T2, T3]
        # XOR Gate
        elif gate.operation == Operation.XOR and not improvedXor:
            # 1. Compute permutation bit
            perm = permutation[l] ^ permutation[r]

            # 2. Translate keys for LHS
            ki0 = F(K[l][0], gate.output, permutation[l])[:SIZE]
            ki1 = F(K[l][1], gate.output, 1 ^ permutation[l])[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translate keys for RHS
            if permutation[r] == 0:
                kj0 = F(K[r][0], gate.output, 0)[:SIZE]
                kj1 = kj0 ^ delta
                T = F(K[r][1], gate.output, 1)[:SIZE] ^ kj1
            else:
                kj1 = F(K[r][1], gate.output, 0)[:SIZE]
                kj0 = kj1 ^ delta
                T = F(K[r][0], gate.output, 1)[:SIZE] ^ kj0

            # 5. Compute the output for the wire
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Set values
            K[gate.output] = [k0, k1]
            permutation[gate.output] = perm
            garbled.C = [T]
        # NOT Gate
        elif gate.operation == Operation.INV:
            permutation[gate.output] = permutation[l]
            K[gate.output] = [K[l][1], K[l][0]]
        # Improved AND Gate
        elif gate.operation == Operation.AND and improvedAnd:
            l = l
            r = r

            # 1. Compute keys
            K0F = F2(K[l][    permutation[l]], gate.output, 0, 0) ^ F2(K[r][    permutation[r]], gate.output, 0, 0)
            K1F = F2(K[l][    permutation[l]], gate.output, 0, 1) ^ F2(K[r][1 ^ permutation[r]], gate.output, 0, 1)
            K2F = F2(K[l][1 ^ permutation[l]], gate.output, 1, 0) ^ F2(K[r][    permutation[r]], gate.output, 1, 0)
            K3F = F2(K[l][1 ^ permutation[l]], gate.output, 1, 1) ^ F2(K[r][1 ^ permutation[r]], gate.output, 1, 1)
            K0, m0 = [K0F[:SIZE], K0F[SIZE]]
            K1, m1 = [K1F[:SIZE], K1F[SIZE]]
            K2, m2 = [K2F[:SIZE], K2F[SIZE]]
            K3, m3 = [K3F[:SIZE], K3F[SIZE]]

            # 2. Compute location of one in the truth table
            s = 2 * (1 ^ permutation[l]) + (1 ^ permutation[r])

            # 3. Output wires and permutation bits
            perm = secrets.randbits(1)
            if s != 0:
                k0 = K0
                k1 = K1 ^ K2 ^ K3
            else:
                k0 = K1 ^ K2 ^ K3
                k1 = K0

            # 4/5. Compute table and additional bits
            t = [
                m0 ^ perm,
                m1 ^ perm,
                m2 ^ perm,
                m3 ^ perm,
            ]
            if s == 3:
                T1 = K0 ^ K1
                T2 = K0 ^ K2
                t[3] = m3 ^ (1 ^ perm)
            elif s == 2:
                T1 = K0 ^ K1
                T2 = K1 ^ K3
                t[2] = m2 ^ (1 ^ perm)
            elif s == 1:
                T1 = K2 ^ K3
                T2 = K0 ^ K2
                t[1] = m1 ^ (1 ^ perm)
            else:
                T1 = K2 ^ K3
                T2 = K1 ^ K3
                t[0] = m0 ^ (1 ^ perm)

            # 6. Set values
            permutation[gate.output] = perm
            K[gate.output] = [k0, k1]
            garbled.C = [T1, T2]
            garbled.t = t
        # Improved XOR Gate
        elif gate.operation == Operation.XOR and improvedXor:
            # 1. Compute permutation bit
            perm = permutation[l] ^ permutation[r]

            # 2. Translate keys for LHS
            ki0 = F(K[l][0], gate.output,     permutation[l])[:SIZE]
            ki1 = F(K[l][1], gate.output, 1 ^ permutation[l])[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translate keys for RHS
            if permutation[r] == 0:
                kj0 = K[r][0]
                kj1 = kj0 ^ delta
                T = F(K[r][1], gate.output, 1)[:SIZE] ^ kj1
            else:
                kj1 = K[r][1]
                kj0 = kj1 ^ delta
                T = F(K[r][0], gate.output, 1)[:SIZE] ^ kj0

            # 5. Compute the output for the wire
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Set values
            K[gate.output] = [k0, k1]
            permutation[gate.output] = perm
            garbled.C = [T]

        gc.gates.append(garbled)

    # 3. Prepare decoding information
    gc.d = []
    for i in range(c.num_wires - c.num_out_wires, c.num_wires):
        perm = permutation[i]
        k0, k1 = K[i]
        d0 = F(k0, i, perm)
        d1 = F(k1, i, 1 ^ perm)
        gc.d.append((d0, d1))

    return gc
