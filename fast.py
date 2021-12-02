#!/usr/bin/env python3
import itertools
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from circuits import Circuit, Gate, Operation
from typing import Tuple, List
import secrets
import hashlib

SIZE = 8 * 16  # underlying primitive size k
MAGIC = 2863311530  # 101010101010101010... in binary

EXTENDED_SIZE = SIZE + 1

def make_bitarray(i: int):
    """
    returns a bitarray representing 'i' of size SIZE
    """
    return int2ba(i, SIZE, endian="little")

def make_bitarray_with(i: int, b: int):
    r = make_bitarray(i)
    r.append(b)
    return r


def make_bitarray_with_2(i: int, a: int, b: int):
    r = make_bitarray(i)
    r.append(a)
    r.append(b)
    return r


def G(A: bitarray, B: bitarray, i: int) -> Tuple[bitarray, bitarray]:
    """
    input: A and B are bitarrays of size SIZE
    returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian="little")
    food = A + B + ia
    hash = hashlib.sha256(food.tobytes()).digest()
    arr = bitarray()
    arr.frombytes(hash)
    return arr[SIZE:], arr[:SIZE]


def make_bitarray(i: int):
    return int2ba(i, SIZE, endian='little')


def F(left: bitarray, index: int, perm: int) -> bitarray:
    right = make_bitarray(index)
    right.append(perm)

    food = left + right
    arr = bitarray()
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())

    return arr[:EXTENDED_SIZE]

def F2(left: bitarray, index: int, permLeft: int, permRight: int) -> bitarray:
    right = make_bitarray(index)
    right.append(permLeft)
    right.append(permRight)

    food = left + right
    arr = bitarray()
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())

    return arr[:EXTENDED_SIZE]


def pick_random_pair():
    """
    returns a random pair of bitarrays of size SIZE
    """
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]


class GarbledGate:
    C: List
    left: int
    right: int
    output: int
    t: List

    def __init__(self, gate: Gate):
        self.left = gate.left
        self.right = gate.right
        self.output = gate.output
        self.C = []
        self.t = None

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
        signal = [-1] * self.num_wires
        wires = [bitarray(endian='big')] * self.num_wires
        for i, x in enumerate(X):
            wires[i] = x[:SIZE]
            signal[i] = x[SIZE]

        for gate in self.gates:
            l = gate.left
            r = gate.right

            # AND Gate
            if gate.operation == Operation.AND:
                if (signal[l] + signal[r]) == 0:
                    T = make_bitarray_with(0, 0)
                else:
                    T = gate.C[(signal[l] * 2 + signal[r]) - 1]
                key = T ^ F2(wires[l], gate.output, signal[l], signal[r]) ^ F2(wires[r], gate.output, signal[l], signal[r])

                wires[gate.output] = key[:SIZE]
                signal[gate.output] = key[SIZE]
            # XOR Gate
            elif gate.operation == Operation.XOR:
                vLeft = F(wires[l], gate.output, signal[l])[:SIZE]
                vRight = F(wires[r], gate.output, signal[r])[:SIZE]
                if signal[r] == 1:
                    k = vLeft ^ vRight ^ gate.C[0]
                else:
                    k = vLeft ^ vRight

                signal[gate.output] = signal[l] ^ signal[r]
            # NOT Gate
            elif gate.operation == Operation.INV:
                signal[gate.output] = 1 ^ signal[l]
                wires[gate.output] = wires[l]
            # Improved AND Gate
            elif gate.operation == Operation.ANDImproved:
                pass
            # Improved XOR Gate
            elif gate.operation == Operation.XORImproved:
                pass

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

    # 1. Setup input wires
    K = []
    permutation = []
    gc.e = []
    for i in range(c.num_in_wires):
        # a) choose random keys
        k0, k1 = pick_random_pair()
        K.append((k0, k1))

        # b) choose permutation bit
        perm = secrets.randbits(1)
        permutation.append(perm)

        # c) Prepare encoding information
        e0 = k0.copy()
        e0.append(perm)
        e1 = k1.copy()
        e1.append(1 ^ perm)
        gc.e.append((e0, e1))
    
    gc.gates = []
    # Iterate over all gates
    table = {}
    for gate in c.gates:
        garbled = GarbledGate(gate)
        # AND Gate
        if gate.operation == Operation.AND:
            # 1. Compute K0
            print(len(K), len(K[0]), len(permutation), gate.left, gate.right)
            K0 = F2(K[gate.left][permutation[gate.left]], gate.output, 0, 0) ^ F2(K[gate.right][permutation[gate.right]], gate.output, 0, 0)

            # 2. Set output wires and permutaion bit
            if permutation[gate.left] == permutation[gate.right] and permutation[gate.left] == 1:
                k0, _ = pick_random_pair()
                perm = secrets.randbits(1)
                k1 = K0[:SIZE]
            else:
                k0 = K0[:SIZE]
                perm = K0[SIZE]
                k1, _ = pick_random_pair()

            kl0 = k0.copy()
            kl0.append(perm)
            kl1 = k1.copy()
            kl1.append(1 ^ perm)
            k = [kl0, kl1]
            

            # 3. Compute gate ciphertexts
            T1 = F2(K[gate.left][permutation[gate.left]], gate.output, 0, 1) ^ \
                 F2(K[gate.right][1 ^ permutation[gate.right]], gate.output, 0, 1) ^ \
                 k[permutation[gate.left] & (1 ^ permutation[gate.right])]
            T2 = F2(K[gate.left][1 ^ permutation[gate.left]], gate.output, 1, 0) ^ \
                 F2(K[gate.right][permutation[gate.right]], gate.output, 1, 0) ^ \
                 k[(1 ^ permutation[gate.left]) & permutation[gate.right]]
            T3 = F2(K[gate.left][1 ^ permutation[gate.left]], gate.output, 1, 1) ^ \
                 F2(K[gate.right][1 ^ permutation[gate.right]], gate.output, 1, 1) ^ \
                 k[(1 ^ permutation[gate.left]) & (1 ^ permutation[gate.right])]

            # 4. Set the values
            K.append(k)
            permutation.append(perm)
            garbled.C = [T1, T2, T3]
        # XOR Gate
        elif gate.operation == Operation.XOR:
            # 1. Compute permutation bit
            perm = permutation[gate.left] ^ permutation[gate.right]

            # 2. Translate keys for LHS
            ki0 = F(K[gate.left][0], gate.output, permutation[gate.left])[:SIZE]
            ki1 = F(K[gate.left][1], gate.output, 1 ^ permutation[gate.left])[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translate keys for RHS
            if permutation[gate.right] == 0:
                kj0 = F(K[gate.right][0], gate.output, 0)[:SIZE]
                kj1 = kj0 ^ delta
                T = F(K[gate.right][1], gate.output, 1)[:SIZE] ^ kj1
            else:
                kj1 = F(K[gate.right][1], gate.output, 0)[:SIZE]
                kj0 = kj1 ^ delta
                T = F(K[gate.right][0], gate.output, 1)[:SIZE] ^ kj0

            # 5. Compute the output for the wire
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Set values
            K.append([k0, k1])
            permutation.append(perm)
            garbled.C = [T]
        # NOT Gate
        elif gate.operation == Operation.INV:
            permutation.append(permutation[gate.left])
            K.append([K[gate.left][1], K[gate.left][0]])
        # Improved AND Gate
        elif gate.operation == Operation.ANDImproved:
            l = gate.left
            r = gate.right

            # 1. Compute keys
            K0F = F2(K[l][permutation[l]], gate.output, 0, 0) ^ F2(K[r][permutation[r]], 0, 0)
            K1F = F2(K[l][permutation[l]], gate.output, 0, 1) ^ F2(K[r][1 ^ permutation[r]], 0, 1)
            K2F = F2(K[l][1 ^ permutation[l]], gate.output, 1, 0) ^ F2(K[r][permutation[r]], 1, 0)
            K3F = F2(K[l][1 ^ permutation[l]], gate.output, 1, 1) ^ F2(K[r][1 ^ permutation[r]], 1, 1)
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
                m0 ^ permutation,
                m1 ^ permutation,
                m2 ^ permutation,
                m3 ^ permutation,
            ]
            if s == 3:
                T1 = K0 ^ K1
                T2 = K0 ^ K2
                t[3] = m3 ^ (1 ^ permutation)
            elif s == 2:
                T1 = K0 ^ K1
                T2 = K1 ^ K3
                t[2] = m2 ^ (1 ^ permutation)
            elif s == 1:
                T1 = K2 ^ K3
                T2 = K0 ^ K2
                t[1] = m1 ^ (1 ^ permutation)
            else:
                T1 = K2 ^ K3
                T2 = K1 ^ K3
                t[0] = m0 ^ (1 ^ permutation)

            # 6. Set values
            permutation.append(perm)
            K.append([k0, k1])
            garbled.C = [T1, T2]
            garbled.t = t
        # Improved XOR Gate
        elif gate.operation == Operation.XORImproved:
            # 1. Compute permutation bit
            perm = permutation[gate.left] ^ permutation[gate.right]

            # 2. Translate keys for LHS
            ki0 = F(K[gate.left][0], gate.output, permutation[gate.left])[:SIZE]
            ki1 = F(K[gate.left][1], gate.output, 1 ^ permutation[gate.left])[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translate keys for RHS
            if permutation[gate.right] == 0:
                kj0 = K[gate.right][0]
                kj1 = kj0 ^ delta
                T = F(K[gate.right][1], gate.output, 1)[:SIZE] ^ kj1
            else:
                kj1 = K[gate.right][1]
                kj0 = kj1 ^ delta
                T = F(K[gate.right][0], gate.output, 1)[:SIZE] ^ kj0

            # 5. Compute the output for the wire
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Set values
            K.append([k0, k1])
            permutation.append(perm)
            garbled.C = [T]

    # 3. Prepare decoding information
    gc.d = []
    for i in range(c.num_wires - c.num_out_wires, c.num_out_wires):
        gate = table[i]
        perm = permutation[i]
        k0, k1 = K[gate]
        d0 = F(k0, gate, perm)
        d1 = F(k1, gate, 1 ^ perm)
        gc.d.append((d0, d1))

    return gc


if __name__ == "__main__":
    f = open("./bristol/adder64.txt")
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
