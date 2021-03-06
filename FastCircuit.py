import itertools
import secrets
import hashlib

from typing import Tuple, List, Dict
from enum import Enum
from bitarray import bitarray
from bitarray.util import int2ba, ba2int


class GateType(Enum):
    NOT = 0,
    AND = 1,
    XOR = 2,
    OR = 3,
    NAND = 4,
    XORImproved = 5,
    ANDImproved = 6


#SIZE = 128
#MAGIC = 2863311530 # 101010101010101010... in binary

SIZE = 8
MAGIC = 0
EXTENDED_SIZE = SIZE + 1


def make_bitarray(i: int):
    return int2ba(i, SIZE, endian='little')


def make_bitarray_with(i: int, b: int):
    r = make_bitarray(i)
    r.append(b)
    return r


def make_bitarray_with_2(i: int, a: int, b: int):
    r = make_bitarray(i)
    r.append(a)
    r.append(b)
    return r


def zero(b: bitarray):
    for bit in b:
        if bit == 1:
            return False
    return True


def pick_random_pair():
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:(SIZE + SIZE)]


def F(left: bitarray, right: bitarray) -> bitarray:
    food = left + right
    arr = bitarray()
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())

    print("F(...) invoked:")
    print("<- " + str(left))
    print("<- " + str(right))
    print("-> " + str(arr[:EXTENDED_SIZE]))

    return arr[:EXTENDED_SIZE]


class CircuitGate:
    def __init__(self, index: int):
        self.Index = index
        self.K = [make_bitarray(MAGIC), make_bitarray(MAGIC)]
        self.Permutation = -1
        self.Signal = -1
        self.Output = make_bitarray(MAGIC)

    def Garble(self):
        return


class GarbledGate(CircuitGate):
    Op: GateType
    Left: CircuitGate
    Right: CircuitGate
    C: List[Tuple[bitarray, bitarray]]

    TXor: bitarray

    def __init__(self, index: int, op: GateType, left: CircuitGate, right: CircuitGate):
        super().__init__(index)
        self.Op = op
        self.Left = left
        self.Right = right
        self.C = [(make_bitarray(MAGIC), make_bitarray(MAGIC))] * 4
        self.TXor = make_bitarray(MAGIC)
        self.TAnd = [make_bitarray(MAGIC), make_bitarray(MAGIC), make_bitarray(MAGIC)]

    def Compute(self, left, right):
        if self.Op == GateType.NOT:
            return left ^ 1
        if self.Op == GateType.AND:
            return left & right
        if self.Op == GateType.XOR:
            return left ^ right
        if self.Op == GateType.OR:
            return left | right
        if self.Op == GateType.NAND:
            return (left & right) ^ 1

    def index(self, a: int) -> bitarray:
        return make_bitarray_with(self.Index, a)

    def index2(self, a: int, b: int) -> bitarray:
        return make_bitarray_with_2(self.Index, a, b)

    def Garble(self):
        super(GarbledGate, self).Garble()
        # i = left, j = right, l = this gate

        # XOR Gates
        if self.Op == GateType.XOR:
            # 1. Output permutation bit
            permutation = self.Left.Permutation ^ self.Right.Permutation

            # 2. Translated keys
            ki0 = F(self.Left.K[0], make_bitarray_with(self.Index, self.Left.Permutation))[:SIZE]
            ki1 = F(self.Left.K[1], make_bitarray_with(self.Index, 1 ^ self.Left.Permutation))[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translated keys for j (right)
            if self.Right.Permutation == 0:
                kj0 = F(self.Right.K[0], make_bitarray_with(self.Index, 0))[:SIZE]
                kj1 = kj0 ^ delta
                T = F(self.Right.K[1], make_bitarray_with(self.Index, 1))[:SIZE] ^ kj1
            else:
                kj1 = F(self.Right.K[1], make_bitarray_with(self.Index, 0))[:SIZE]
                kj0 = kj1 ^ delta
                T = F(self.Right.K[0], make_bitarray_with(self.Index, 1))[:SIZE] ^ kj0

            # 5. Compute keys for output wire l
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Do something with the result
            self.K = [k0, k1]
            self.Permutation = permutation
            self.TXor = T

        # AND Gates
        elif self.Op == GateType.AND:
            # 1. Compute K0
            K0 = F(self.Left.K[self.Left.Permutation], make_bitarray_with_2(self.Index, 0, 0))

            # 2. Set output wires and permutation bits
            if self.Left.Permutation == self.Right.Permutation and self.Right.Permutation == 1:
                k0, _ = pick_random_pair()
                permutation = secrets.randbits(1)
                k1 = K0[:SIZE]
            else:
                k0 = K0[:SIZE]
                permutation = K0[SIZE]
                k1, _ = pick_random_pair()

            Kl0 = k0.copy()
            Kl0.append(permutation)
            Kl1 = k1.copy()
            Kl1.append(1 ^ permutation)
            K = [Kl0, Kl1]

            # 3. Compute gate ciphertexts
            T1 = F(self.Left.K[self.Left.Permutation], make_bitarray_with_2(self.Index, 0, 1)) ^ F(self.Right.K[1 ^ self.Right.Permutation], make_bitarray_with_2(self.Index, 0, 1)) ^ K[self.Left.Permutation & (1 ^ self.Right.Permutation)]
            T2 = F(self.Left.K[1 ^ self.Left.Permutation], make_bitarray_with_2(self.Index, 1, 0)) ^ F(self.Right.K[self.Right.Permutation], make_bitarray_with_2(self.Index, 1, 0)) ^ K[(1 ^ self.Left.Permutation) & self.Right.Permutation]
            T3 = F(self.Left.K[1 ^ self.Left.Permutation], make_bitarray_with_2(self.Index, 1, 1)) ^ F(self.Right.K[1 ^ self.Right.Permutation], make_bitarray_with_2(self.Index, 1, 1)) ^ K[(1 ^ self.Left.Permutation) & (1 ^ self.Right.Permutation)]

            # 4. Return
            self.K = [k0, k1]
            self.Permutation = permutation
            self.TAnd = [T1, T2, T3]

        # Improved XOR Gates:
        elif self.Op == GateType.XORImproved:
            # 1. Output permutation bit
            permutation = self.Left.Permutation ^ self.Right.Permutation

            # 2. Translated keys
            ki0 = F(self.Left.K[0], make_bitarray_with(self.Index, self.Left.Permutation))[:SIZE]
            ki1 = F(self.Left.K[1], make_bitarray_with(self.Index, 1 ^ self.Left.Permutation))[:SIZE]

            # 3. New offset
            delta = ki0 ^ ki1

            # 4. Translated keys for j (right)
            if self.Right.Permutation == 0:
                kj0 = self.Right.K[0]
                kj1 = kj0 ^ delta
                T = F(self.Right.K[1], make_bitarray_with(self.Index, 1))[:SIZE] ^ kj1
            else:
                kj1 = self.Right.K[1]
                kj0 = kj1 ^ delta
                T = F(self.Right.K[0], make_bitarray_with(self.Index, 1))[:SIZE] ^ kj0

            # 5. Compute keys for output wire l
            k0 = ki0 ^ kj0
            k1 = k0 ^ delta

            # 6. Do something with the result
            self.K = [k0, k1]
            self.Permutation = permutation
            self.TXor = T

        # Improved AND Gates
        elif self.Op == GateType.ANDImproved:
            # 1. Compute keys
            K0F = F(self.Left.K[self.Left.Permutation], self.index2(0, 0)) ^ F(self.Right.K[self.Right.Permutation], self.index2(0, 0))
            K1F = F(self.Left.K[self.Left.Permutation], self.index2(0, 1)) ^ F(self.Right.K[1 ^ self.Right.Permutation], self.index2(0, 1))
            K2F = F(self.Left.K[1 ^ self.Left.Permutation], self.index2(1, 0)) ^ F(self.Right.K[self.Right.Permutation], self.index2(1, 0))
            K3F = F(self.Left.K[1 ^ self.Left.Permutation], self.index2(1, 1)) ^ F(self.Right.K[1 ^ self.Right.Permutation], self.index2(1, 1))
            K0, m0 = [K0F[:SIZE], K0F[SIZE]]
            K1, m1 = [K1F[:SIZE], K1F[SIZE]]
            K2, m2 = [K2F[:SIZE], K2F[SIZE]]
            K3, m3 = [K3F[:SIZE], K3F[SIZE]]

            # 2. Compute the location of 1 in the truth table
            s = 2 * (1 ^ self.Left.Permutation) + (1 ^ self.Right.Permutation)

            # 3. Output wire keys and permutation bits
            permutation = secrets.randbits(1)
            if s != 0:
                k0 = K0
                k1 = K1 ^ K2 ^ K3
            else:
                k0 = K1 ^ K2 ^ K3
                k1 = K0

            # 4. Compute T1, T2
            # 5. Compute additional 4 bits
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

            self.Permutation = permutation
            self.K = [k0, k1]
            self.TAnd = [T1, T2]
            self.t = t

            print(str(self.Index) + "] s   -- " + str(s))
            print(str(self.Index) + "] pi  -- " + str(self.Permutation))
            print(str(self.Index) + "] k0  -- " + str(self.K[0]))
            print(str(self.Index) + "] k1  -- " + str(self.K[1]))
            print(str(self.Index) + "] T   -- " + str(self.TAnd))
            print(str(self.Index) + "] t   -- " + str(self.t))

        # NOT Gate
        # by convention the left input will be the only input to the NOT gate
        elif self.Op == GateType.NOT:
            #self.Index = self.Left.Index
            self.Permutation = self.Left.Permutation
            self.K = [self.Left.K[1], self.Left.K[0]]
        else:
            print("Cannot garble a " + str(self.Op) + " gate!")

    def Evaluate(self):
        # XOR Gates:
        if self.Op == GateType.XOR:
            vLeft = F(self.Left.Output, make_bitarray_with(self.Index, self.Left.Signal))[:SIZE]
            vRight = F(self.Right.Output, make_bitarray_with(self.Index, self.Right.Signal))[:SIZE]
            if self.Right.Signal:
                k = vLeft ^ vRight ^ self.TXor
            else:
                k = vLeft ^ vRight

            self.Output = k
            self.Signal = self.Left.Signal ^ self.Right.Signal
            print(str(self.Index) + "] sig -- " + str(self.Signal))
        # AND Gates:
        elif self.Op == GateType.AND:
            if (self.Left.Signal + self.Right.Signal) == 0:
                T = make_bitarray_with(0, 0)
            else:
                T = self.TAnd[(self.Left.Signal * 2 + self.Right.Signal) - 1]
            key = T ^ F(self.Left.Output, make_bitarray_with_2(self.Index, self.Left.Signal, self.Right.Signal)) ^ F(self.Right.Output, make_bitarray_with_2(self.Index, self.Left.Signal, self.Right.Signal))
            self.Output = key[:SIZE]
            self.Signal = key[SIZE]
        # Improved XOR Gates:
        elif self.Op == GateType.XORImproved:
            if self.Right.Signal == 0:
                # Note that since we know the signal is 0 we do not need to XOR with TXor
                k = F(self.Left.Output, make_bitarray_with(self.Index, self.Left.Signal))[:SIZE] ^ self.Right.Output
            else:
                vLeft = F(self.Left.Output, make_bitarray_with(self.Index, self.Left.Signal))[:SIZE]
                vRight = F(self.Right.Output, make_bitarray_with(self.Index, 0))[:SIZE]
                k = vLeft ^ vRight ^ self.TXor

            self.Output = k
            self.Signal = self.Left.Signal ^ self.Right.Signal
        # Improved AND Gates:
        elif self.Op == GateType.ANDImproved:
            KF = F(self.Left.Output, self.index2(self.Left.Signal, self.Right.Signal)) ^ F(self.Right.Output, self.index2(self.Left.Signal, self.Right.Signal))
            k, m = [KF[:SIZE], KF[SIZE]]

            print(str(self.Index) + "] lk  -- " + str(self.Left.Output))
            print(str(self.Index) + "] li  -- " + str(self.index2(self.Left.Signal, self.Right.Signal)))
            print(str(self.Index) + "] rk  -- " + str(self.Right.Output))
            print(str(self.Index) + "] ri  -- " + str(self.index2(self.Left.Signal, self.Right.Signal)))

            print(str(self.Index) + "] k   -- " + str(k))
            print(str(self.Index) + "] m   -- " + str(m))

            signals = 2 * self.Left.Signal + self.Right.Signal
            print(str(self.Index) + "] sigs-- " + str(signals))

            if signals == 0:
                self.Output = k
            elif signals == 1:
                self.Output = k ^ self.TAnd[0]
            elif signals == 2:
                self.Output = k ^ self.TAnd[1]
            else:
                self.Output = k ^ self.TAnd[0] ^ self.TAnd[1]
            self.Signal = m ^ self.t[signals]

            print(str(self.Index) + "] sig -- " + str(self.Signal))
            print(str(self.Index) + "] out -- " + str(self.Output))
        # NOT Gate
        # by convention the left input will be the only input to the NOT gate
        elif self.Op == GateType.NOT:
            self.Signal = 1 ^ self.Left.Signal
            self.Output = self.Left.Output
        else:
            print("Cannot evaluate a " + str(self.Op) + " gate!")


class InputGate(CircuitGate):
    def __init__(self, index: int):
        super().__init__(index)

    def Garble(self):
        super(InputGate, self).Garble()

        k0, k1 = pick_random_pair()
        self.K = [k0, k1]

        print(str(self.Index) + "] k0  -> " + str(self.K[0]))
        print(str(self.Index) + "] k1  -> " + str(self.K[1]))

        self.Permutation = secrets.randbits(1)
        print(str(self.Index) + "] pi  -> " + str(self.Permutation))

        e0 = k0.copy()
        e0.append(self.Permutation)
        e1 = k1.copy()
        e1.append(1 ^ self.Permutation)
        self.E = [e0, e1]

        print(str(self.Index) + "] e0  -> " + str(self.E[0]))
        print(str(self.Index) + "] e1  -> " + str(self.E[1]))

    def Encode(self, value):
        self.Output = self.E[value]
        print(str(self.Index) + "] in  -> " + str(self.Output))

    def Evaluate(self):
        self.Signal = self.Output[SIZE]
        self.Output = self.Output[:SIZE]

        print(str(self.Index) + "] sig -> " + str(self.Signal))
        print(str(self.Index) + "] out -> " + str(self.Output))
        return


class OutputGate:
    def __init__(self, value: CircuitGate):
        self.Value = value
        self.D = [make_bitarray(MAGIC), make_bitarray(MAGIC)]

    def Garble(self):
        g = self.Value
        #self.D[0] = F(g.K[g.Permutation], make_bitarray_with(g.Index, g.Permutation))
        #arr = make_bitarray_with(g.Index, 1 ^ g.Permutation)
        #self.D[1] = F(g.K[1 ^ g.Permutation], arr)
        # TODO: Changing it to this seems to fix the issue, but I do not think this is correct!
        self.D[0] = F(g.K[0], make_bitarray_with(g.Index, g.Permutation))
        self.D[1] = F(g.K[1], make_bitarray_with(g.Index, 1 ^ g.Permutation))
        print("X] d0  <- " + str(self.D[0]))
        print("X] d1  <- " + str(self.D[1]))

    def Evaluate(self):
        arr = make_bitarray_with(self.Value.Index, self.Value.Signal)
        output = F(self.Value.Output, arr)
        self.Value.Output = output
        print("X] out <- " + str(self.Value.Output))

    def Decode(self):
        g = self.Value
        if g.Output == self.D[0]:
            return 0
        elif g.Output == self.D[1]:
            return 1
        else:
            return -1


class FastCircuit:
    def __init__(self, gates: List[CircuitGate], inputs: List[InputGate],
                outputs: List[OutputGate], intermediates: List[GarbledGate]):
        self.Inputs = inputs
        self.Outputs = outputs
        self.Gates = gates
        self.Intermediates = intermediates

    def Garble(self):
        for gate in self.Gates:
            gate.Garble()
        for gate in self.Outputs:
            gate.Garble()

    def Encode(self, values):
        for i, gate in enumerate(self.Inputs):
            gate.Encode(values[i])

    def Evaluate(self):
        for gate in self.Inputs:
            gate.Evaluate()
        for gate in self.Intermediates:
            gate.Evaluate()
        for gate in self.Outputs:
            gate.Evaluate()

    def Decode(self):
        result = []
        for gate in self.Outputs:
            result.append(gate.Decode())
        return result


# Simple circuit with one AND gate
input1 = InputGate(0)
input2 = InputGate(1)
ins = [input1, input2]

xorGate = GarbledGate(2, GateType.NOT, input1, input2)
steps = [xorGate]

outputGate = OutputGate(xorGate)
outs = [outputGate]

all = [input1, input2, xorGate]
circuit = FastCircuit(all, ins, outs, steps)

# Garble
a = 1
b = 0
print("##### Garble")
circuit.Garble()
print("##### Encode")
circuit.Encode([a, b])
print("##### Evaluate")
circuit.Evaluate()
print("##### Decode")
result = circuit.Decode()

print(result)
correct = (a ^ b) == result[0]
print("FastCircuit done! " + str(correct))
