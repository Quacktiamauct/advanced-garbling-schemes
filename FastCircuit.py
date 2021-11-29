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
    NAND = 4


#SIZE = 128
#MAGIC = 2863311530 # 101010101010101010... in binary

SIZE = 4
MAGIC = 5
EXTENDED_SIZE = SIZE + 1


def make_bitarray(i: int):
    return int2ba(i, SIZE, endian='little')

def make_bitarray_with(i: int, b: int):
    r = make_bitarray(i)
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

    def Garble(self):
        super(GarbledGate, self).Garble()

        if self.Op == GateType.XOR:
            # i = left, j = right, l = this gate
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
            self.Permutation = permutation
            self.K = [k0, k1]
            self.TXor = T
        else:
            print("Cannot garble a " + str(self.Op) + " gate!")


    def Evaluate(self):
        if self.Op == GateType.XOR:
            vLeft = F(self.Left.Output, make_bitarray_with(self.Index, self.Left.Signal))[:SIZE]
            vRight = F(self.Right.Output, make_bitarray_with(self.Index, self.Right.Signal))[:SIZE]
            if self.Right.Signal:
                k = vLeft ^ vRight ^ self.TXor
            else:
                k = vLeft ^ vRight

            self.Output = k
            self.Signal = self.Left.Signal ^ self.Right.Signal
        else:
            print("Cannot evaluate a " + str(self.Op) + " gate!")


class InputGate(CircuitGate):
    def __init__(self, index: int):
        super().__init__(index)

    def Garble(self):
        super(InputGate, self).Garble()

        k0, k1 = pick_random_pair()
        self.K = [k0, k1]

        self.Permutation = secrets.randbits(1)

        e0 = k0.copy()
        e0.append(self.Permutation)
        e1 = k1.copy()
        e1.append(1 ^ self.Permutation)
        self.E = [e0, e1]


    def Encode(self, value):
        self.Output = self.E[value]


    def Evaluate(self):
        self.Signal = self.Output[SIZE]
        self.Output = self.Output[:SIZE]
        return


class OutputGate:
    def __init__(self, value: CircuitGate):
        self.Value = value
        self.D = [make_bitarray(MAGIC), make_bitarray(MAGIC)]

    def Garble(self):
        g = self.Value
        self.D[0] = F(g.K[g.Permutation], make_bitarray_with(g.Index, g.Permutation))
        self.D[1] = F(g.K[1 ^ g.Permutation], make_bitarray_with(g.Index, 1 ^ g.Permutation))
        return


    def Evaluate(self):
        output = F(self.Value.Output, make_bitarray_with(self.Value.Index, self.Value.Signal))
        self.Value.Output = output
        return

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
        for gate in self.Inputs:
            gate.Garble()
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

xorGate = GarbledGate(2, GateType.XOR, input1, input2)
steps = [xorGate]

outputGate = OutputGate(xorGate)
outs = [outputGate]

all = [input1, input2, xorGate]
circuit = FastCircuit(all, ins, outs, steps)

# Garble
circuit.Garble()
circuit.Encode([0, 0])
circuit.Evaluate()
result = circuit.Decode()

print(result)
print("FastCircuit done!")