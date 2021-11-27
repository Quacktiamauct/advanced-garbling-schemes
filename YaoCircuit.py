import itertools
import secrets
import hashlib

from typing import Tuple, List
from enum import Enum
from bitarray import bitarray
from bitarray.util import int2ba, ba2int


class GateType(Enum):
    NOT = 0,
    AND = 1,
    XOR = 2,
    OR = 3,
    NAND = 4


SIZE = 128
MAGIC = 2863311530 # 101010101010101010... in binary

def make_bitarray(i: int):
    return int2ba(i, 128, endian='little')


def zero(b: bitarray):
    for bit in b:
        if bit == 1:
            return False
    return True


def pick_random_pair():
    rnd = secrets.token_bytes(32)
    arr = bitarray()
    arr.frombytes(rnd)
    return arr[:SIZE], arr[SIZE:]


def G(left: bitarray, right: bitarray, i: int) -> Tuple[bitarray, bitarray]:
    """
        input: A and B are bitarrays of size SIZE
        returns a 2*SIZE bitarray
    """
    ia = int2ba(i, SIZE, endian='little')
    food = left + right + ia
    arr = bitarray()
    arr.frombytes(hashlib.sha256(food.tobytes()).digest())
    return arr[SIZE:], arr[:SIZE]


class CircuitGate:
    def __init__(self, index: int):
        self.Index = index
        self.K = [make_bitarray(MAGIC), make_bitarray(MAGIC)]
        self.Output = make_bitarray(MAGIC)

    def Garble(self):
        kL, kR = pick_random_pair()
        self.K = [kL, kR]


class GarbledGate(CircuitGate):
    Op: GateType
    Left: CircuitGate
    Right: CircuitGate
    C: List[Tuple[bitarray, bitarray]]

    def __init__(self, index: int, op: GateType, left: CircuitGate, right: CircuitGate):
        super().__init__(index)
        self.Op = op
        self.Left = left
        self.Right = right
        self.C = [(make_bitarray(MAGIC), make_bitarray(MAGIC))] * 4

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

        for j, (leftValue, rightValue) in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
            gL, gR = G(self.Left.K[leftValue], self.Right.K[rightValue], self.Index)

            gateValue = self.Compute(leftValue, rightValue)
            garbledValue = self.K[gateValue]
            # TODO: This XOR might not work
            self.C[j] = gL ^ garbledValue, gR

        # Permute the table
        permute = secrets.choice(list(itertools.permutations([0, 1, 2, 3])))
        self.C = [self.C[k] for k in permute]

    def Evaluate(self):
        leftValue = self.Left.Output
        rightValue = self.Right.Output
        gL, gR = G(leftValue, rightValue, self.Index)

        evaluation = -1
        for i in range(4):
            cL, cR = self.C[i]
            k = gL ^ cL
            t = gR ^ cR
            if zero(t):
                evaluation = k

        # If everything works and is done correctly this should not happen
        if evaluation == -1:
            print("ABORT SOMETHING IS WRONG WITH WIRE " + str(self.Index) + " !!!")
        else:
            self.Output = evaluation


class InputGate(CircuitGate):
    def __init__(self, index: int):
        super().__init__(index)

    def Encode(self, value):
        self.Output = self.K[value]


class OutputGate:
    def __init__(self, value: CircuitGate):
        self.Value = value

    def Decode(self):
        if self.Value.K[0] == self.Value.Output:
            return 0
        elif self.Value.K[1] == self.Value.Output:
            return 1
        else:
            return -1


class YaoCircuit:
    def __init__(self, gates: List[CircuitGate], inputs: List[InputGate],
                outputs: List[OutputGate], intermediates: List[GarbledGate]):
        self.Inputs = inputs
        self.Outputs = outputs
        self.Gates = gates
        self.Intermediates = intermediates

    def Garble(self):
        for gate in self.Gates:
            gate.Garble()

    def Encode(self, values):
        for i, gate in enumerate(self.Inputs):
            gate.Encode(values[i])

    def Evaluate(self):
        for gate in self.Intermediates:
            gate.Evaluate()

    def Decode(self):
        result = []
        for gate in self.Outputs:
            result.append(gate.Decode())
        return result


# Simple circuit with one AND gate
input1 = InputGate(0)
input2 = InputGate(1)
input3 = InputGate(2)
input4 = InputGate(3)
ins = [input1, input2, input3, input4]

andGate = GarbledGate(4, GateType.AND, input1, input2)
xorGate = GarbledGate(5, GateType.XOR, input3, input4)
orGate = GarbledGate(6, GateType.OR, andGate, xorGate)
steps = [andGate, xorGate, orGate]

outputGate = OutputGate(orGate)
outs = [outputGate]

all = [input1, input2, input3, input4, andGate, xorGate, orGate]
circuit = YaoCircuit(all, ins, outs, steps)

# Garble
circuit.Garble()
circuit.Encode([1, 0, 1, 1])
circuit.Evaluate()
result = circuit.Decode()

print(result)

print("YaoCircuit done!")
