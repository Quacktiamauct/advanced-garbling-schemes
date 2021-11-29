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


SIZE = 128
MAGIC = 2863311530 # 101010101010101010... in binary

def make_bitarray(i: int):
    return int2ba(i, SIZE, endian='little')


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
        print(self.Output)


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


def parse(raw : str):
    """
    Parse a string into a circuit.
    """
    rows = raw.split('\n')
    num_gates, num_wires = rows[0].strip().split(' ')
    num_gates = int(num_gates)
    num_wires = int(num_wires)
    num_inputs = int(rows[1].strip().split(' ')[0])
    input_sizes = [int(i) for i in rows[1].strip().split(' ')[1:]]
    # num_outputs = int(rows[2].strip().split(' ')[0]) # assume 1
    output_sizes = [int(i) for i in rows[2].strip().split(' ')[1:]]
    inputs = [InputGate(i) for i in range(sum(input_sizes))]
    gate_dict: Dict[int, CircuitGate] = {i: gate for i, gate in enumerate(inputs)}
    for r in rows[4:]:
        if r == '':
            continue
        tmp = r.strip().split(' ')
        num_inputs, num_outputs = tmp[0:2]
        num_inputs = int(num_inputs)
        # num_output = int(num_outputs)
        input_wires = [int(a) for a in tmp[2:2+num_inputs]]
        output_wires = [int(a) for a in tmp[2+num_inputs:-1]]
        operation = tmp[-1]
        op_dict = {
            'AND': GateType.AND,
            'OR': GateType.OR,
            'XOR': GateType.XOR,
            'NAND': GateType.NAND,
            'NOT': GateType.NOT
            }
        leftGate = gate_dict[input_wires[0]]
        rightGate = gate_dict[input_wires[1]]
        gate_dict[output_wires[0]] = GarbledGate(output_wires[0],
            op_dict[operation], leftGate, rightGate)
    gates = list(gate_dict.values())
    input_size = sum(input_sizes)
    output_size = sum(output_sizes)
    outputs = []
    for i in range(output_size):
        outputs.append(OutputGate(gate_dict[num_wires-i-1]))
    gates.extend(outputs)
    # HACK: stupid type hints don't work with this
    return YaoCircuit(gates,
        inputs,
        outputs,
        gates[input_size:len(gate_dict)])


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

# if __name__ == '__main__':
#     f = open("./adder64.txt")
#     raw = f.read()
#     c = parse(raw)
#     c.Garble()
#     a_num = 5
#     b_num = 7
#     a = int2ba(a_num, 64, endian='little')
#     b = int2ba(b_num, 64, endian='little')
#     bits = []
#     for aBit in a:
#         bits.append(aBit)
#     for bBit in b:
#         bits.append(bBit)
#     c.Encode(bits)
#     c.Evaluate()
#     res = c.Decode()
#     print(f"{a_num} + {b_num} = {res}")




