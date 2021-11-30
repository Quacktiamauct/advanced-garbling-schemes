# For working with bristol circuits
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from enum import Enum
from typing import List


class Operation(Enum):
    INV = (0,)
    AND = (1,)
    XOR = (2,)
    OR = (3,)

    def __str__(self):
        return self.name


op_dict = {
    "AND": Operation.AND,
    "XOR": Operation.XOR,
    "INV": Operation.INV,
    "OR": Operation.OR,
}


class Gate:
    operation: Operation
    left: int
    right: int
    output: int

    def __init__(self, operation, left, right, output):
        self.operation = operation
        self.left = left
        self.right = right
        self.output = output

    def op(self, *args) -> int:
        if self.operation == Operation.INV:
            return 1 ^ args[0]
        elif self.operation == Operation.AND:
            return args[0] & args[1]
        elif self.operation == Operation.XOR:
            return args[0] ^ args[1]
        elif self.operation == Operation.OR:
            return args[0] | args[1]
        else:
            raise Exception("Invalid operation")

    def __str__(self):
        return f"{self.operation} {self.left} {self.right} {self.output}"


class Circuit:
    def __init__(self, *args):
        """
        Initialize a circuit.
        """
        if len(args) == 1:
            self.__parse(args[0])
        else:
            self.__new(*args)

    def __new(
        self, num_inputs, input_sizes, num_outputs, output_sizes, num_wires, gates
    ):
        self.num_inputs = num_inputs
        self.input_sizes = input_sizes
        self.num_outputs = num_outputs
        self.output_sizes = output_sizes
        self.num_wires = num_wires
        self.gates = gates

    def __str__(self):
        """
        Return a string representation of the circuit.
        """
        s = f"{len(self.gates)} {self.num_wires}\n"
        s += f"{self.num_inputs} {' '.join([str(i) for i in self.input_sizes])}\n"
        s += f"{self.num_outputs} {' '.join([str(i) for i in self.output_sizes])}\n"
        for gate in self.gates:
            s += f"{gate.left} {gate.right} {gate.output} {gate.operation}\n"
        return s

    def eval(self, *input) -> bitarray:
        """
        Evaluates the circuit with the given inputs
        """
        # handle inputs
        if len(input) != self.num_inputs:
            raise ValueError(
                "Wrong number of inputs."
                + f" Provided {len(input)}, but needed {self.num_inputs}"
            )
        for i, e in enumerate(input):
            if len(e) != self.input_sizes[i]:
                raise ValueError(
                    "Wrong input size. "
                    + f"Provided {len(e)}, but needed {self.input_sizes[i]}"
                )
        # setup registers/wires
        k = 0
        wires = bitarray(self.num_wires, endian="little")
        for i in range(self.num_inputs):
            for j in range(self.input_sizes[i]):
                wires[k] = input[i][j]
                k += 1
        # evaluate gates
        for gate in self.gates:
            wires[gate.output] = gate.op(wires[gate.left], wires[gate.right])
        # return output
        output = wires[-self.output_sizes[0] :]  # handle multiple outputs
        return output

    def __parse(self, raw: str):
        """
        Parse a string into a circuit.
        """
        rows = raw.split("\n")
        num_gates, num_wires = rows[0].strip().split(" ")
        num_gates = int(num_gates)
        c_num_wires = int(num_wires)
        c_num_inputs = int(rows[1].strip().split(" ")[0])
        c_input_sizes = [int(i) for i in rows[1].strip().split(" ")[1:]]
        c_num_outputs = int(rows[2].strip().split(" ")[0])
        c_output_sizes = [int(i) for i in rows[2].strip().split(" ")[1:]]
        gates = []
        for r in rows[4:]:
            if r == "":
                continue
            tmp = r.strip().split(" ")
            num_inputs = tmp[0]
            num_inputs = int(num_inputs)
            input_wires = [int(a) for a in tmp[2 : 2 + num_inputs]]
            output_wires = [int(a) for a in tmp[2 + num_inputs : -1]]
            operation = tmp[-1]
            left = input_wires[0]
            if len(input_wires) == 1:
                right = -1
            else:
                right = input_wires[1]
            gates.append(Gate(op_dict[operation], left, right, output_wires[0]))
        self.__new(
            c_num_inputs,
            c_input_sizes,
            c_num_outputs,
            c_output_sizes,
            c_num_wires,
            gates,
        )


if __name__ == "__main__":
    f = open("./bristol/sub64.txt")
    raw = f.read()
    c = Circuit(raw)
    a = int2ba(7, 64, endian="little")
    b = int2ba(5, 64, endian="little")
    ab = c.eval(a, b)
    print(ba2int(ab))
