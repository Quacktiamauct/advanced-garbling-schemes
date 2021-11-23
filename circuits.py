import bfcl
from bfcl import circuit, gate


class Gate:
    def __init__(self, num_inputs, num_outputs, input_wires, output_wires, operation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_wires = input_wires
        self.output_wires = output_wires
        self.operation = operation

    def __str__(self):
        return f"{self.num_inputs} {self.num_outputs} {self.input_wires} {self.output_wires} {self.operation}"


class Circuit:
        def __init__(self, num_inputs, input_sizes, num_outputs, output_sizes, num_wires, gates):
            self.num_inputs = num_inputs
            self.input_sizes = input_sizes
            self.num_outputs = num_outputs
            self.output_sizes = output_sizes
            self.num_wires = num_wires
            self.gates = gates

        def __str__(self):
            return f"{self.num_inputs} {self.input_sizes} {self.num_outputs} {self.output_sizes} " + "\n".join(map(str, self.gates))

        def eval(self, *input):
            if len(input) != self.num_inputs:
                raise ValueError("Wrong number of inputs")
            for i,e in enumerate(input):
                if len(e) != self.input_sizes[i]:
                    raise ValueError("Wrong input size")
            wires = [0] * self.num_wires
            k = 0
            for i in range(self.num_inputs):
                for j in range(self.input_sizes[i]):
                    wires[k] = input[i][j]
                    k += 1
            for gate in self.gates:
                if gate.operation == "AND":
                    wires[gate.output_wires[0]] = wires[gate.input_wires[0]] & wires[gate.input_wires[1]]
                elif gate.operation == "XOR":
                    wires[gate.output_wires[0]] = wires[gate.input_wires[0]] ^ wires[gate.input_wires[1]]
                else:
                    raise ValueError("Unknown operation")
            output = wires[-self.output_sizes[0]:] # handle multiple outputs
            print(wires[:64])
            print(wires[64:128])
            print(wires[-64:])
            return output


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
    output_sizes = [int(i) for i in rows[2].strip().split(' ')[1:]]
    gates = []
    for r in rows[4:]:
        if r == '':
            continue
        tmp = r.strip().split(' ')
        num_inputs, num_outputs = tmp[0:2]
        num_inputs = int(num_inputs)
        num_output = int(num_outputs)
        input_wires = [int(a) for a in tmp[2:2+num_inputs]]
        output_wires = [int(a) for a in tmp[2+num_inputs:-1]]
        operation = tmp[-1]
        gates.append(
            Gate(num_inputs, num_output, input_wires, output_wires, operation)
        )
    return Circuit(num_inputs, input_sizes, num_outputs, output_sizes, num_wires, gates)

def int_to_bitarray(i, size=None):
    arr = [int(x) for x in bin(i)[2:]]
    if size != None:
        arr = [0] * (size - len(arr)) + arr
    return arr

def bitarray_to_int(arr):
    return int(''.join(map(str, arr)), 2)



f = open("./adder64.txt")
raw = f.read()
c = parse(raw)
a = int_to_bitarray(5, 64)
b = int_to_bitarray(5, 64)
ab = c.eval(a, b)
print(bitarray_to_int(ab))

