# For working with bristol circuits
from bitarray import bitarray
from bitarray.util import int2ba, ba2int

class Gate:
    def __init__(self, num_inputs, num_outputs, input_wires, output_wires, operation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_wires = input_wires
        self.output_wires = output_wires
        self.operation = operation

    def op(self, *args):
        if self.operation == 'AND':
            return args[0] & args[1]
        elif self.operation == 'XOR':
            return args[0] ^ args[1]
        else:
            raise Exception('Invalid operation')

    def __str__(self):
        return f"{self.num_inputs} {self.num_outputs} {self.input_wires} {self.output_wires} {self.operation}"


class Circuit:
        def __init__(self, *args):
            """
            Initialize a circuit.
            """
            if len(args) == 1:
                self.__parse(args[0])
            else:
                self.__new(*args)

        def __new(self, num_inputs, input_sizes, num_outputs, output_sizes, num_wires, gates):
            self.num_inputs = num_inputs
            self.input_sizes = input_sizes
            self.num_outputs = num_outputs
            self.output_sizes = output_sizes
            self.num_wires = num_wires
            self.gates = gates

        def __str__(self):
            return f"{self.num_inputs} {self.input_sizes} {self.num_outputs} {self.output_sizes} " + "\n".join(map(str, self.gates))

        def eval(self, *input):
            """
            Evaluates the circuit with the given inputs
            """
            # handle inputs
            if len(input) != self.num_inputs:
                raise ValueError("Wrong number of inputs")
            for i,e in enumerate(input):
                if len(e) != self.input_sizes[i]:
                    raise ValueError("Wrong input size")
            # setup registers/wires
            k = 0
            wires = bitarray(self.num_wires, endian='little')
            for i in range(self.num_inputs):
                for j in range(self.input_sizes[i]):
                    wires[k] = input[i][j]
                    k += 1
            # evaluate gates
            for gate in self.gates:
                wires[gate.output_wires[0]] = gate.op(*[wires[i] for i in gate.input_wires])
            # return output
            output = wires[-self.output_sizes[0]:] # handle multiple outputs
            return output

        def __parse(self, raw : str):
            """
            Parse a string into a circuit.
            """
            rows = raw.split('\n')
            num_gates, num_wires = rows[0].strip().split(' ')
            num_gates = int(num_gates)
            num_wires = int(num_wires)
            num_inputs = int(rows[1].strip().split(' ')[0])
            input_sizes = [int(i) for i in rows[1].strip().split(' ')[1:]]
            num_outputs = int(rows[2].strip().split(' ')[0])
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
            self.__new(num_inputs, input_sizes, num_outputs, output_sizes, num_wires, gates)



f = open("./adder64.txt")
raw = f.read()
c = Circuit(raw)

a = int2ba(5, 64, endian='little')
b = int2ba(5, 64, endian='little')

ab = c.eval(a, b)

print(ba2int(ab))


