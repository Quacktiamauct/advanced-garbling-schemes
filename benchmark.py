from bitarray.util import urandom
from bitarray import bitarray
from fast import garble as garble_fast
from half import garble as garble_half
from yao import garble as garble_yao
from circuits import Circuit
from typing import Callable, Any, List
import time

fp_sqrt = Circuit(open("bristol/FP-sqrt.txt").read())
adder64 = Circuit(open("bristol/adder64.txt").read())
sub64 = Circuit(open("bristol/sub64.txt").read())
mult64 = Circuit(open("bristol/mult64.txt").read())
aes256 = Circuit(open("bristol/aes_256.txt").read())
sha256 = Circuit(open("bristol/sha256.txt").read())

N = 100

def sanity_check(circuit: Circuit, *args: bitarray):
    """
    Sanity check for the circuit.
    """
    try:
        circuit.eval(*args)
    except Exception as e:
        print(f"Sanity check failed.")
        print(f"args used: {args}")
        raise e


def generate_args(circuit: Circuit) -> List[bitarray]:
    """
    Generate random arguments for the circuit
    """
    args = []
    for i in range(circuit.num_inputs):
        args.append(urandom(circuit.input_sizes[i], endian='little'))
    return args


def benchmark(circuit: Circuit, garble : Callable[[Circuit], Any]):
    """
    This function is used to test the performance of a circuit
    using different garbling schemes
    """
    args = generate_args(circuit)
    sanity_check(circuit, *args)

    # Garbling
    garbling_times = []
    for _ in range(N):
        start = time.time_ns()
        gc = garble(circuit)
        end = time.time_ns()
        garbling_times.append(end - start)
    avg_garble_time = 1e-3 * sum(garbling_times) / N
    gc = garble(circuit)

    # Evaluating
    evaluation_times = []
    for _ in range(N):
        start = time.time_ns()
        gc.eval(*args)
        end = time.time_ns()
        evaluation_times.append(end - start)
    avg_eval_time = 1e-3 * sum(evaluation_times) / N
    print(f"Avg. garbling time:     {avg_garble_time:9.2f} miliseconds")
    print(f"Avg. eval time:         {avg_eval_time:9.2f} miliseconds")

def main():
    print(" ========== Benchmarking adder64 ==========")
    print("YAO:")
    benchmark(adder64, garble_yao)
    print("FAST (normal):")
    benchmark(adder64, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(adder64, garble_fast)
    print("HALF GATES:")
    benchmark(adder64, garble_half)

    print()
    print(" ========== Benchmarking sub64 ==========")
    print("YAO:")
    benchmark(sub64, garble_yao)
    print("FAST (normal):")
    benchmark(sub64, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(sub64, garble_fast)
    print("HALF GATES:")
    benchmark(sub64, garble_half)

    print()
    print(" ========== Benchmarking mult64 ==========")
    print("YAO:")
    benchmark(mult64, garble_yao)
    print("FAST (normal):")
    benchmark(mult64, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(mult64, garble_fast)
    print("HALF GATES:")
    benchmark(mult64, garble_half)

    print()
    print(" ========== Benchmarking FP-sqrt ==========")
    print("YAO:")
    benchmark(fp_sqrt, garble_yao)
    print("FAST (normal):")
    benchmark(fp_sqrt, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(fp_sqrt, garble_fast)
    print("HALF GATES:")
    benchmark(fp_sqrt, garble_half)

    print()
    print(" ========== Benchmarking aes256 ==========")
    print("YAO:")
    benchmark(aes256, garble_yao)
    print("FAST (normal):")
    benchmark(aes256, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(aes256, garble_fast)
    print("HALF GATES:")
    benchmark(aes256, garble_half)

    print(" ========== Benchmarking sha256 ==========")
    print("YAO:")
    benchmark(sha256, garble_yao)
    print("FAST (normal):")
    benchmark(sha256, lambda c: garble_fast(c, improvedXor=False, improvedAnd=False))
    print("FAST (improved):")
    benchmark(sha256, garble_fast)
    print("HALF GATES:")
    benchmark(sha256, garble_half)

if __name__ == '__main__':
    main()
