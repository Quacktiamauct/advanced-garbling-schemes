import sys
import hashlib
from benchmark import adder64, sub64, aes256, sha256, fp_sqrt, mult64

from fast import garble as garble_fast
from half import garble as garble_half
from yao import garble as garble_yao

from benchmark import generate_args, sanity_check

import util

if __name__ == '__main__':
    bad_fast = lambda c: garble_fast(c, improvedXor=False, improvedAnd=False)
    garbles = {
        "YAO": garble_yao,
        "FAST unimproved": bad_fast,
        "FAST improved": garble_fast,
        "HALF": garble_half
    }
    circuits = {
        "adder64": adder64,
        "sub64":   sub64,
        "mult64":  mult64,
        "fp_sqrt": fp_sqrt,
        "sha256":  sha256,
        "aes256":  aes256
    }
    for g in garbles:
        print(f" =========== using {g} ===========")
        for c in circuits:
            print(f"{c}:")
            args = generate_args(circuits[c])
            sanity_check(circuits[c], *args)
            garbled_circuit = garbles[g](circuits[c])
            print("PRF called for garble", util.prf_called)
            util.prf_called = 0
            garbled_circuit.eval(*args)
            print("PRF called for eval", util.prf_called)
            util.prf_called = 0
    pass
