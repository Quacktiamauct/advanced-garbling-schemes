import pytest
from fast import garble
from circuits import Circuit
from bitarray.util import ba2int, int2ba
from bitarray import bitarray

def many(file):
    print("Testing file: " + file)
    f = open(file)
    c = Circuit(f.read())
    MAX = 1000
    success = 0
    for _ in range(MAX):
        gc = garble(c)
        for a,b in [(0,0), (0,1), (1,0), (1,1)]:
            arg = bitarray([a,b], endian='little')
            try:
                res = gc.eval(arg)
                if res != c.eval(arg):
                    print("FAILED:", a, b, res, c.eval(arg))
                else:
                    success += 1
            except Exception as e:
                print("FAILED:", a, b, "expcetion:", e)
    print(success, "/", MAX*4, "tests passed")
    assert MAX * 4 == success

def test_xor():
    many("bristol/simple_xor.txt")

def test_and():
    many("bristol/simple_and.txt")

def test_adder():
    f = open("./bristol/adder64.txt")
    c = Circuit(f.read())
    a = int2ba(7, 64, "little")
    b = int2ba(5, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    print(f"{ba2int(a)} + {ba2int(b)} = {ba2int(res)}")
    assert ba2int(a) + ba2int(b) == ba2int(res)

def test_mult():
    f = open("./bristol/mult64.txt")
    c = Circuit(f.read())
    a = int2ba(5, 64, "little")
    b = int2ba(3, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    assert ba2int(a) * ba2int(b) == ba2int(res)

def test_sub():
    f = open("./bristol/sub64.txt")
    c = Circuit(f.read())
    a = int2ba(5, 64, "little")
    b = int2ba(5, 64, "little")
    gc = garble(c)
    res = gc.eval(a, b)
    assert ba2int(a) - ba2int(b) == ba2int(res)