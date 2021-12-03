import secrets
import pytest
from half import garble
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

# Test of simple gates
def test_xor():
    many("bristol/simple_xor.txt")

def test_and():
    many("bristol/simple_and.txt")

def test_not():
    file = "bristol/not.txt"
    print("Testing file: " + file)
    f = open(file)
    c = Circuit(f.read())
    MAX = 1000
    success = 0
    for _ in range(MAX):
        gc = garble(c)
        for a in [0, 1]:
            arg = bitarray([a], endian='little')
            try:
                res = gc.eval(arg)
                if res != c.eval(arg):
                    print("FAILED:", a, res, c.eval(arg))
                else:
                    success += 1
            except Exception as e:
                print("FAILED:", a, "expcetion:", e)
    print(success, "/", MAX*2, "tests passed")
    assert MAX * 2 == success

# Arithmetic on unsigned integers
def test_adder():
    for _ in range(10):
        num1 = secrets.randbits(32)
        num2 = secrets.randbits(32)
        f = open("./bristol/adder64.txt")
        c = Circuit(f.read())
        a = int2ba(num1, 64, "little")
        b = int2ba(num2, 64, "little")
        gc = garble(c)
        res = gc.eval(a, b)
        assert ba2int(a) + ba2int(b) == ba2int(res)

def test_mult():
    for _ in range(10):
        num1 = secrets.randbits(32)
        num2 = secrets.randbits(32)
        f = open("./bristol/mult64.txt")
        c = Circuit(f.read())
        a = int2ba(num1, 64, "little")
        b = int2ba(num2, 64, "little")
        gc = garble(c)
        res = gc.eval(a, b)
        assert ba2int(a) * ba2int(b) == ba2int(res)

def test_sub():
    for _ in range(10):
        num1 = secrets.randbits(32)
        num2 = secrets.randbits(32)
        if num2 > num1:
            tmp = num1
            num1 = num2
            num2 = tmp

        f = open("./bristol/sub64.txt")
        c = Circuit(f.read())
        a = int2ba(num1, 64, "little")
        b = int2ba(num2, 64, "little")
        gc = garble(c)
        res = gc.eval(a, b)
        assert ba2int(a) - ba2int(b) == ba2int(res)