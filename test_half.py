import pytest
from half import garble
from circuits import Circuit
from bitarray.util import ba2int, int2ba
from bitarray import bitarray

def test_adder():
    num1 = 64 + 4865
    num2 = 64 + 123
    a = int2ba(num1, 64, "little")
    b = int2ba(num2, 64, "little")
    expected = int2ba(num1 + num2, 64, "little")
    failed = 0
    passed = 0
    for something in range(100):
        f = open("./bristol/adder64.txt")
        c = Circuit(f.read())
        gc = garble(c)
        res = gc.eval(a, b)

        delta = res ^ expected
        if expected == res:
            passed += 1
            if delta[0] == 0:
                print("\33[32mPASS\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[32mPASS\33[0m -- \33[32m LSB FAIL\33[0m")
        else:
            failed += 1

            print("a         = " + str(a) + " (" + str(ba2int(a)) + ")")
            print("b         = " + str(b) + " (" + str(ba2int(b)) + ")")
            print("res       = " + str(res) + " (" + str(ba2int(res)) + ")")
            print("expecting = " + str(expected) + " (" + str(ba2int(expected)) + ")")
            print("delta     = " + str(delta) + " (" + str(ba2int(delta)) + ")")

            if delta[0] == 0:
                print("\33[31mFAIL\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[31mFAIL\33[0m -- \33[32m LSB FAIL\33[0m")

    print("")

    total = failed + passed
    if passed == total:
        print("\33[32mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    elif passed == 0:
        print("\33[31mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    else:
        print("\33[33mPasses " + str(passed) + " / " + str(total) + "\33[0m")

    assert total == passed

def test_sub():
    num1 = 2
    num2 = 1
    a = int2ba(num1, 64, "little")
    b = int2ba(num2, 64, "little")
    expected = int2ba(num1 - num2, 64, "little")
    failed = 0
    passed = 0
    for something in range(100):
        f = open("./bristol/sub64.txt")
        c = Circuit(f.read())
        gc = garble(c)
        res = gc.eval(a, b)

        delta = res ^ expected
        if expected == res:
            passed += 1
            if delta[0] == 0:
                print("\33[32mPASS\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[32mPASS\33[0m -- \33[32m LSB FAIL\33[0m")
        else:
            failed += 1

            print("a         = " + str(a) + " (" + str(ba2int(a)) + ")")
            print("b         = " + str(b) + " (" + str(ba2int(b)) + ")")
            print("res       = " + str(res) + " (" + str(ba2int(res)) + ")")
            print("expecting = " + str(expected) + " (" + str(ba2int(expected)) + ")")
            print("delta     = " + str(delta) + " (" + str(ba2int(delta)) + ")")

            if delta[0] == 0:
                print("\33[31mFAIL\33[0m -- \33[32m LSB PASS\33[0m")
            else:
                print("\33[31mFAIL\33[0m -- \33[32m LSB FAIL\33[0m")

    print("")

    total = failed + passed
    if passed == total:
        print("\33[32mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    elif passed == 0:
        print("\33[31mPasses " + str(passed) + " / " + str(total) + "\33[0m")
    else:
        print("\33[33mPasses " + str(passed) + " / " + str(total) + "\33[0m")

    assert passed == total

def test_xor():
    f = open("./bristol/simple_xor.txt")
    c = Circuit(f.read())

    xorPass = 0
    xorFail = 0
    for num in range(2500):
        gc = garble(c)
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            res = gc.eval(int2ba(a, 1, "little"), int2ba(b, 1, "little"))
            r = ba2int(res)
            if r == (a ^ b):
                xorPass += 1
            else:
                xorFail += 1

    xorTotal = xorFail + xorPass
    if xorTotal == xorPass:
        print("\33[32mXOR PASS\33[0m --> " + str(xorPass) + " / " + str(xorTotal))
    else:
        print("\33[31mXOR FAIL\33[0m --> " + str(xorPass) + " / " + str(xorTotal))

    assert xorTotal == xorPass

def test_and():
    f = open("./bristol/simple_and.txt")
    c = Circuit(f.read())

    andPass = 0
    andFail = 0
    for num in range(2500):
        gc = garble(c)
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            res = gc.eval(int2ba(a, 1, "little"), int2ba(b, 1, "little"))
            r = ba2int(res)
            if r == (a & b):
                andPass += 1
            else:
                andFail += 1

    andTotal = andFail + andPass
    if andTotal == andPass:
        print("\33[32mAND PASS\33[0m --> " + str(andPass) + " / " + str(andTotal))
    else:
        print("\33[31mAND FAIL\33[0m --> " + str(andPass) + " / " + str(andTotal))

    assert andTotal == andPass

def test_not():
    f = open("./bristol/not.txt")
    c = Circuit(f.read())

    notPass = 0
    notFail = 0
    for num in range(2500):
        gc = garble(c)
        for a in [0, 1]:
            res = gc.eval(int2ba(a, 1, "little"))
            r = ba2int(res)
            if r == (a ^ 1):
                notPass += 1
            else:
                notFail += 1

    notTotal = notFail + notPass
    if notTotal == notPass:
        print("\33[32mNOT PASS\33[0m --> " + str(notPass) + " / " + str(notTotal))
    else:
        print("\33[31mNOT FAIL\33[0m --> " + str(notPass) + " / " + str(notTotal))

    assert notTotal == notPass