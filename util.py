from Crypto.Cipher import AES
from Crypto.Hash import CMAC
from hashlib import sha256
import secrets
import time


KEY = b'155E587643AAA76AD28B9A350E213E19'


def hash_aes(string: bytes):
    """
    Hash a string using AES-128-CMAC
    :param string: The string to hash
    :return: The hashed string
    """
    cmac = CMAC.new(KEY, ciphermod=AES)
    cmac.update(string)
    return cmac.digest()


def prf(string):
    """
    Hash a string using SHA3-256
    :param string: The string to hash
    :return: The hashed string
    """
    return sha256(string).digest()


if __name__ == '__main__':
    start = time.time()
    rnd = secrets.token_bytes(4096)
    for _ in range(10000):
        hash_aes(rnd)
        # h = SHA3_256.new()
        # h.update(rnd)
        # h.digest()
        # sha256(rnd).digest()
    end = time.time()
    print(f"{end-start:2.4} seconds")
