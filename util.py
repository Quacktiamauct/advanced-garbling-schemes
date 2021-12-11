from Crypto.Cipher import AES
from Crypto.Hash import CMAC
import hashlib

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


prf_called = 0

def prf(string):
    """
    Hash a string using SHA3-256
    :param string: The string to hash
    :return: The hashed string
    """
    # global prf_called
    # prf_called += 1
    return hashlib.sha256(string).digest()



if __name__ == '__main__':
    pass
