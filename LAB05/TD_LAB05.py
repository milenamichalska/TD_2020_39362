import numpy as np
import matplotlib.pyplot as plt
import binascii

#ascii to binary stream
def S2BS(str, switch):
    if (switch == 'littleEndian'):
        out = bin(int.from_bytes(str.encode(), 'little'))
        print (out)
        return out
    if (switch == 'bigEndian'):
        out = bin(int.from_bytes(str.encode(), 'big'))
        print (out)
        return out

S2BS('Milena', 'littleEndian')
#0b11000010110111001100101011011000110100101001101

m = S2BS('Milena', 'bigEndian')
#0b10011010110100101101100011001010110111001100001

def zad2():
    A1 = 0.2
    A2 = 0.8
    Tb = 2
    N = 1 / Tb

    t=np.linspace(0, len(m), 50 * (len(m)/Tb))
    x=np.linspace(0, len(m), len(m))

    f = N * (Tb**-1)
    f0 = (N + 1) / Tb
    f1 = (N + 2) / Tb

zad2()
