import numpy as np
import matplotlib.pyplot as plt
import binascii
from scipy.interpolate import interp1d

#ascii to binary stream
def S2BS(str, switch):
    if (switch == 'littleEndian'):
        out = bin(int.from_bytes(str.encode(), 'little'))
    if (switch == 'bigEndian'):
        out = bin(int.from_bytes(str.encode(), 'big'))
    print (out)
    return out

S2BS('Milena', 'littleEndian')
#0b11000010110111001100101011011000110100101001101

m = S2BS('Milena', 'bigEndian')
#0b10011010110100101101100011001010110111001100001

m = np.array(list(m)[2:])

def kodowanie(s):
    b = ''
    # wczytaj po 4 bity
    while len(s) >= 4:
        porcja = s[0:4]
        b += str(hamming(porcja))
        s = s[4:]
    return b

def hamming(bits):
    t1 = str(parzystosc(bits, [0,1,3]))
    t2 = str(parzystosc(bits, [0,2,3]))
    t3 = str(parzystosc(bits, [1,2,3]))
    #zwraca podane bity + bity parzystości
    return t1 + t2 + ''.join(bits[0]) + t3 + ''.join(bits[1:])

def parzystosc(s, i):
    return (int(s[i[0]]) + int(s[i[1]]) + int(s[i[2]])) % 2

#zadanie 1
mHamming = kodowanie(m)
print(mHamming)
mHamming = np.array(list(mHamming))

# zadanie 2
def negacja(stream, bit):
    stream[bit] = 0 if (stream[bit] == 1) else 1
    return stream

def dekodowanie(s):
    b = ''
    # wczytaj po 7 bitów
    while len(s) >= 7:
        porcja = s[0:7]
        bits = hammingDecoding(porcja)
        b += bits
        s = s[7:]
    return b

def parzystoscD(s, i):
    sum = 0
    for index in i:
        sum += int(s[index]) 
    return sum % 2

def hammingDecoding(bits):
    #ponowne policzenie bitów parzystości
    t1 = int(parzystoscD(bits, [0,2,4,6]))
    t2 = int(parzystoscD(bits, [1,2,5,6]))
    t3 = int(parzystoscD(bits, [3,4,5,6]))

    n = (t1 * 2**0) + (t2 * 2**1) + (t3 * 2**2)
    if (n > 0): 
        bits = negacja(bits, n - 1)
        print('W transmisji nastąpił błąd!')
    
    return ''.join(bits[[2,4,5,6]])

#zadanie 3
mHammingD = dekodowanie(mHamming)
print (''.join(m))
print (mHammingD)

#zadanie 4

