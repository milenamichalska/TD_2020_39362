import numpy as np
import matplotlib.pyplot as plt
import binascii
from scipy.interpolate import interp1d
from scipy import signal

#ascii to binary stream
def S2BS(str, switch):
    if (switch == 'littleEndian'):
        out = bin(int.from_bytes(str.encode(), 'little'))
    if (switch == 'bigEndian'):
        out = bin(int.from_bytes(str.encode(), 'big'))
    print (out)
    return out

#etap 1 - zamiana na binarne

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

#etap 2 - kodowanie Hamminga
mHamming = kodowanie(m)
print(mHamming)
mHamming = np.array(list(mHamming))

def tile(value, count):
    return [value for __ in range(int(count))]

def CLK(fs, samples, len):
    t = np.linspace(0, fs * len, samples * len, endpoint=True)
    clkSignal = signal.square(2 * np.pi * samples * t)
    return t, clkSignal

def TTL(signal, fs, samples): 
    time = np.linspace(0, fs * len(signal), samples * len(signal))

    s_samples = np.array(range(samples * len(signal)))
    for i, bit in enumerate(signal):
        s_samples[i * samples : (i + 1) * samples] = tile(bit, samples)
    return time, s_samples

#etap3 - zamiana na sygnał TTL i modulacja

fs = 10000

t, ttl = TTL(mHamming, fs, 1000)
plt.plot(t, ttl)
plt.show()

A1 = 1.0
A2 = 0.0
Tb = 0.1

N = 1 / Tb

l = len(mHamming)
    
t = np.linspace(0, l, int(50 * (l / Tb)))
x = np.linspace(0, l, l)

f = 0.2
f0 = 0.4
f1 = 0.7

interpolacja = interp1d (x, mHamming, kind='previous')
Tbs = interpolacja(t)

ASK = []
for i, j in zip(Tbs,t):
    if i==1:
        ASK.append(A1 * np.sin(2 * np.pi * j * f * np.pi))
    if i==0:
        ASK.append(A2 * np.sin(2 * np.pi * j * f * np.pi))

plt.title('Kluczowanie ASK')
plt.plot(t, ASK)
plt.show()

demodulacjaASK=[]

for i, j in zip(ASK, t):
    demodulacjaASK.append(A1 * np.sin(2 * np.pi * i * j * f * np.pi))

plt.title('Demodulacja ASK')
plt.plot(t, demodulacjaASK)
plt.show()

FSK = []
for i, j in zip(Tbs,t):
    if i==1:
        FSK.append(A1 * np.sin(2 * np.pi * j * f1 * np.pi))
    if i==0:
        FSK.append(A2 * np.sin(2 * np.pi * j * f0 * np.pi))

plt.title('Kluczowanie FSK')
plt.plot(t, FSK)
plt.show()

demodulacjaFSK_f0 = []
demodulacjaFSK_f1 = []
for i,j in zip(FSK, t):
    demodulacjaFSK_f0.append(A1 * np.sin(2 * np.pi * i * j * f1 * np.pi))
    demodulacjaFSK_f1.append(A2 * np.sin(2 * np.pi * i * j * f0 * np.pi))

plt.title('Demodulacja FSK x1')
plt.plot(t, demodulacjaFSK_f0)
plt.show()

plt.title('Demodulacja FSK x2')
plt.plot(t, demodulacjaFSK_f1)
plt.show()

PSK = []
for i, j in zip(Tbs,t):
    if i==1:
        PSK.append(A1 * np.sin(2 * np.pi * j * 1 * np.pi))
    if i==0:
        PSK.append(A2 * np.sin(2 * np.pi * j * 1 * 0))

plt.title('Kluczowanie PSK')
plt.plot(t, PSK)
plt.show()

demodulacjaPSK=[]
for i,j in zip(PSK, t):
    demodulacjaPSK.append(A1 * np.sin(2 * np.pi * i * j * f * np.pi))
    
plt.title('Demodulacja PSK')
plt.plot(t, demodulacjaPSK)
plt.show()

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

mHammingD = dekodowanie(mHamming)
print (''.join(m))
print (mHammingD)

t, ttl = TTL(mHammingD, fs, 1000)
plt.plot(t, ttl)
plt.show()
