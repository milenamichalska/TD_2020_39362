import numpy as np
import matplotlib.pyplot as plt
import binascii
from scipy.interpolate import interp1d

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

m = np.array(list(m)[2:])

A1 = 0.2
A2 = 0.8
Tb = 2

N = 1 / Tb

l = len(m)
    
t = np.linspace(0, l, 50 * (l // Tb))
x = np.linspace(0, l, l)

f = N * (Tb**-1)
f0 = (N + 1) / Tb
f1 = (N + 2) / Tb

interpolacja = interp1d (x, m, kind='previous')
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

FSK = []
for i, j in zip(Tbs,t):
    if i==1:
        FSK.append(A1 * np.sin(2 * np.pi * j * f1 * np.pi))
    if i==0:
        FSK.append(A2 * np.sin(2 * np.pi * j * f0 * np.pi))

plt.title('Kluczowanie FSK')
plt.plot(t, FSK)
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

def zad234():
    demodulacjaASK=[]

    for i, j in zip(ASK, t):
        demodulacjaASK.append(A1 * np.sin(2 * np.pi * j * f * np.pi))

    plt.title('Demodulacja ASK')
    plt.plot(t, demodulacjaASK)
    plt.show()

    demodulacjaFSK_f0 = []
    demodulacjaFSK_f1 = []
    for i,j in zip(FSK, t):
        demodulacjaFSK_f0.append(A1 * np.sin(2 * np.pi * j * f1 * np.pi))
        demodulacjaFSK_f1.append(A2 * np.sin(2 * np.pi * j * f0 * np.pi))

    plt.title('Demodulacja FSK x1')
    plt.plot(t, demodulacjaFSK_f0)
    plt.show()

    plt.title('Demodulacja FSK x2')
    plt.plot(t, demodulacjaFSK_f1)
    plt.show()

    demodulacjaPSK=[]
    for i,j in zip(PSK, t):
        demodulacjaPSK.append(A1 * np.sin(2 * np.pi * j * f * np.pi))
    
    plt.title('Demodulacja PSK')
    plt.plot(t, demodulacjaPSK)
    plt.show()

    #zadanie3
    cASK=[]
    
    for i in range(l):
        xi = 0
        for j in range(20):
            xi = xi + demodulacjaASK[(i * 20) + j]
        cASK.append(xi)

    interpolacjaASK = interp1d(x, cASK, kind='previous')
    TbsASK = interpolacjaASK(t)

    plt.title('Demodulacja ASK dla p(t)')
    plt.plot(t, TbsASK)
    plt.show()

    cFSK_f0 = []

    for i in range(l):
        x0 = 0
        for j in range(20):
            x0 = x0 + demodulacjaFSK_f0[(i * 20) + j]
            cFSK_f0.append(x0)

    cFSK_f1 = []
    for i in range(l):
        x1 = 0
        for j in range(20):
            x1 = x1 + demodulacjaFSK_f1[(i * 20) + j]
        cFSK_f1.append(x1)
        
    cFSK = []
    for i in range(l):
        cFSK.append(cFSK_f0[i] - cFSK_f1[i])
        
    interpolacjaFSK = interp1d(x, cFSK, kind='previous')
    TbsFSK = interpolacjaFSK(t)

    plt.title('Demodulacja FSK dla p(t)')
    plt.plot(t, TbsFSK)
    plt.show()

    cPSK = []
    for i in range(l):
        xi = 0
        for j in range(20):
            xi = xi + demodulacjaPSK[(i * 20) + j]
        cPSK.append(xi)

    interpolacjaPSK = interp1d(x, cPSK, kind='previous')
    TbsPSK = interpolacjaPSK(t)

    plt.title('Demodulacja PSK dla p(t)')
    plt.plot(t, TbsPSK)
    plt.show()

    #zadanie 4
    h = 0.4
    
    progASK = []
    for p in TbsASK:
        if p < h:
            p = 0
        else:
            p = 1
        progASK.append(p)

    plt.title('Demodulacja ASK dla m(t)')
    plt.plot(t, progASK)
    plt.show()

    progFSK = []
    for p in TbsFSK:
        if p < h:
            p = 0
        else:
            p = 1
        progFSK.append(p)

    plt.title('Demodulacja FSK dla m(t)')
    plt.plot(t, progFSK)
    plt.show()

    progPSK = []
    for p in TbsPSK:
        if p < h:
            p = 0
        else:
            p = 1
        progPSK.append(p)

    plt.title('Demodulacja PSK dla m(t)')
    plt.plot(t, progPSK)
    plt.show()

zad234()
