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

#dyskretna transformata fouriera z poprzedniego laboratorium
def DFT(x):
    xk=[]
    N=len(x)                    
    wn=np.exp((1j*2*np.pi)/N)
    for k in range(N):
        sum=0
        for n in range(N):
            sum+=x[n]*wn**(-k*n)
        xk.append(sum)
    return xk

def pasmo(mk_p):
    fmin = np.min(mk_p)
    fmax = np.max(mk_p)
    W = fmax - fmin
    print(W)

A1 = 0.2
A2 = 0.8
Tb = 2

def zad2():
    N = 1 / Tb

    l = len(m)
    
    t = np.linspace(0, l, 50 * (l // Tb))
    x = np.linspace(0, l, l)

    print(m.size)
    print(x.size)

    f = N * (Tb**-1)
    f0 = (N + 1) / Tb
    f1 = (N + 2) / Tb

    interp = interp1d (x, m, kind='previous')
    Tbs = interp(t)

    plt.title('Sygnał informacyjny m(t)')
    plt.plot(t, Tbs)
    plt.show()

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

zad2()

def zad345():
    N=2
    m1 = m[:10]

    t = np.linspace(0, 10, 50 * (10 // Tb))
    x = np.linspace(0, 10, 10)

    f = N * (Tb**-1)
    f0 = (N + 1) / Tb
    f1 = (N + 2) / Tb

    interp = interp1d (x, m1, kind='previous')
    Tbs = interp(t)

    ASK = []
    for i, j in zip(Tbs,t):
        if i==1:
            ASK.append(A1 * np.sin(2 * np.pi * j * f * np.pi))
        if i==0:
            ASK.append(A2 * np.sin(2 * np.pi * j * f * np.pi))

    plt.title('Kluczowanie ASK')
    plt.plot(t, ASK)
    plt.show()

    widmoASK = np.abs(DFT(ASK))
    plt.title('Widmo ASK')
    plt.plot(t, widmoASK)
    plt.show()

    pasmo(ASK)

    FSK = []
    for i, j in zip(Tbs,t):
        if i==1:
            FSK.append(A1 * np.sin(2 * np.pi * j * f1 * np.pi))
        if i==0:
            FSK.append(A2 * np.sin(2 * np.pi * j * f0 * np.pi))

    plt.title('Kluczowanie FSK')
    plt.plot(t, FSK)
    plt.show()

    widmoFSK = np.abs(DFT(FSK))
    plt.title('Widmo FSK')
    plt.plot(t, widmoFSK)
    plt.show()

    pasmo(FSK)

    PSK = []
    for i, j in zip(Tbs,t):
        if i==1:
            PSK.append(A1 * np.sin(2 * np.pi * j * 1 * np.pi))
        if i==0:
            PSK.append(A2 * np.sin(2 * np.pi * j * 1 * 0))

    plt.title('Kluczowanie PSK')
    plt.plot(t, PSK)
    plt.show()

    widmoPSK = np.abs(DFT(PSK))
    plt.title('Widmo PSK')
    plt.plot(t, widmoPSK)
    plt.show()

    pasmo(PSK)

zad345()
