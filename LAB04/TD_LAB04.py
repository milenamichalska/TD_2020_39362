import numpy as np
import matplotlib.pyplot as plt

# 39362
a = 2
b = 6
c = 3
d = 9
e = 3

A = 1.0
fn = 8
fm = 3

def m(t):
    return A*np.sin(2 * np.pi * fm * t)

def z_A(t, k_A):
    return (k_A * m(t) + 1) * np.cos(2 * np.pi * fn * t)

def z_P(t, k_P):
    return np.cos(2 * np.pi * fn * t + k_P * m(t))

def zad1():
    t = np.linspace(0, 1, 200)
    plt.plot(t, m(t))
    plt.show()

    t = np.linspace(0, 1, 1000)

    #1>k_A>0; k_P<2;
    k_A = 0.7
    k_P = 1.4
    plt.plot(t, z_A(t, k_A))
    plt.show()

    plt.plot(t, z_P(t, k_P))
    plt.show()

    #12>k_A>2; \pi>k_P>0;
    k_A = 7
    k_P = 1/4 * np.pi
    plt.plot(t, z_A(t, k_A))
    plt.show()

    plt.plot(t, z_P(t, k_P))
    plt.show()

    #1k_A>BA; k_P>AB
    k_A = 64
    k_P = 32
    plt.plot(t, z_A(t, k_A))
    plt.show()

    plt.plot(t, z_P(t, k_P))
    plt.show()


zad1()

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

# skala częstotliwości
def scale_f(x,fs):
    N = len(x)
    fk = []
    for k in range(N):
        fk.append(k * (fs / N))
    return fk

def widmo(y):
    y = DFT(y)
    i_n = np.imag(y) #liczby urojone
    r_n = np.real(y) #liczby rzeczywiste
    MK = []
    for i in range(0,len(r_n)):
        MK.append(np.sqrt(r_n[i]**2+i_n[i]**2))
    return MK

def plotWidmoA(t, k_A):
    trans = DFT(z_A(t, k_A))
    mk = widmo(z_A(t, k_A))
    mk_p = 10 * np.log10(mk)
    fk = scale_f(trans, 200)
    plt.bar(fk, mk_p)
    plt.show()

def plotWidmoP(t, k_P):
    trans = DFT(z_P(t, k_P))
    mk = widmo(z_P(t, k_P))
    mk_p = 10 * np.log10(mk)
    fk = scale_f(trans, 200)
    plt.bar(fk, mk_p)
    plt.show()

def zad2():
    t = np.linspace(0, 1, 100)

    k_A = 0.7
    plotWidmoA(t, k_A)

    k_A = 7
    plotWidmoA(t, k_A)

    k_A = 64
    plotWidmoA(t, k_A)

    k_P = 1.4
    plotWidmoP(t, k_P)

    k_P = 1/4 * np.pi
    plotWidmoP(t, k_P)

    k_P = 32
    plotWidmoP(t, k_P)

zad2()

def pasmo(mk_p):
    fmin = np.min(mk_p)
    fmax = np.max(mk_p)
    W = fmax - fmin
    print(W)

def zad3():
    t = np.linspace(0, 1, 100)

    k_A = 0.7
    mk = widmo(z_A(t, k_A))
    mk_p = 10 * np.log10(mk)
    pasmo(mk_p)

    #28.430241886756765

    k_A = 7
    mk = widmo(z_A(t, k_A))
    mk_p = 10 * np.log10(mk)
    pasmo(mk_p)

    #23.83422822501708

    k_A = 64
    mk = widmo(z_A(t, k_A))
    mk_p = 10 * np.log10(mk)
    pasmo(mk_p)

    #31.98105998953324

zad3()
