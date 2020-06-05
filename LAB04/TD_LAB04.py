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
    return A*np.sin(2*np.pi*fm*t)

def z_A(k_A, t):
    return (k_A * m(t) + 1) * np.cos(2*np.pi*fn*t)

def z_P(k_P, t):
    return np.cos(2*np.pi*fn*t + k_P * m(t))

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
