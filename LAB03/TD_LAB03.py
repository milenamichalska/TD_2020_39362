import numpy as np
import matplotlib.pyplot as plt

# 39362
a = 2
b = 6
c = 3

A = 1.0
f = b
fi = c * np.pi
n = np.linspace(0, a * 100 + b * 10 + c)

# Funkcja rysująca
def draw_plot(x,y,z):
    plt.plot(x,y,z)
    plt.show()
 
def DFT(x):
    # x(k) - próbki harmoniczne
    xk = []
    #N - liczba próbek
    N = len(x)

    for k in range(N):
        sum = 0
        for n in range(N):
            #wn - współczynnik skrętu
            wn = np.cos(x[n]) + n*np.sin(x[n])
            sum += x[n] * wn**(-k*n)
        xk.append(sum)
    return xk

def s(A,f,fi,t):
    return A * np.sin(2 * np.pi * f * t + fi)

def zad2():
    #n w zakresie (0, ABC)
    t = np.linspace(0, 1, a * 100 + b * 10 + c)
    #ton prosty z poprzedniego zadania
    plt.plot(t, s(A, f, t, fi))
    plt.show()

zad2()