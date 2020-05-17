import numpy as np
import matplotlib.pyplot as plt

# 39362
b = 6
c = 3

A = 1.0
f = b
fi = c * np.pi

def s(A,f,fi,t):
    return A * np.sin(2 * np.pi * f * t + fi)

def zad1():
    #sygnal sprobkowany
    t = np.linspace(0,2,200)
    plt.plot(t, s(A, f, t, fi))
    plt.show()

def zad2():
    #kwantyzacja q16
    q=16
    t = np.linspace(0,1,200)
    plt.step(t, np.round((s(A,f,t,fi)+A)*(q**2/2)))
    plt.plot(t, np.round((s(A,f,t,fi)+A)*(q**2/2)))
    plt.show()
    
def zad3():
    #kwantyzacja q8
    q=8
    t = np.linspace(0,1,100)
    plt.step(t, np.round((s(A,f,t,fi)+A)*(q**2/2)))
    plt.plot(t, np.round((s(A,f,t,fi)+A)*(q**2/2)))
    plt.show()

zad1()
zad2()
zad3()
