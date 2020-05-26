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
 
def DFT(x):
    # x(k) - próbki harmoniczne
    xk = []
    # N - liczba próbek
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

# skala częstotliwości
def scale_f(x,fs):
    N = len(x)
    fk = []
    for k in range(N):
        fk.append(k * (fs / N))
    return fk

def zad2():
    # n w zakresie (0, ABC)
    t = np.linspace(0, 1, a * 100 + b * 10 + c)
    # ton prosty z poprzedniego zadania
    plt.plot(t, s(A, f, t, fi))
    plt.show()

    imag = np.imag(DFT(s(A, f, fi, n))) # liczby urojone
    real = np.real(DFT(s(A, f, fi, n))) # liczby rzeczywiste

    mk = [] # widmo amplitudowe
    for i in range(0, len(real)):
        mk.append(np.sqrt(real[i]**2 + imag[i]**2))

    mkprim = 10 * np.log10(mk) # skala decybelowa
    plt.bar(scale_f(DFT(s(A, f, n, fi)), 100), mkprim)
    plt.show()

zad2()

def x(t):
    return a * t**2 + b * t + c

def y(t):
    return 2 * x(t) * x(t) + 12 + np.cos(t)

def z(t):
    return np.sin(2 * np.pi * 7 * t ) * x(t) - 0.2 * np.log10(np.fabs(y(t) + np.pi))

def u(t):
    return np.sin(np.abs(y(t) * y(t) * z(t))) - 1.8 * np.sin(0.4 * t * z(t) * x(t))

def v(t):
    if (t < 0.22) and (t >= 0):
        return (1 - 7*t) * np.sin((2 * np.pi * t * 10) / (t + 0.04))
    if (t < 0.7) and (t >= 0.22):
        return 0.63 * t * np.sin(125 * t)
    if (t <= 1.0) and (t >= 0.7):
        return t**(-0.662) + 0.77 * np.sin(8 * t)

def p(t,j):
    sum=0
    for n in np.arange(1, j+1):
        sum += ((np.cos(12 * t * n**2) + np.cos(16*t*n)) / n**2)
    return sum

def showPlot(x,y):
    plt.plot(x,y)
    plt.show()

def zad3():
    w = np.linspace(0, 1, 22050)
    t = np.linspace(-10, 10, 100)
    showPlot(scale_f(x(t), 100), DFT(x(t)))

    t = np.linspace(0,1,500)
    showPlot(scale_f(y(t), 5000), DFT(y(t)))
    showPlot(scale_f(z(t), 4000), DFT(z(t)))
    showPlot(scale_f(u(t), 3000), DFT(u(t)))

    # h=[]
    # for i in w:
    #     h.append(v(i))
    # showPlot(scale_f(h,2000),DFT(h))

    showPlot(scale_f(p(t, 3), 2000), DFT(p(t, 3)))
    showPlot(scale_f(p(t, 9), 1000), DFT(p(t, 9)))
    showPlot(scale_f(p(t, 26), 1000),DFT(p(t, 26))) 

zad3()

def IDFT(x):
    xk = []
    N = len(x)
    for k in range(N):
        sum = 0
        for n in range(N):
            wn = np.cos(x[n]) + n * np.sin(x[n])
            sum += 1/N + x[n] * wn**(k * n)
        xk.append(sum)
    return xk

def zad4():
    showPlot(n, IDFT(s(A,f,n,fi)))

zad4()
    
