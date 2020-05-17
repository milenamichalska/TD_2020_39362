import matplotlib.pyplot as plt
import numpy as np

# 39362
a = 2
b = 6
c = 3

# zadanie 1

def delta():
    delta = b**2 - 4 * (a*c)
    if delta == 0:
        x0 = (-1 * b) / (2 * a)
        print("Funkcja ma jedno miejsce zerowe ktore wynosi: ", x0)
    elif delta > 0:
        x1 = (-1 * b) + np.sqrt(delta) / (2*a)
        x2 = (-1 * b) - np.sqrt(delta) / (2*a)
        print("Funkcja ma dwa miejsca zerowe ktore wynosza: ", round(x1,2), " i ", round(x2))
    else:
        print("Delta mniejsza od 0, brak rozwiązań!")

def ekstremum():
    xe = (-1 * b) / (2*a)
    ye = a * xe**2 + b * xe + c
    print("Ekstremum w punkcie (", xe, ", ", ye, ")")

delta()
ekstremum()

r=[]
l=[]

for xp in np.arange (-10, 11, 0.01):
    yp = a * xp**2 + b * xp + c
    r.append(xp)
    l.append(yp)

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(r,l)
axes.plot
plt.show()

# zadanie 2
w = np.linspace(0, 1, 22050)
t = np.linspace(-10, 10, 100)

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

def showPlot(x,y):
    plt.plot(x,y)
    plt.show()

showPlot(w,y(w))
showPlot(w,z(w))
showPlot(w,u(w))

h=[]
for i in w:
    h.append(v(i))
showPlot(w,h)

def p(t,j):
    sum=0
    for n in np.arange(1, j+1):
        sum += ((np.cos(12 * t * n**2) + np.cos(16*t*n)) / n**2)
    return sum
showPlot(w,p(w,2))
showPlot(w,p(w,4))
showPlot(w,p(w,26))
