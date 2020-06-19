import numpy as np
import matplotlib.pyplot as plt
import binascii
from scipy.interpolate import interp1d
from scipy import signal

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

def tile(value, count):
    return [value for __ in range(int(count))]

def CLK(fs, samples, len):
    t = np.linspace(0, fs * len, samples * len, endpoint=True)
    clkSignal = signal.square(2 * np.pi * samples * t)
    return t, clkSignal

t, clk = CLK(5, 1000, len(m))
plt.plot(t, clk)
plt.show()

def TTL(signal, fs, samples): 
    time = np.linspace(0, fs * len(signal), samples * len(signal))

    s_samples = np.array(range(samples * len(signal)))
    for i, bit in enumerate(signal):
        s_samples[i * samples : (i + 1) * samples] = tile(bit, samples)
    return time, s_samples

t, ttl = TTL(m, 5, 1000)
plt.plot(t, ttl)
plt.show()

# następne zadania tylko dla kodowania Manchester, zgodnie z sugestią prowadzącego

def koderManchester(clk, ttl):
    manchester = []
    val = 0
    
    for i in range(len(clk) - 1):
        if (clk[i] == 1 and clk[i + 1] == -1): 
            if (ttl[i] == 0):
                val = 1
            else:
                val = -1
                
        elif (clk[i] == -1 and clk[i + 1] == 1):
            if (ttl[i] == ttl[i + 1]):
                val *= -1
        manchester.append(val)
    manchester.append(ttl[-1])
    return manchester

manchester = koderManchester(clk, ttl)
plt.plot(t, manchester)
plt.show()


def dekoderManchester(clk, manchester, samples):
    signal = []
    for i in range(len(clk) - 1):
        if (clk[i] == 1 and clk[i + 1] == 0): 
            signal.append(manchester[i])
    return signal

manchesterDekodowany = dekoderManchester(clk, manchester, 1000)
plt.plot(t, manchesterDekodowany)
plt.show()
