import numpy as np
import matplotlib.pyplot as plt
import random

symbol_count = 500
mu, sigma = 0, 1
SNR = 10

Nsigma = 1/(SNR*1.0)

def gen_symbol():
    symbol = []
    for i in range(symbol_count):
        symbol.append((random.randint(0, 1), random.randint(0, 1)))
    return symbol

def make_complex(symbol):
    data = []
    for i in symbol:
        data.append(0.71*(1-2*i[0] + (1-2*i[1])*1j))

    return data

def ChannelGain(data):
    Hi = np.random.normal(mu, sigma, symbol_count)
    Hq = np.random.normal(mu, sigma, symbol_count)

    Hq1 = map(lambda x: x * 1j, Hq)
    H = map(lambda x,y: x+y, Hi, Hq1)

    mult = map(lambda x,y: x*y, data, H)
    return mult, H

def AWGN(data):
    Ni = np.random.normal(mu, Nsigma, symbol_count)
    Nq = np.random.normal(mu, Nsigma, symbol_count)

    Nq1 = map(lambda x: x * 1j, Nq)
    N = map(lambda x,y: x+y, Ni, Nq1)

    res = map(lambda x,y: x+y, data, N)

    return  res

data, H = ChannelGain(make_complex(gen_symbol()))    
data = AWGN(data)


X = map(lambda x, y: (x/y).real, data, H)
Y = map(lambda x, y: (x/y).imag, data, H)

plt.grid(color='r', linestyle='--', linewidth=1)
plt.scatter(X, Y, color='red')
plt.scatter([0.7, 0.7, -0.7, -0.7], [0.7, -0.7, 0.7, -0.7], color="yellow")
plt.show()
