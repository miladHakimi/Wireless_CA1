import random
import numpy as np
import matplotlib.pyplot as plt

def gen_symbol(symbol_count):
    symbol = []
    for i in range(symbol_count):
        symbol.append((random.randint(0, 1), random.randint(0, 1)))
    return symbol

def make_complex(symbol):
    data = []
    for i in symbol:
        data.append(0.71*(1-2*i[0] + (1-2*i[1])*1j))

    return data

def ChannelGain(data, mu, sigma, symbol_count):
    Hi = np.random.normal(mu, sigma, symbol_count)
    Hq = np.random.normal(mu, sigma, symbol_count)

    Hq1 = map(lambda x: x * 1j, Hq)
    H = map(lambda x,y: x+y, Hi, Hq1)

    mult = map(lambda x,y: x*y, data, H)
    return mult, H

def AWGN(data, mu, Nsigma, symbol_count):
    Ni = np.random.normal(mu, Nsigma, symbol_count)
    Nq = np.random.normal(mu, Nsigma, symbol_count)

    Nq1 = map(lambda x: x * 1j, Nq)
    N = map(lambda x,y: x+y, Ni, Nq1)

    res = map(lambda x,y: x+y, data, N)

    return  res

def err_prob(X, Y, data):
    count = 0

    x1 = map(lambda x: 0 if x>=0 else 1, X)
    y1 = map(lambda x: 0 if x>=0 else 1, Y)
    z = map(lambda x,y: (x, y), x1, y1)

    for i in range(len(z)):
        for j in range(2):
            if z[i][j] != data[i][j]:
                count += 1/2.0

    return count/(len(data)) * 100

def part_1(symbol_count, mu, sigma, Nsigma):

    data, H = ChannelGain(make_complex(gen_symbol(symbol_count)), mu, sigma, symbol_count)    
    data = AWGN(data, mu, Nsigma, symbol_count)

    X = map(lambda x, y: (x/y).real, data, H)
    Y = map(lambda x, y: (x/y).imag, data, H)
    
    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.scatter(X, Y, color='red')
    plt.scatter([0.7, 0.7, -0.7, -0.7], [0.7, -0.7, 0.7, -0.7], color="yellow")
    plt.show()


def part_2(test_count, symbol_count, mu, sigma):
    err_mean = []
    SNR_points = []
    for i in range(test_count):
    
        symbols = gen_symbol(symbol_count)
        data, H = ChannelGain(make_complex(symbols), mu, sigma, symbol_count)    
        data = AWGN(data, mu, 1/((i/5.0+1)*1.0), symbol_count)

        X = map(lambda x, y: (x/y).real, data, H)
        Y = map(lambda x, y: (x/y).imag, data, H)
        
        err_mean.append(err_prob(X, Y, symbols))
        SNR_points.append((i)*1.0)

    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.plot(SNR_points, err_mean, color='red')
    plt.show()
