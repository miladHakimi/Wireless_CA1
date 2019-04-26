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

def convert_to_symbol(X, Y):

    x1 = map(lambda x: 0 if x>=0 else 1, X)
    y1 = map(lambda x: 0 if x>=0 else 1, Y)
    z = map(lambda x,y: (x, y), x1, y1)

    return z

def err_prob(z, data):
    count = 0

    for i in range(len(z)):
            if z[i] != data[i]:
                count += 1

    return count/(len(data)*1.0) * 100

def gen_data(size):
    data = []
    for i in range(size):
        data.append((random.randint(0, 1)))

    return data

def gen_parity(data):
    return [data[0] ^ data[1] ^ data[2], data[0] ^ data[1] ^ data[3], data[1] ^ data[2] ^ data[3]]

def to_hamming(data):
    output = []
    
    gen_matrix = np.matrix([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )
    for i in range(0, len(data), 4):
        mat = np.array(data[i:i+4]).reshape(4, 1)
        data_out = np.dot(gen_matrix, mat).reshape(1, 7).tolist()[0]
        x = map(lambda x: x%2, data_out)

        output += x
    
    return output

def list_to_dec(data):
    out = 0
    for bit in data:
        out = (out<<1) | bit
    
    return out

def unpack(data, symbol_bits=2):
    out = []
    for i in data:
        for j in range(symbol_bits):
            out.append(i[j])

    return out

def pack_data(data):
    out = []
    for i in range(0, len(data), 2):
        out.append((data[i], data[i+1]))
    
    return out

def err_correction(data):
    output = []

    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])

    for i in range(0, len(data), 7):
        r = np.array(data[i:i+7]).reshape(7, 1)
        z = np.dot(H, r).reshape(1, 3).tolist()[0]
        z = map(lambda x: x%2, z)
        err_ind = list_to_dec(z)
        if err_ind != 0:
            r[err_ind-1] = 1 - r[err_ind-1]
        output += r.reshape(1, 7).tolist()[0]
    
    return output

def decode(data):
    out = []
    R = np.array([
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    for i in range(0, len(data), 7):
        Rr = np.dot(R, np.array(data[i:i+7]).reshape(7, 1))
        out += Rr.reshape(1, 4).tolist()[0]

    return out

def make_tuple(data):
    res = []
    for i in range(0, len(data), 2):
        res.append((data[i], data[i+1]))
    return res

def QAM_to_complex(data):
    output = []
    for i in range(0, len(data), 4):
        I = (-2*data[i+1] + 3)*(2*data[i]-1)
        Q = (-2*data[i+3] + 3)*(2*data[i+2]-1)
        output.append(1/(3*2**(1/2.0))*(I+Q*1j))

    return output
def QAM_demodulate(X, Y):
    return map(lambda x, y: (0 if x<0 else 1, 1 if (abs(x)-0.31)<0 else 0, 0 if y<0 else 1, 1 if (abs(y)-0.31)<0 else 0), X, Y)
    
def part_1(symbol_count, mu, sigma, Nsigma):

    data, H = ChannelGain(make_complex(gen_symbol(symbol_count)), mu, sigma, symbol_count)    
    data = AWGN(data, mu, Nsigma, symbol_count)

    X = map(lambda x, y: (x/y).real, data, H)
    Y = map(lambda x, y: (x/y).imag, data, H)
    
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
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
        z = convert_to_symbol(X, Y)
        
        err_mean.append(err_prob(unpack(z), unpack(symbols)))
        SNR_points.append((i)*1.0)

    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.plot(SNR_points, err_mean, color='red')
    plt.show()

def part3(test_count, symbol_count, mu, sigma):
    err_mean = []
    SNR_points = []
    for i in range(test_count):
        raw_data1 = gen_data(symbol_count)
        raw_data = to_hamming(raw_data1)
        raw_data = make_tuple(raw_data)
        data = make_complex(raw_data)
        data, H = ChannelGain(data, mu, sigma, len(data))
        data = AWGN(data, mu, 1/((i+1)*1.0), len(data))

        X = map(lambda x, y: (x/y).real, data, H)
        Y = map(lambda x, y: (x/y).imag, data, H)
        
        data = convert_to_symbol(X, Y)
        data = unpack(data)
        data = err_correction(data)
        data = decode(data)
        data = pack_data(data)
        err_mean.append(err_prob(raw_data1, unpack(data)))
        SNR_points.append((i)*1.0)

    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.plot(SNR_points, err_mean, color='red')
    plt.show()


def part4_1(symbol_count, mu, sigma, Nsigma):
    data = gen_data(symbol_count)
    data = QAM_to_complex(data)
    data, H = ChannelGain(data, mu, sigma, len(data))
    data = AWGN(data, mu, Nsigma, len(data))
    X = map(lambda x, y: (x/y).real, data, H)
    Y = map(lambda x, y: (x/y).imag, data, H)
    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.scatter(X, Y, color='red')
    x = (np.array([3, 3 ,3, 3, 1, 1, 1, 1, -1, -1, -1, -1, -3, -3, -3, -3])*(1/(3*2**(1/2.0)))).tolist()
    y = (np.array([3, 1, -1, -3]*4)*(1/(3*2**(1/2.0)))).tolist()

    plt.scatter(x, y, color="yellow")
    plt.show()

def part4_2(test_count, symbol_count, mu, sigma):
    err_mean = []
    SNR_points = []
    for i in range(test_count):
        raw_data1 = gen_data(symbol_count)
        raw_data = QAM_to_complex(raw_data1)
        data, H = ChannelGain(raw_data, mu, sigma, symbol_count/4)    
        data = AWGN(data, mu, 1/((i+0+1)*2.0), symbol_count/4)

        X = map(lambda x, y: (x/y).real, data, H)
        Y = map(lambda x, y: (x/y).imag, data, H)
        z = QAM_demodulate(X, Y)

        err_mean.append(err_prob(unpack(z, 4), raw_data1))
        SNR_points.append((i/2))

    plt.grid(color='r', linestyle='--', linewidth=1)
    plt.plot(SNR_points, err_mean, color='red')
    plt.show()