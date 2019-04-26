import matplotlib.pyplot as plt
import numpy as np
from Utility import *

symbol_count = 6000
mu, sigma = 0, 1
SNR = 30
Nsigma = 1/(SNR*1.0)
test_count = 100


# part_1(symbol_count, mu, sigma, Nsigma)
# part_2(test_count, symbol_count, mu, sigma)
# part3(test_count, symbol_count, mu, sigma)
# part4(symbol_count, mu, sigma, Nsigma)

err_mean = []
SNR_points = []
for i in range(test_count):
    raw_data1 = gen_data(symbol_count)
    raw_data = QAM_to_complex(raw_data1)
    data, H = ChannelGain(raw_data, mu, Nsigma, symbol_count/4)    
    data = AWGN(data, mu, 1/((i+0+1)*5.0), symbol_count/4)

    X = map(lambda x, y: (x/y).real, data, H)
    Y = map(lambda x, y: (x/y).imag, data, H)
    z = QAM_demodulate(X, Y)
    
    print("err prob for i = " + str(i) +" = " + str(err_prob(unpack(z, 4), raw_data1)))
    err_mean.append(err_prob(unpack(z, 4), raw_data1))
    SNR_points.append((i/5))

plt.grid(color='r', linestyle='--', linewidth=1)
plt.plot(SNR_points, err_mean, color='red')
plt.show()