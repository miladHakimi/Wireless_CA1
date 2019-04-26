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