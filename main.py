import numpy as np

mu, sigma = 0, 1

def make_complex(symbol):
    return 1-2*symbol[0] + (1-2*symbol[1])*1j

def AWGN(data):
    Hi = np.random.normal(mu, sigma, 1)
    Hq = np.random.normal(mu, sigma, 1)

    return  data*(Hi+Hq*1j)
