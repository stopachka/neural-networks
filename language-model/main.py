import numpy as np
import scipy.io as sio

# -----------------------------------------------------------------------------
# transform data

# -----------------------------------------------------------------------------
# init

def init():
    dataset = sio.loadmat('coursera/data.mat')['data']

if __name__ == '__main__':
    init()
