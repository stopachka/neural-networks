import numpy as np
import scipy.io as sio

# -----------------------------------------------------------------------------
# transform data

def batch(dataset):
    sequences = {key: dict() for key in ['train', 'valid', 'test']}
    print sequences


# -----------------------------------------------------------------------------
# init

def init():
    dataset = sio.loadmat('coursera/data.mat')['data']
    sequences = batch(dataset)

if __name__ == '__main__':
    init()
