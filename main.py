import scipy.io as sio
import numpy as np

def load_data(name):
    return sio.loadmat('coursera/Datasets/' + name)

def add_bias(examples):
    return np.column_stack([examples, np.ones(examples.shape[0])])

def learn(datastet):
    neg_examples = add_bias(datastet['neg_examples_nobias'])
    pos_examples = add_bias(datastet['pos_examples_nobias'])
    w_init = datastet['w_init']
    w_gen_feas = datastet['w_gen_feas']
