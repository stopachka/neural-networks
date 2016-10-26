import scipy.io as sio
import numpy as np

# -----------------------------------------------------------------------------
# Learn

def load_data(name):
    return sio.loadmat('coursera/Datasets/' + name)

def add_bias(examples):
    return np.column_stack([examples, np.ones(examples.shape[0])])

def activation(example, w):
    return np.dot(example, w)

def eval_weight(neg_examples, pos_examples, w):
    def mistakes(f, examples):
        return map(
            lambda (idx, row): idx,
            filter(
                lambda (idx, row): f(activation(row, w)),
                enumerate(examples)
            )
        )

    neg_mistakes = mistakes(lambda a: a >= 0, neg_examples)
    pos_mistakes = mistakes(lambda a: a < 0, pos_examples)

    return (neg_mistakes, pos_mistakes)

def learn(datastet):
    neg_examples = add_bias(datastet['neg_examples_nobias'])
    pos_examples = add_bias(datastet['pos_examples_nobias'])
    initial_w = datastet['w_init']
    feasible_w = datastet['w_gen_feas']

# -----------------------------------------------------------------------------
# Plot

# -----------------------------------------------------------------------------
# Init

def init():
    learn(load_data('dataset1.mat'))

init()
