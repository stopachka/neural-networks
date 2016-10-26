import scipy.io as sio
import numpy as np

# -----------------------------------------------------------------------------
# Learn

def load_data(name):
    return sio.loadmat('coursera/Datasets/' + name)

def add_bias(examples):
    return np.column_stack([examples, np.ones(examples.shape[0])])

def initial_weight(w):
    if (w == None):
        return np.random.randn(3, 1)
    else:
        return w

def initial_feasible(w):
    if (w == None):
        return numpy.array([])
    else:
        return w

def activation(example, w):
    return np.dot(example, w)

def eval_weight(neg_examples, pos_examples, w):
    def mistakes(filter_fn, examples):
        res = []
        for (idx, row) in enumerate(examples):
            a = activation(row, w)
            if filter_fn(a):
                res.append(idx)

        return res

    mistakes0 = mistakes(
        lambda activation: activation >= 0,
        neg_examples
    )

    mistakes1 = mistakes(
        lambda activation: activation < 0,
        pos_examples
    )

    return (mistakes0, mistakes1)

def learn(datastet):
    neg_examples = add_bias(datastet['neg_examples_nobias'])
    pos_examples = add_bias(datastet['pos_examples_nobias'])
    initial_w = initial_weight(datastet['w_init'])
    feasible_w = initial_feasible(datastet['w_gen_feas'])

    print initial_w
    print feasible_w


# -----------------------------------------------------------------------------
# Plot

# -----------------------------------------------------------------------------
# Init

def init():
    learn()

init()
