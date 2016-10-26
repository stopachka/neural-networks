import numpy as np
import scipy.io as sio
from plot import plot_perceptron

# -----------------------------------------------------------------------------
# learn

def load_data(name):
    return sio.loadmat('coursera/Datasets/' + name)

def add_bias(examples):
    return np.column_stack([examples, np.ones(examples.shape[0])])

def activation(example, w):
    return np.dot(example, w)[0]

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

def update_weights(neg_examples, pos_examples, w, learn_rate):
    def apply_neg(w, example):
        a = activation(example, w)
        if a >= 0:
            return w + learn_rate * np.column_stack(example).T * (0.0 - a)
        else:
            return w

    def apply_pos(w, example):
        a = activation(example, w)
        if a < 0:
            return w + learn_rate * np.column_stack(example).T * (1.0 - a)
        else:
            return w

    return reduce(apply_pos, pos_examples, reduce(apply_neg, neg_examples, w))

def learn(
    neg_examples,
    pos_examples,
    w,
    feasible,
    error_history = [],
    weight_dist_history = [],
    learn_rate = 1/2.0
):
    def recur(w, error_history, weight_dist_history):
        (error_history, weight_dist_history) = display(
            neg_examples,
            pos_examples,
            w,
            feasible,
            error_history,
            weight_dist_history
        )

        new_w = update_weights(neg_examples, pos_examples, w, learn_rate)

        choice = raw_input('Continue? (y/n)')

        if choice == 'y':
            recur(new_w, error_history, weight_dist_history)
        else:
            print 'Got ', choice, '. See ya!'

    recur(w, error_history, weight_dist_history)


# -----------------------------------------------------------------------------
# display

def display(
    neg_examples,
    pos_examples,
    w,
    feasible,
    error_history,
    weight_dist_history
):
    (neg_mistakes, pos_mistakes) = eval_weight(neg_examples, pos_examples, w)

    error_history = error_history + [len(neg_mistakes) + len(pos_mistakes)]
    weight_dist_history = weight_dist_history + [np.linalg.norm(w - feasible)]

    print 'negative sample errors: ', len(pos_mistakes)
    print 'positive sample errors: ', len(pos_mistakes)
    print 'weights: ', w,

    plot_perceptron(
        neg_examples,
        pos_examples,
        neg_mistakes,
        pos_mistakes,
        error_history,
        w,
        weight_dist_history
    )

    return (error_history, weight_dist_history)

# -----------------------------------------------------------------------------
# init

def init():
    dataset = load_data('dataset1.mat')

    neg_examples = add_bias(dataset['neg_examples_nobias'])
    pos_examples = add_bias(dataset['pos_examples_nobias'])
    initial_w = dataset['w_init']
    feasible_w = dataset['w_gen_feas']

    learn(neg_examples, pos_examples, initial_w, feasible_w)

init()
