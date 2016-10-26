import numpy as np
import matplotlib.pyplot as plt

# Code taken from
# https://github.com/bpjsincl/coursera-neural-net/blob/master/assignment1/assignment1.py
# Did not wish to rewrite this, as it doesn't help with neural network related learning

def plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1,
                    num_err_history, w, w_dist_history):
    """The top-left plot shows the dataset and the classification boundary given by
    the weights of the perceptron. The negative examples are shown as circles
    while the positive examples are shown as squares. If an example is colored
    green then it means that the example has been correctly classified by the
    provided weights. If it is colored red then it has been incorrectly classified.
    The top-right plot shows the number of mistakes the perceptron algorithm has
    made in each iteration so far.
    The bottom-left plot shows the distance to some generously feasible weight
    vector if one has been provided (note, there can be an infinite number of these).
    Points that the classifier has made a mistake on are shown in red,
    while points that are correctly classified are shown in green.
    The goal is for all of the points to be green (if it is possible to do so).
    Args:
        neg_examples    : The num_neg_examples x 3 matrix for the examples with target 0.
                          num_neg_examples is the number of examples for the negative class.
        pos_examples    : The num_pos_examples x 3 matrix for the examples with target 1.
                          num_pos_examples is the number of examples for the positive class.
        mistakes0       : A vector containing the indices of the datapoints from class 0 incorrectly
                          classified by the perceptron. This is a subset of neg_examples.
        mistakes1       : A vector containing the indices of the datapoints from class 1 incorrectly
                          classified by the perceptron. This is a subset of pos_examples.
        num_err_history : A vector containing the number of mistakes for each
                          iteration of learning so far.
        w               : A 3-dimensional vector corresponding to the current weights of the
                          perceptron. The last element is the bias.
        w_dist_history  : A vector containing the L2-distance to a generously
                          feasible weight vector for each iteration of learning so far.
                          Empty if one has not been provided.
    """
    f = plt.figure(1)

    neg_correct_ind = np.setdiff1d(range(len(neg_examples)), mistakes0)
    pos_correct_ind = np.setdiff1d(range(len(pos_examples)), mistakes1)
    assert all(m_idx not in set(neg_correct_ind) for m_idx in mistakes0) and \
        all(m_idx not in set(pos_correct_ind) for m_idx in mistakes1)

    plt.subplot(2,2,1)
    plt.hold(True)
    if np.size(neg_examples):
        plt.plot(neg_examples[neg_correct_ind][:, 0], neg_examples[neg_correct_ind][:, 1], 'og', markersize=10)
    if np.size(pos_examples):
        plt.plot(pos_examples[pos_correct_ind][:, 0], pos_examples[pos_correct_ind][:, 1], 'sg', markersize=10)

    if len(mistakes0):
        plt.plot(neg_examples[mistakes0][:, 0], neg_examples[mistakes0][:, 1], 'or', markersize=10)
    if len(mistakes1):
        plt.plot(pos_examples[mistakes1][:, 0], pos_examples[mistakes1][:, 1], 'sr', markersize=10)

    plt.title('Perceptron Classifier')
    # In order to plot the decision line, we just need to get two points.
    plt.plot([-5, 5], [(-w[-1] + 5 * w[0]) / w[1], (-w[-1] - 5 * w[0]) / w[1]], 'k')
    plt.xlim([-1,4])
    plt.ylim([-2,2])
    plt.hold(False)

    plt.subplot(2,2,2)
    plt.plot(range(len(num_err_history)), num_err_history)
    plt.xlim([-1, max(15, len(num_err_history))])
    plt.ylim([0, len(neg_examples) + len(pos_examples) + 1])
    plt.title('Number of errors')
    plt.xlabel('Iteration')
    plt.ylabel('Number of errors')

    plt.subplot(2,2,3)
    plt.plot(range(len(w_dist_history)), w_dist_history)
    plt.xlim([-1, max(15, len(num_err_history))])
    plt.ylim([0, 15])
    plt.title('Distance')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.show()
