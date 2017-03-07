import numpy as np
import matplotlib.pyplot as plt
import random

R = np.arange(1, 100000, 1)
P = np.load('pred.npy')
T = np.load('truth.npy')


def plot(truth, pred):
    delta = 45
    i = int(random.randint(0, truth.shape[0]-1))
    j = i+delta
    X = np.arange(1, truth.shape[0]+1)

    A, = plt.plot(X[i:j], truth[i:j], 'bo-', label='Truth')
    B, = plt.plot(X[i:j], pred[i:j], 'ro-', label='Predictions')
    plt.legend([A, B], ['Truth', 'Predictions'])
    plt.show()
