import numpy as np
import matplotlib.pyplot as plt
import random

#R = np.arange(1, 100000, 1)
#P = np.load('pred.npy')
#T = np.load('truth.npy')


def plot(truth, pred):
    #delta = 45
    #i = int(random.randint(0, truth.shape[0]-1))
    #j = i+delta
    X = np.arange(1, truth.shape[0]+1)

    #A, = plt.plot(X[i:j], truth[i:j], 'bo-', label='Truth')
    #B, = plt.plot(X[i:j], pred[i:j], 'ro-', label='Predictions')
    print truth.shape, pred.shape
    #A, = plt.plot(X, truth, 'bo-', label='Truth')
    #B, = plt.plot(X, pred, 'ro-', label='Predictions')
    i=1
    print "Top: truth"
    print "Bottom: pred"
    plt.axvline(x=0, ymin=0, ymax=1, linewidth=0.2, color='k')
    for (t,p) in zip(truth, pred):
        t, p = int(t), int(p)
        print t, p
        plt.axvline(x=i, ymin=0, ymax = 1, linewidth=0.2, color='k')
        if t:
            plt.axvline(x=i, ymin=0.5, ymax=1, linewidth=3, color='k')
        if p >= 0.5:
            plt.axvline(x=i, ymin=0, ymax=0.5, linewidth=3, color='g')
        i=i+1
    plt.axvline(x=i, ymin=0, ymax=1, linewidth=0.2, color='k')

    #plt.legend([A, B], ['Truth', 'Predictions'])
    plt.show()


if __name__ == "__main__":
    pre = np.array([1,0,0,0,0,1,0,0,1])
    tru = np.array([0,0,1,0,1,0,0,1,0])
    plot(tru, pre)

