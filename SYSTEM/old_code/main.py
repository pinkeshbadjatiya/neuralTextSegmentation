#########################
# Theano for Training a
# Neural Network
#########################

import numpy as np

import theano
import theano.tensor as tensor
from plot import plot

import pdb

rng = np.random.RandomState(1234)

HIDDEN_UNITS = 2
OUTPUT_UNITS = 1

x = np.asarray([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

y = np.asarray([[1], [0], [0], [1]])

SAMPLE_COUNT = x.shape[0]
INPUT_DIM = x.shape[1]

def layer(n_in, n_out=None, bias=False):
    if bias:
        assert(n_out == None)
        return theano.shared(np.zeros(n_in))
    return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=theano.config.floatX), borrow=True)

# symbol declarations
sx = tensor.matrix()
sy = tensor.matrix()

W1 = layer(INPUT_DIM, HIDDEN_UNITS)
b1 = layer(HIDDEN_UNITS, bias=True)

W2 = layer(HIDDEN_UNITS, OUTPUT_UNITS)
b2 = layer(OUTPUT_UNITS, bias=True)



# symbolic expression-building
out = tensor.tanh(tensor.dot(tensor.tanh(tensor.dot(sx, W1) + b1), W2) + b2)
err = 0.5 * tensor.sum((out - sy) ** 2)
gW1, gb1, gW2, gb2 = tensor.grad(err, [W1, b1, W2, b2])
lr = 0.01

# compile a fast training function
train = theano.function([sx, sy], err,
    updates=[
        (W1, W1 - lr * gW1),
        (b1, b1 - lr * gb1),
        (W2, W2 - lr * gW2),
        (b2, b2 - lr * gb2)])


def predict(input_x):
    out = tensor.tanh(tensor.dot(tensor.tanh(tensor.dot(input_x, W1) + b1), W2) + b2)
    print out
    return out

if __name__ == "__main__":
    # now do the computations
    errors = []
    epocs = 1000
    for i in xrange(epocs):
        err_i = train(x, y)
        errors.append(err_i)        
        print err_i
    pdb.set_trace()
    predict([[0, 0]])
    plot(range(1, epocs+1), errors)
