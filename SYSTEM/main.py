#########################
# Theano for Training a
# Neural Network on MNIST
#########################

import numpy as np

import theano
import theano.tensor as tensor
from plot import plot

rng = np.random.RandomState(1234)

HIDDEN_UNITS = 3
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


#w = theano.shared(np.random.normal(loc=0, scale=.1, size=(INPUT_DIM, HIDDEN_UNITS)))
#b = theano.shared(np.zeros(HIDDEN_UNITS))
#v = theano.shared(np.zeros((HIDDEN_UNITS, OUTPUT_UNITS)))
#c = theano.shared(np.zeros(OUTPUT_UNITS))

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

if __name__ == "__main__":
    # now do the computations
    errors = []
    epocs = 10000
    for i in xrange(epocs):
        #print x[i, i * batchsize: (i + 1) * batchsize].shape
        #x_i = x[i, i * batchsize: (i + 1) * batchsize].reshape((batchsize, 1))
        #y_i = y[i].reshape((1,1))
        err_i = train(x, y)
        errors.append(err_i)        
        print err_i
    plot(errors, range(1, epocs+1))
