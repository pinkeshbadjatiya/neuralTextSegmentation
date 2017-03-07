import numpy as np
from keras.utils.np_utils import to_categorical

def to_categorical_MULTI_DIM(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    
    #y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1

    sz = y.shape[:-1]
    assert y.shape[-1] == 1
    assert len(sz) == 2 # For samples and no of sentences

    categorical = np.zeros(sz + (nb_classes,))
    for i in range(sz[0]):
        categorical[i] = to_categorical(y[i], nb_classes=nb_classes) 
    return categorical

from theano.tensor import basic as tensor
import keras.backend as K

def w_binary_crossentropy(target, output):
    # Weighted binary_crossentropy
    # Where W is the loss penalty factor when a split sentence is classified as a normal sentence
    W = 5
    return K.mean(-(W * target * tensor.log(output) + (1.0 - target) * tensor.log(1.0 - output)), axis=-1)
