#!/usr/bin/python2

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pdb
from sample_handler import get_input
from scoring import *
import theano

import helper


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def run_neural_net(X_train, Y_train, X_test, Y_test):
    # Rows are samples, columns are features

    INPUT_NODES = X_train.shape[1]
    OUTPUT_NODES = len(Y_train[0])      # Earlier it was 1
    print "X:", X.shape

    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=INPUT_NODES, init='uniform', activation='relu'))
    model.add(Dropout(0.8))
    #model.add(Dense(X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(OUTPUT_NODES, init='uniform', activation='sigmoid'))
    #model = Sequential([
    #    Dense(32, input_dim=X.shape[1], init='uniform'),
    #    Activation('relu'),
    #    #Dense(10, init='uniform'),
    #    #Activation('relu'),
    #    Dense(1, init='uniform'),
    #    Activation('sigmoid'),
    #])

    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall, precision, window_diff_metric, size1, size2])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall, precision])

    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=100, batch_size=10)

    # evaluate the model
    print 'Evaluating...\n'
    scores = model.evaluate(X_test, Y_test)
    print Y_test.shape 
    for (name, score) in zip(model.metrics_names, scores):
        print " %s: %0.3f" % (name, score)

    predictions = model.predict(X_test)         # calculate predictions
    rounded = np.round(predictions)
    print helper.windiff_metric_NUMPY(Y_test, rounded)
    pdb.set_trace()
    #rounded = [round(x) for x in predictions]  # round predictions
    #print(predictions)
    #pdb.set_trace()


def sample_data():
    # load pima indians dataset
    dataset = np.loadtxt("/home/pinkesh/DATASETS/PIMA_DATASET/pima-indians-diabetes.data", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    return X, Y


if __name__=="__main__":
    #X, Y = sample_data()
    SAMPLE_TYPE, X, Y = get_input(shuffle=False)

    # Split test-train data
    train_ratio = 0.8
    print 'X(train)=', X.shape[0]*train_ratio
    print 'X(test)=', X.shape[0]*(1-train_ratio)
    train_samples = int(train_ratio * X.shape[0])

    #pdb.set_trace()
    run_neural_net(X[:train_samples+1,:], Y[:train_samples+1], X[train_samples+1:,:], Y[train_samples+1:])
