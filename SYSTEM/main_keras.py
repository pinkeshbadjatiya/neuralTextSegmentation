#!/usr/bin/python2

from keras.models import Sequential
from keras.layers import Dense
import numpy
import pdb
from data_handler import get_input
from scoring import *

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

WINDOW_SIZE_winddiff_metric = 15

def window_diff_metric(y_true, y_pred):
    yT = T.concatenate([np.array([0]), T.extra_ops.cumsum(y_true, axis=0)])
    yP = T.concatenate([np.array([0]), T.extra_ops.cumsum(y_pred, axis=0)])

    winT = (yT - T.roll(yT, WINDOW_SIZE_winddiff_metric))[WINDOW_SIZE_winddiff_metric:]
    winP = (yP - T.roll(yP, WINDOW_SIZE_winddiff_metric))[WINDOW_SIZE_winddiff_metric:]

    result = T.mean(T.eq(winT - winP, 0))
    return {
        'WinDiff': result,
    }

def run_neural_net(X_train, Y_train, X_test, Y_test):
    # Rows are samples, columns are features

    INPUT_NODES = X_train.shape[1]
    OUTPUT_NODES = len(Y_train[0])      # Earlier it was 1

    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=INPUT_NODES, init='uniform', activation='relu'))
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall, precision, window_diff_metric])

    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=10, batch_size=10)

    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    predictions = model.predict(X_test)         # calculate predictions
    #rounded = [round(x) for x in predictions]  # round predictions
    print(predictions)
    pdb.set_trace()


def sample_data():
    # load pima indians dataset
    dataset = numpy.loadtxt("/home/pinkesh/DATASETS/PIMA_DATASET/pima-indians-diabetes.data", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    return X, Y


if __name__=="__main__":
    #X, Y = sample_data()
    X, Y = get_input(shuffle=False)

    # Split test-train data
    train_ratio = 0.8
    print 'X(train)=', X.shape[0]*train_ratio
    print 'X(test)=', X.shape[0]*(1-train_ratio)
    train_samples = int(train_ratio * X.shape[0])

    #pdb.set_trace()
    run_neural_net(X[:train_samples,:], Y[:train_samples], X[train_samples:,:], Y[train_samples:])
