from keras.models import Sequential
from keras.layers import Dense
import numpy
import pdb
from data_handler import get_input
from scoring import *

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)



def run_neural_net(X_train, Y_train, X_test, Y_test):
    # Rows are samples, columns are features
    
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], init='uniform', activation='relu'))
    #model.add(Dense(X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    #model = Sequential([
    #    Dense(32, input_dim=X.shape[1], init='uniform'),
    #    Activation('relu'),
    #    #Dense(10, init='uniform'),
    #    #Activation('relu'),
    #    Dense(1, init='uniform'),
    #    Activation('sigmoid'),
    #])
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall, precision])
    
    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=10, batch_size=10)
    
    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # calculate predictions
    predictions = model.predict(X_test)
    # round predictions
    #rounded = [round(x) for x in predictions]
    print(predictions)
    pdb.set_trace()


def sample_data():
    # load pima indians dataset
    dataset = numpy.loadtxt("/home/pinkesh/DATASETS/PIMA_DATASET/pima-indians-diabetes.data", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    return X, Y
        

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], numpy.asarray(b)[p]


if __name__=="__main__":
    #X, Y = sample_data()
    X, Y = get_input()
    X, Y = unison_shuffled_copies(X, Y)

    # Split test-train data
    train_ratio = 0.8
    print 'X(train)=', X.shape[0]*train_ratio
    print 'X(test)=', X.shape[0]*(1-train_ratio)
    train_samples = int(train_ratio * X.shape[0])
    
    #pdb.set_trace()
    run_neural_net(X[:train_samples,:], Y[:train_samples], X[train_samples:,:], Y[train_samples:])

