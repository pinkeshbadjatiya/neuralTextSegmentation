from keras.models import Sequential
from keras.layers import Dense
import numpy
import pdb


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def run_neural_net(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10)
    
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x) for x in predictions]
    print(rounded)



if __name__=="__main__":
    # load pima indians dataset
    dataset = numpy.loadtxt("/home/pinkesh/DATASETS/PIMA_DATASET/pima-indians-diabetes.data", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]  # Rows are samples, columns are features
    Y = dataset[:,8]


    run_neural_net(X, Y)

