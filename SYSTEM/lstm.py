#from data_handler import get_data
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD

import numpy as np
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
#from gensim.parsing.preprocessing import STOPWORDS
#from keras.utils import np_utils
import codecs
#import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
import sys


import helper
from sample_handler import get_input



def lstm_model(sequence_length, embedding_dim):

    which_model = 1

    if which_model == 1:
        ngram_filters = [1, 3, 6, 9, 18]
        conv_hidden_units = [300, 200, 200, 200, 200]
    
        graph_in = Input(shape=(sequence_length, embedding_dim))
        convs = []
        for i, n_gram in enumerate(ngram_filters):
            conv = Convolution1D(nb_filter=conv_hidden_units[i],
                                 filter_length=n_gram,
                                 border_mode='same',
                                 activation='relu')(graph_in)
            #pool = GlobalMaxPooling1D()(conv)
            convs.append(conv)

        if len(ngram_filters)>1:
            out = Merge(mode='concat')(convs)
        else:
            out = convs[0]
        conv_model = Model(input=graph_in, output=out)


        model = Sequential()
        model.add(conv_model)
        model.add(Dropout(0.5))
        model.add(LSTM(200, return_sequences=True))#, input_shape=(sequence_length, embedding_dim)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print model.summary()
        return model

    elif which_model == 2:
        model = Sequential()
        model.add(LSTM(200, return_sequences=True, input_shape=(sequence_length, embedding_dim)))
        #model.add(LSTM(400, return_sequences=True, input_shape=(sequence_length, embedding_dim)))
        #model.add(Dropout(0.5))
        #model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print model.summary()
        return model



    #model = Sequential()
    #model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length))
    #model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    
    #model.add(LSTM(400, return_sequences=True, input_shape=(sequence_length, embedding_dim)))
    #model.add(LSTM(64, input_shape=(sequence_length, embedding_dim)))
    #model.add(Dropout(0.2))
    #model.add(Activation('sigmoid'))

    #model.add(LSTM(200, return_sequences=True))
    #model.add(Dropout(0.4))
    #model.add(Activation('relu'))

    #model.add(LSTM(32))
    #model.add(Dropout(0.6))
    #model.add(Activation('relu'))

    #model.add(Dense(1, activation="sigmoid"))
    #model.add(Activation('sigmoid'))
    #opt = SGD(lr=0.01)

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model


def train_LSTM(X, Y, model, train_split=0.8, epochs=10, batch_size=32):

    # Works for TYPE2 but check for others
    samples = X.shape[0]
    train_samples = int(X.shape[0]*train_split)
    X_train, Y_train = X[:train_samples+1,:,:], Y[:train_samples+1,:,:]
    X_test, Y_test = X[train_samples+1:,:,:], Y[train_samples+1:,:,:]
    print "X_train:", X_train.shape
    print "X_test:", X_test.shape

    model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

    model.evaluate(X_test, Y_test, batch_size=batch_size)

    pred = model.predict(X_test)
    rounded = np.round(pred)

    result = helper.windiff_metric_NUMPY(Y_test, rounded)
    print result
    pdb.set_trace()

    #rounded = [round(x) for x in pred]
    

#    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
#    print cv_object
#    p, r, f1 = 0., 0., 0.
#    p1, r1, f11 = 0., 0., 0.
#    sentence_len = X.shape[1]
#    for train_index, test_index in cv_object.split(X):
#        shuffle_weights(model)
#        model.layers[0].set_weights([weights])
#        X_train, y_train = X[train_index], y[train_index]
#        X_test, y_test = X[test_index], y[test_index]
#        #pdb.set_trace()
#        y_train = y_train.reshape((len(y_train), 1))
#        X_temp = np.hstack((X_train, y_train))
#        for epoch in xrange(epochs):
#            for X_batch in batch_gen(X_temp, batch_size):
#                x = X_batch[:, :sentence_len]
#                y_temp = X_batch[:, sentence_len]
#		         try:
#                    y_temp = np_utils.to_categorical(y_temp, nb_classes=3)
#                except Exception as e:
#                    print e
#                    print y_temp
#                print x.shape, y.shape
#                loss, acc = model.train_on_batch(x, y_temp)#, class_weight=class_weights)
#                print loss, acc
#
#        y_pred = model.predict_on_batch(X_test)
#        y_pred = np.argmax(y_pred, axis=1)
#        print classification_report(y_test, y_pred)
#        print precision_recall_fscore_support(y_test, y_pred)
#        print y_pred
#        p += precision_score(y_test, y_pred, average='weighted')
#        p1 += precision_score(y_test, y_pred, average='micro')
#        r += recall_score(y_test, y_pred, average='weighted')
#        r1 += recall_score(y_test, y_pred, average='micro')
#        f1 += f1_score(y_test, y_pred, average='weighted')
#        f11 += f1_score(y_test, y_pred, average='micro')
#
#
#    print "macro results are"
#    print "average precision is %f" %(p/10)
#    print "average recall is %f" %(r/10)
#    print "average f1 is %f" %(f1/10)
#
#    print "micro results are"
#    print "average precision is %f" %(p1/10)
#    print "average recall is %f" %(r1/10)
#    print "average f1 is %f" %(f11/10)


if __name__ == "__main__":
    #Tweets = select_tweets()
    #tweets = Tweets
    SAMPLE_TYPE, X, Y = get_input(sample_type=2, shuffle_documents=True, pad=True)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X.shape[0], X.shape[1], X.shape[2]          #MAX_SEQUENCE_LENGTH is is already padded
    if SAMPLE_TYPE == 1:
        Y = Y[:,-1].reshape((NO_OF_SAMPLES, 1))    # For LSTM
    elif SAMPLE_TYPE == 2:
        # because of TimeDistributed layer :/
        Y = Y.reshape((NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, 1))
    else:
        print "INVALID SAMPLE TYPE!"

    ##gen_vocab()
    #filter_vocab(20000)
    #X, y = gen_sequence()
    #Y = y.reshape((len(y), 1))

    # X.shape = (no_of_samples, no_of_sentences_in_sample, dimension_of_word2vec)
    
    #data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    #y = np.array(y)
    #data, y = sklearn.utils.shuffle(data, y)
    #W = get_embedding_weights()
    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    #model = lstm_model(data.shape[1], 25, get_embedding_weights())
    #score = model.evaluate(X_test, y_test, batch_size=16)
    #pdb.set_trace()
    train_LSTM(X, Y, model, train_split=0.7, epochs=15, batch_size=4)
    #score = model.evaluate(X_test, y_test, batch_size=16)
    #pdb.set_trace()
