#from data_handler import get_data
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, RepeatVector, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional
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

    which_model = 2

    if which_model == 1:
        # Look_back = -inf, at looks back at the whole document
        # Simple LSTM with full document look back.
        # CHeck the convolution filters, they should be `probably` in a TimeDistributed wrapper
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
        #model.add(Bidirectional(LSTM(200, return_sequences=True)))#, input_shape=(sequence_length, embedding_dim)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif which_model == 2:
        # Look back is equal to -INF
        # This model creates a Stateful LSTM with lookback of the whole document
        # Input should be of the format (TOTAL_DOCUMENTS, TOTAL_SEQUENCES, SEQUENCE_DIM)
        # Also train using the custom trainer

        # STATEFUL MODEL
        print "Building the stateful-BLSTM"
        stateful_blstm, s_forward, s_backward = Sequential(), Sequential(), Sequential()
        s_forward.add(LSTM(300, batch_input_shape=(1, 1, embedding_dim), return_sequences=False, stateful=True))
        s_forward.add(Dropout(0.3))
        s_backward.add(LSTM(300, batch_input_shape=(1, 1, embedding_dim), return_sequences=False, stateful=True, go_backwards=True))
        s_backward.add(Dropout(0.3))
        stateful_blstm.add(Merge([s_forward, s_backward], mode='concat'))
        stateful_blstm.add(Dense(1, activation='sigmoid'))

        # NOT-STATEFUL MODEL
        print "Building the stateful-BLSTM"
        not_stateful_blstm, ns_forward, ns_backward = Sequential(), Sequential(), Sequential()
        ns_forward.add(LSTM(300, batch_input_shape=(1, 1, embedding_dim), return_sequences=False, stateful=False))
        ns_forward.add(Dropout(0.3))
        ns_backward.add(LSTM(300, batch_input_shape=(1, 1, embedding_dim), return_sequences=False, stateful=False, go_backwards=True))
        ns_backward.add(Dropout(0.3))
        not_stateful_blstm.add(Merge([ns_forward, ns_backward], mode='concat'))
        not_stateful_blstm.add(Dense(1, activation='sigmoid'))


        # Convolution layers
        #print "Building Convolution layers"
        #ngram_filters = [1, 2, 3, 5, 7, 9, 11]
        #conv_hidden_units = [300, 150, 150, 150, 150, 150, 150]
    
        #graph_in = Input(shape=(sequence_length, embedding_dim))
        #convs = []
        #for i, n_gram in enumerate(ngram_filters):
        #    conv = Convolution1D(nb_filter=conv_hidden_units[i],
        #                         filter_length=n_gram,
        #                         border_mode='same',
        #                         activation='relu')(graph_in)
        #    #pool = GlobalMaxPooling1D()(conv)
        #    convs.append(conv)

        #if len(ngram_filters)>1:
        #    out = Merge(mode='concat')(convs)
        #else:
        #    out = convs[0]
        #conv_model = Model(input=graph_in, output=out)

        print 'Build STATEFUL model...'
        model = Sequential()
        #forwardd(conv_model)
        #model.add(RepeatVector(2))
        #model.add(blstm)
        model.add(Merge([stateful_blstm, not_stateful_blstm], mode='concat'))
        #model.add(LSTM(600, batch_input_shape=(1, 1, embedding_dim), return_sequences=False, stateful=True))
        #model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


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


def custom_fit(X, Y, model, train_split=0.8, epochs=10):
        
    if train_split == 1:
        X_test, Y_test = X, Y
    else:
        # This is only for training! (If train_split =1 then only TEST)
        X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)
        print "Batch size = 1"
        print 'Train...'
        for epoch in range(epochs):
            mean_tr_acc = []
            mean_tr_loss = []
            for i in range(len(X_train)):
                #y_true = Y_train[i]
                for sequence, truth in zip(X_train[i], Y_train[i]): # Sequence in document
                    sequence = sequence.reshape((1, sequence.shape[0]))
                    sequence = np.expand_dims(sequence, axis=0)
                    tr_loss, tr_acc = model.train_on_batch([sequence, sequence, sequence, sequence], truth)

                    mean_tr_acc.append(tr_acc)
                    mean_tr_loss.append(tr_loss)
                model.reset_states()
        
            print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
            print('loss training = {}'.format(np.mean(mean_tr_loss)))
            print('___________________________________')
    
    mean_te_acc = []
    mean_te_loss = []
    predictions = []
    for i in range(len(X_test)):
        for sequence, truth in zip(X_test[i], Y_test[i]):
            sequence = sequence.reshape((1, sequence.shape[0]))
            sequence = np.expand_dims(sequence, axis=0)
            te_loss, te_acc = model.test_on_batch([sequence, sequence, sequence, sequence], truth)

            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
        model.reset_states()

        predictions.append([])
        for sequence, truth in zip(X_test[i], Y_test[i]):
            sequence = sequence.reshape((1, sequence.shape[0]))
            sequence = np.expand_dims(sequence, axis=0)
            y_pred = model.predict_on_batch([sequence, sequence, sequence, sequence])
            predictions[i].append(y_pred)
        model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    
    print "Check windiff value"
    #rounded = np.round(predictions)
    result = helper.windiff_metric_NUMPY(Y_test, predictions, win_size=-1, rounded=False)
    print result
    print('___________________________________')


def split_data(X, Y, train_split):
    samples = X.shape[0]
    train_samples = int(X.shape[0]*train_split)
    X_train, Y_train = X[:train_samples+1,], Y[:train_samples+1,]
    X_test, Y_test = X[train_samples+1:,], Y[train_samples+1:,]
    print "X_train:", X_train.shape, " | Y_train:", Y_train.shape
    print "X_test:", X_test.shape, " | Y_test:", Y_test.shape
    return X_train, Y_train, X_test, Y_test


def train_LSTM(X, Y, model, train_split=0.8, epochs=10, batch_size=32):

    # Clinical
    SAMPLE_TYPE_cli, X_cli, Y_cli = get_input(sample_type=4, shuffle_documents=False, pad=False)
    
    which_model = 2
    if which_model == 2:
        custom_fit(X, Y, train_split=train_split, model=model, epochs=epochs)
        print "Clinical Data"
        custom_fit(X_cli, Y_cli, train_split=1, model=model)  # Test clinical

    elif which_modle == 1:
        # Works for TYPE2 but check for others
        # Both these lines work for which_model == 1
        X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)
        model.fit(X_train, Y_train, shuffle=False, nb_epoch=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
    
        # WIkipedia
        #model.evaluate(X_test, Y_test, batch_size=batch_size)
        #pred = model.predict(X_test)
        #rounded = np.round(pred)
        #result = helper.windiff_metric_NUMPY(Y_test, rounded)
        #print result
    
        
        # Clinical
        # Temporary TRUNCATION
        TRUNCATE_LEN = X_train.shape[1]
        print "NOTE: Truncating the Test dataset(clinical) from %d sentences to %d sentences." %(X_cli.shape[1], TRUNCATE_LEN)
        X_cli, Y_cli = X_cli[:,:TRUNCATE_LEN,:], Y_cli[:,:TRUNCATE_LEN,:]
        model.evaluate(X_cli, Y_cli, batch_size=batch_size)
        pred = model.predict(X_cli)
        rounded = np.round(pred)
        result = helper.windiff_metric_NUMPY(Y_cli, rounded, win_size=10, rounded=True)
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
    # For which_model == 2
    SAMPLE_TYPE, X, Y = get_input(sample_type=2, shuffle_documents=True, pad=False)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X.shape[0], -1, X[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded

    # For which_model == 1
    #SAMPLE_TYPE, X, Y = get_input(sample_type=2, shuffle_documents=True, pad=True)
    #NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X.shape[0], X.shape[1], X.shape[2]          #MAX_SEQUENCE_LENGTH is is already padded
    #if SAMPLE_TYPE == 1:
    #    Y = Y[:,-1].reshape((NO_OF_SAMPLES, 1))    # For LSTM
    #elif SAMPLE_TYPE == 2:
    #    # because of TimeDistributed layer :/
    #    Y = Y.reshape((NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, 1))
    #else:
    #    print "INVALID SAMPLE TYPE!"

    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    train_LSTM(X, Y, model, train_split=0.7, epochs=10, batch_size=4)
