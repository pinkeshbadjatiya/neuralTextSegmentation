#from data_handler import get_data
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, RepeatVector, Input, Merge, merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional
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


SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = None, None, None, None
SAMPLE_TYPE_wiki, X_wiki, Y_wiki = None, None, None
SAMPLE_TYPE_bio, X_bio, Y_bio = None, None, None
GLOVE_EMBEDDING_DIM = 300

def lstm_model(sequence_length, embedding_dim, embedding_matrix, vocab_size):

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
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    elif which_model == 2:
        # Look back is equal to -INF
        # This model creates a Stateful LSTM with lookback of the whole document
        # Input should be of the format (TOTAL_DOCUMENTS, TOTAL_SEQUENCES, SEQUENCE_DIM)
        # Also train using the custom trainer

        # Convolution layers
        print "Building Convolution layers"
        ngram_filters = [1, 2, 3, 5]
        conv_hidden_units = [300, 150, 150, 150]
        #ngram_filters = [1, 2, 3, 5, 7, 9, 11]
        #conv_hidden_units = [300, 150, 150, 150, 150, 150, 150]
    
        print 'Build MAIN model...'
        main_input = Input(batch_shape=(1, embedding_dim), dtype='float32', name='main_input')
        embedded_input = Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, weights=[embedding_matrix], input_length=embedding_dim, init='uniform', trainable=True)(main_input)
        convs = []
        for i, n_gram in enumerate(ngram_filters):
            conv = Convolution1D(nb_filter=conv_hidden_units[i],
                                 filter_length=n_gram,
                                 border_mode='same',
                                 activation='relu')(embedded_input)
            #pool = GlobalMaxPooling1D()(conv)
            convs.append(conv)
        convoluted_input = Merge(mode='concat')(convs)
        CONV_DIM = sum(conv_hidden_units)

        # Dropouts for LSTMs can be merged
        ForLSTM_stateful = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=True)(convoluted_input)
        RevLSTM_stateful = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=True, go_backwards=True)(convoluted_input)
        BLSTM_stateful = merge([ForLSTM_stateful, RevLSTM_stateful], mode='concat')
        #BLSTM_stateful = merge([Dropout(0.3)(ForLSTM_stateful), Dropout(0.3)(RevLSTM_stateful)], mode='concat')
        BLSTM_prior = Dense(1)(BLSTM_stateful)

        ForLSTM = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=False)(convoluted_input)
        RevLSTM = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=False, go_backwards=True)(convoluted_input)
        #BLSTM = merge([Dropout(0.3)(ForLSTM), Dropout(0.3)(RevLSTM)], mode='concat')
        BLSTM = merge([ForLSTM, RevLSTM], mode='concat')
        BLSTM_memory = Dense(1)(BLSTM)

        #x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
        probablity = merge([BLSTM_memory, BLSTM_prior], mode='concat')
        output = Dense(1, activation='sigmoid')(probablity)
        model = Model(input=[main_input], output=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    print model.summary()
    return model


def custom_fit(X, Y, model, train_split=0.8, epochs=10):
        
    if train_split == 0:
        X_test, Y_test = X, Y
    else:
        # This is only for training! (If train_split =1 then only TEST)
        X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)

        print "Batch size = 1"
        print 'Train...'
        _total_docs = len(X_train)
        _total_sentences = sum([sequence.shape[0] for sequence in X_train])
        for epoch in range(epochs):
            mean_tr_acc = []
            mean_tr_loss = []
            _sentence_no = 0
            for i in range(len(X_train)):
                #y_true = Y_train[i]
                for sequence, truth in zip(X_train[i], Y_train[i]): # Sequence in document
                    sequence = sequence.reshape((1, sequence.shape[0]))
                    #sequence = np.expand_dims(sequence, axis=0)
                    tr_loss, tr_acc = model.train_on_batch([sequence], truth)

                    mean_tr_acc.append(tr_acc)
                    mean_tr_loss.append(tr_loss)
                    _sentence_no += 1
                    print ">> Epoch: %d/%d | Doc: %d/%d | Sent: %d/%d" %(epoch+1, epochs, i+1, _total_docs, _sentence_no+1, _total_sentences)
                model.reset_states()
        
            print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
            print('loss training = {}'.format(np.mean(mean_tr_loss)))
            print('___________________________________')
    
    # Testing
    mean_te_acc = []
    mean_te_loss = []
    predictions = []
    _total_docs = len(X_test)
    _total_sentences = sum([sequence.shape[0] for sequence in X_test])
    _sentence_no = 0
    for i in range(len(X_test)):
        for sequence, truth in zip(X_test[i], Y_test[i]):
            sequence = sequence.reshape((1, sequence.shape[0]))
            #sequence = np.expand_dims(sequence, axis=0)
            te_loss, te_acc = model.test_on_batch([sequence], truth)

            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
            _sentence_no += 1
            print ">> TEST >> Doc: %d/%d | Sent: %d/%d" %(i+1, _total_docs, _sentence_no+1, _total_sentences)
        model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    print("Predicting...")
    
    for i in range(len(X_test)):
        predictions.append([])
        for sequence, truth in zip(X_test[i], Y_test[i]):
            sequence = sequence.reshape((1, sequence.shape[0]))
            #sequence = np.expand_dims(sequence, axis=0)
            y_pred = model.predict_on_batch([sequence])
            predictions[i].append(y_pred)
        model.reset_states()

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


def train_LSTM(X, Y, model, embedding_W, train_split=0.8, epochs=10, batch_size=32):
    global X_wiki, Y_wiki, X_cli, Y_cli, X_bio, Y_bio

    which_model = 2
    if which_model == 2:
        custom_fit(X, Y, model=model, train_split=train_split, epochs=epochs)
        print "Clinical Data"
        custom_fit(X_cli, Y_cli, model=model, train_split=0)  # Test clinical

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
    

if __name__ == "__main__":
    # For which_model == 2
    #SAMPLE_TYPE_wiki, X_wiki, Y_wiki, trained_sample_handler = get_input(sample_type=2, shuffle_documents=True, pad=True)
    #NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_wiki.shape[0], -1, X_wiki[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # For which_model == 2
    # Biography data for training
    SAMPLE_TYPE_bio, X_bio, Y_bio, trained_sample_handler = get_input(sample_type=5, shuffle_documents=False, pad=True)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_bio.shape[0], -1, X_bio[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # Clinical - Only for testing
    SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = get_input(sample_type=4, shuffle_documents=False, pad=True, trained_sent2vec_model=trained_sample_handler)
    

    dictionary_object = trained_sample_handler.dictionary
    embedding_W = dictionary_object.get_embedding_weights()

    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embedding_W, len(dictionary_object.word2id_dic))
    #train_LSTM(X_wiki, Y_wiki, model, embedding_W, train_split=0.7, epochs=1, batch_size=1)
    train_LSTM(X_bio, Y_bio, model, embedding_W, train_split=0.7, epochs=1, batch_size=1)
