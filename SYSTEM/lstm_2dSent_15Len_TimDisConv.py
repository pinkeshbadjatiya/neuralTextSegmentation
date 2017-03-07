from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, RepeatVector, Input, Merge, merge, Convolution1D, Convolution2D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
from my_categorical import to_categorical_MULTI_DIM, w_binary_crossentropy

import numpy as np
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
import codecs
import gensim, sklearn
from string import punctuation
from collections import defaultdict
import sys
import math

import helper
from sample_handler import get_input


SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = None, None, None, None
SAMPLE_TYPE_wiki, X_wiki, Y_wiki = None, None, None
SAMPLE_TYPE_bio, X_bio, Y_bio = None, None, None
GLOVE_EMBEDDING_DIM = 300
SCALE_LOSS_FUN = True

SEQUENCES_LENGTH_FOR_TRAINING = 40

def lstm_model(sequences_length_for_training, embedding_dim, embedding_matrix, vocab_size):

    which_model = 2
    # Look back is equal to -INF
    # This model creates a Stateful LSTM with lookback of the whole document
    # Input should be of the format (TOTAL_DOCUMENTS, TOTAL_SEQUENCES, SEQUENCE_DIM)
    # Also train using the custom trainer

    # Convolution layers
    print "Building Convolution layers"
    ngram_filters = [1, 2, 3, 4, 5]
    conv_hidden_units = [300, 300, 300, 300, 300]
    
    print 'Build MAIN model...'
    #pdb.set_trace()
    main_input = Input(shape=(SEQUENCES_LENGTH_FOR_TRAINING, embedding_dim), dtype='float32', name='main_input')
    embedded_input = TimeDistributed(Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, weights=[embedding_matrix], input_length=embedding_dim, init='uniform'))(main_input)
    convs = []
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv = TimeDistributed(Convolution1D(nb_filter=hidden_units,
                             filter_length=n_gram,
                             border_mode='same',
                             activation='relu'))(embedded_input)
        flattened = TimeDistributed(Flatten())(conv)
        #pool = GlobalMaxPooling1D()(conv)
        convs.append(flattened)
    convoluted_input = Merge(mode='concat')(convs)
    CONV_DIM = sum(conv_hidden_units)

    # Dropouts for LSTMs can be merged
    #ForLSTM_stateful = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=True)(convoluted_input)
    #RevLSTM_stateful = LSTM(300, batch_input_shape=(1, embedding_dim, CONV_DIM), return_sequences=False, stateful=True, go_backwards=True)(convoluted_input)
    #BLSTM_stateful = merge([ForLSTM_stateful, RevLSTM_stateful], mode='concat')
    #BLSTM_stateful = merge([Dropout(0.3)(ForLSTM_stateful), Dropout(0.3)(RevLSTM_stateful)], mode='concat')
    #BLSTM_prior = Dense(1)(BLSTM_stateful)

    #encoded = Bidirectional(LSTM(512, input_shape=(SEQUENCES_LENGTH_FOR_TRAINING, CONV_DIM), return_sequences=True, stateful=False), merge_mode='concat')(convoluted_input)
    #decoded = Bidirectional(LSTM(512, input_shape=(SEQUENCES_LENGTH_FOR_TRAINING, CONV_DIM), return_sequences=True, stateful=False), merge_mode='concat')(encoded)
    encoded = Bidirectional(LSTM(512, input_shape=(SEQUENCES_LENGTH_FOR_TRAINING, CONV_DIM), return_sequences=True, stateful=False), merge_mode='concat')(convoluted_input)
    encoded_drop = Dropout(0.3)(encoded)
    more_encoded = Bidirectional(LSTM(512), merge_mode='concat')(encoded_drop)
    more_encoded_drop = Dropout(0.3)(more_encoded)

    encoded_input = RepeatVector(SEQUENCES_LENGTH_FOR_TRAINING)(more_encoded)
    decoded = Bidirectional(LSTM(512, input_shape=(SEQUENCES_LENGTH_FOR_TRAINING, CONV_DIM), return_sequences=True, stateful=False), merge_mode='concat')(encoded_input)
    decoded_drop = Dropout(0.3)(decoded)
    
    dense_out = Dense(300)(decoded_drop)
    dense_out_drop = Dropout(0.3)(dense_out)

    output = TimeDistributed(Dense(1, activation='sigmoid'))(dense_out_drop)
    model = Model(input=[main_input], output=output)
    model.layers[1].trainable = False
    model.compile(loss=w_binary_crossentropy, optimizer='rmsprop', metrics=['accuracy', 'recall'])


    print model.summary()
    return model

def batch_gen_consecutive_segments_from_big_seq(X_with_doc, Y_with_doc, batch_size, one_side_context_size):
    # X should be (no_of_documents, total_sequences, bla, bla, bla ...)
    # Same should be Y
    X_Left_batch, Y_Left_batch, X_Right_batch, Y_Right_batch, X_Mid_batch, Y_Mid_batch = [], [], [], [], [], []
    for X, Y in zip(X_with_doc, Y_with_doc):   # iterate over document
        total_seq = X.shape[0]
        for i in range(total_seq):   # sample count
            xL_i, xL_j, sent_i, xR_i, xR_j = i-one_side_context_size, i, i, i+1, i+one_side_context_size+1
            if total_seq < 2*one_side_context_size + 1:
                print "Too Small sequence"
                continue

            # Check if padding for the one_side_context_size required for both LEFT & RIGHT
            X_temp_Left, X_temp_Right, Y_temp_Left, Y_temp_Right = None, None, None, None
            if xL_i < 0:
                X_temp_Left, Y_temp_Left = np.array(X[0: xL_j]), np.array(Y[0: xL_j]) # Create resizable array not view
                shpX, shpY = list(X_temp_Left.shape), list(Y_temp_Left.shape)
                shpX[0], shpY[0] = one_side_context_size, one_side_context_size
                X_temp_Left, Y_temp_Left = X_temp_Left.resize(tuple(shpX)), Y_temp_Left.resize(tuple(shpY))
            else:
                X_temp_Left, Y_temp_Left = X[xL_i: xL_j], Y[xL_i: xL_j]

            if xR_j > total_seq:
                X_temp_Right, Y_temp_Right = np.array(X[xR_i: total_seq]), np.array(Y[xR_i: total_seq]) # Create resizable array not view
                shpX, shpY = list(X_temp_Right.shape), list(Y_temp_Right.shape)
                shpX[0], shpY[0] = one_side_context_size, one_side_context_size
                X_temp_Right, Y_tempRight = X_temp_Right.resize(tuple(shpX)), Y_temp_Right.resize(tuple(shpY))
            else:
                X_temp_Right, Y_temp_Right = X[xR_i: xR_j], Y[xR_i: xR_j]

            X_Left_batch.append(X_left_temp), X_Right_batch.append(X_Right_temp)
            #Y_Left_batch.append(Y_Left_temp), Y_Right_batch.append(Y_Right_temp)       # No need for context groundtruths. May be we can test with that also later?
            X_Mid_batch.append(X[sent_i]), Y_Mid_batch.append(Y[sent_i])
            if len(X_batch) == batch_size:
                yield np.array(X_Left_batch), np.array(X_Mid_batch), np.asarray(X_Right_batch), np.array(Y_Mid_batch)
                X_Left_batch, Y_Left_batch, X_Right_batch, Y_Right_batch, X_Mid_batch, Y_Mid_batch = [], [], [], [], [], []
    if len(X_batch):
        yield np.array(X_Left_batch), np.array(X_Mid_batch), np.asarray(X_Right_batch), np.array(Y_Mid_batch)


def batch_gen_SHORT_SEQ_for_training_from_big_seq(X_with_doc, Y_with_doc, batch_size, no_of_sent_in_batch):
    # X should be (no_of_documents, total_sequences, bla, bla, bla ...)
    # Same should be Y
    X_batch, Y_batch, actual_sentences_batch = [], [], []
    for X, Y in zip(X_with_doc, Y_with_doc):   # iterate over document
        total_seq = X.shape[0]
        for i in range(int(math.ceil(total_seq*1.0/no_of_sent_in_batch))):   # sample count
            x_i, x_j = i*no_of_sent_in_batch, (i+1)*no_of_sent_in_batch
            X_temp, Y_temp = X[x_i: x_j], Y[x_i: x_j]
            count_sent = X_temp.shape[0]

            # Check if padding for the no of sentences required
            if X_temp.shape[0] != no_of_sent_in_batch:
                X_temp, Y_temp = np.array(X_temp), np.array(Y_temp) # Create resizable array not view
                shpX, shpY = list(X_temp.shape), list(Y_temp.shape)
                shpX[0], shpY[0] = no_of_sent_in_batch, no_of_sent_in_batch
                X_temp.resize(tuple(shpX)), Y_temp.resize(tuple(shpY))    # Resize to fit the batch size (ZEROES)

            X_batch.append(X_temp), Y_batch.append(Y_temp), actual_sentences_batch.append(count_sent)
            if len(X_batch) == batch_size:
                yield np.array(X_batch), np.asarray(Y_batch), actual_sentences_batch
                X_batch, Y_batch, actual_sentences_batch = [], [], []
    if len(X_batch):
        yield np.asarray(X_batch), np.asarray(Y_batch), actual_sentences_batch



def custom_fit(X, Y, model, batch_size, train_split=0.8, epochs=10):
        
    if train_split == 0:
        X_test, Y_test = X, Y
    else:
        # This is only for training! (If train_split =1 then only TEST)
        X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)

        class_weight = None
        if SCALE_LOSS_FUN:
            classes, counts = np.unique(Y_train, return_counts=True)
            class_weight = dict(zip(classes, counts/float(sum(counts))))
            print class_weight

        print "Batch size = 1"
        print 'Train...'
        for epoch in range(epochs):
            mean_tr_acc, mean_tr_loss, mean_tr_rec = [], [], []
            for batch_X, batch_Y, _ct in batch_gen_SHORT_SEQ_for_training_from_big_seq(X_train, Y_train, batch_size, SEQUENCES_LENGTH_FOR_TRAINING):
                #pdb.set_trace()
                #batch_Y_vec = to_categorical_MULTI_DIM(batch_Y, nb_classes=2)
                #print batch_Y.shape, batch_Y_vec.shape
                #tr_loss, tr_acc, tr_rec = model.train_on_batch(batch_X, batch_Y_vec, class_weight=class_weight)
                tr_loss, tr_acc, tr_rec = model.train_on_batch(batch_X, batch_Y)

                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
                mean_tr_rec.append(tr_rec)
                #print ">> Epoch: %d/%d" %(epoch+1, epochs)
            #model.reset_states()
        
            print ">> Epoch: %d/%d" %(epoch+1, epochs)
            print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
            print('recall training = {}'.format(np.mean(mean_tr_rec)))
            print('loss training = {}'.format(np.mean(mean_tr_loss)))
            print('___________________________________')
    
    # Testing
    mean_te_acc, mean_te_loss, mean_te_rec = [], [], []
    print ">> (TEST) >> Testing, X:", X_test.shape, "Y:", Y_test.shape
    for batch_X, batch_Y, _ct in batch_gen_SHORT_SEQ_for_training_from_big_seq(X_test, Y_test, batch_size, SEQUENCES_LENGTH_FOR_TRAINING):
        #batch_Y_vec = to_categorical_MULTI_DIM(batch_Y, nb_classes=2)
        #te_loss, te_acc, te_rec = model.test_on_batch(batch_X, batch_Y_vec, class_weight=class_weight)
        te_loss, te_acc, te_rec = model.test_on_batch(batch_X, batch_Y)

        mean_te_acc.append(te_acc)
        mean_te_loss.append(te_loss)
        mean_te_rec.append(te_rec)
    #model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('recall testing = {}'.format(np.mean(mean_te_rec)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    
    # Predicting
    print("Predicting...")
    predictions = []
    for batch_X, batch_Y, _sent_count in batch_gen_SHORT_SEQ_for_training_from_big_seq(X_test, Y_test, batch_size, SEQUENCES_LENGTH_FOR_TRAINING):
        #batch_y_pred_vec = model.predict_on_batch(batch_X)
        #batch_y_pred = np.argmax(batch_y_pred_vec, axis=2)
        batch_y_pred = model.predict_on_batch(batch_X)

        for y_pred, sent_ct in zip(batch_y_pred, _sent_count):
            shp = list(y_pred.shape)
            shp[0] = sent_ct
            predictions.append(np.resize(y_pred, tuple(shp)))
    #model.reset_states()
    #pdb.set_trace()

    print "Check windiff value"
    #rounded = np.round(predictions)
    predictions = np.concatenate(predictions, axis=0)
    result = helper.windiff_metric_ONE_SEQUENCE(np.concatenate(Y_test, axis=0), predictions, win_size=-1, rounded=False)
    print result
    print('___________________________________')
    if not train_split:
        pdb.set_trace()



#def convert_to_a_BIG_sequence(output):
#    seq = []
#    for batchs in output:
#        for batch in batchs:
#            for sentence in batch:
#                seq.append(sentence)
#    return np.asarray(seq)
    


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
        custom_fit(X, Y, model=model, batch_size=batch_size, train_split=train_split, epochs=epochs)
        print "############## Clinical Data ###########"
        custom_fit(X_cli, Y_cli, model=model, batch_size=batch_size, train_split=0, epochs=-1)  # Test clinical
        print "############## Biography Data ###########"
        custom_fit(X_bio, Y_bio, model=model, batch_size=batch_size, train_split=0, epochs=-1)  # Test biography

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
    SAMPLE_TYPE_wiki, X_wiki, Y_wiki, trained_sample_handler = get_input(sample_type=2, shuffle_documents=True, pad=True)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_wiki.shape[0], -1, X_wiki[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # For which_model == 2
    # Biography data for training
    SAMPLE_TYPE_bio, X_bio, Y_bio, trained_sample_handler = get_input(sample_type=5, shuffle_documents=False, pad=True, trained_sent2vec_model=trained_sample_handler)
    #NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_bio.shape[0], -1, X_bio[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # Clinical - Only for testing
    SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = get_input(sample_type=4, shuffle_documents=False, pad=True, trained_sent2vec_model=trained_sample_handler)
    

    dictionary_object = trained_sample_handler.dictionary
    embedding_W = dictionary_object.get_embedding_weights()

    model = lstm_model(SEQUENCES_LENGTH_FOR_TRAINING, EMBEDDING_DIM, embedding_W, len(dictionary_object.word2id_dic))
    train_LSTM(X_wiki, Y_wiki, model, embedding_W, train_split=0.7, epochs=10, batch_size=64)
    #train_LSTM(X_bio, Y_bio, model, embedding_W, train_split=0.7, epochs=1, batch_size=32)
