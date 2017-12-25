#from gpu_config import *

from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, RepeatVector, Input, Merge, merge, Convolution1D, Convolution2D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
from my_categorical import to_categorical_MULTI_DIM, w_binary_crossentropy
from keras.utils.np_utils import to_categorical


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
from tabulate import tabulate

#from attention_lstm import AttentionLSTM
#from attention_lstm_without_weights import Attention
from final_attention_layer import Attention
from AttentionWithContext import AttentionWithContext
import progbar, time

trained_sample_handler = None
SAMPLE_TYPE_wiki, X_wiki, Y_wiki = None, None, None                 # Training + Development (need to split)
SAMPLE_TYPE_wikitest, X_wikitest, Y_wikitest = None, None, None     # Testing
SAMPLE_TYPE_cli, X_cli, Y_cli = None, None, None              # Testing
SAMPLE_TYPE_bio, X_bio, Y_bio = None, None, None              # Testing
SAMPLE_TYPE_fic, X_fic, Y_fic = None, None, None              # Testing

GLOVE_EMBEDDING_DIM = 300
SCALE_LOSS_FUN = True

# SEQUENCES_LENGTH_FOR_TRAINING = 40

ONE_SIDE_CONTEXT_SIZE = 15

LOAD_SAVED_MODEL_AND_CONTINUE_TRAIN = False
SAVE_MODEL_AFTER_EACH_EPOCH = False
TRAINABLE_EMBEDDINGS = False


np.random.seed(2345)   # IMP seed
saved_model_epoch_done = None

def load_saved_model():
    global saved_model_epoch_done
    print "====== Loading the saved model ======="
    filename = sys.argv[1]
    epoch_done = filename.split(".")[0].split("_")[-1]
    print ">>> Continuing from epoch %d <<<<" %(int(epoch_done)+1)
    saved_model_epoch_done = int(epoch_done)

    model = load_model(filename)
    return model
        


def baseline_perceptron(sequence_len):

    print 'Build MAIN model...'
    #pdb.set_trace()
    main_input = Input(shape=(sequence_len,), dtype='float32', name='main-input')

    encode_1 = Dense(300, name='dense-1')(main_input)
    drop_out_1 = Dropout(0.3)(encode_1)

    encode_2 = Dense(300, name='dense-2')(drop_out_1)
    drop_out_2 = Dropout(0.3)(encode_2)

    encode_3 = Dense(300, name='dense-2')(drop_out_2)
    drop_out_3 = Dropout(0.3)(encode_3)

    output = Dense(2, activation='sigmoid')(drop_out_3)

    model = Model(input=[main_input], output=output)
    #model.compile(loss=w_binary_crossentropy, optimizer='rmsprop', metrics=['accuracy', 'recall'])
    model.compile(loss=w_binary_crossentropy, optimizer='adadelta', metrics=['accuracy', 'recall'])

    print model.summary(line_length=150, positions=[.46, .65, .77, 1.])
    return model


def baseline_CNN(sequences_length_for_training, embedding_dim, embedding_matrix, vocab_size):

    which_model = 2

    print 'Build MAIN model...'
    ngram_filters = [2, 3, 4, 5]
    conv_hidden_units = [200, 200, 200, 200]
    
    main_input = Input(shape=(embedding_dim,), dtype='float32', name='main-input')

    main_input_embedder = Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, weights=[embedding_matrix], input_length=embedding_dim, init='uniform')
    embedded_input_main = main_input_embedder(main_input)

    convsM = []
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv_layer = Convolution1D(nb_filter=hidden_units,
                             filter_length=n_gram,
                             border_mode='same',
                             #border_mode='valid',
                             activation='tanh', name='Convolution-'+str(n_gram)+"gram")
        mid = conv_layer(embedded_input_main)

        # Use Flatten() instead of MaxPooling()
        #flat_M = TimeDistributed(Flatten(), name='TD-flatten-mid-'+str(n_gram)+"gram")(mid)
        #convsM.append(flat_M)

        # Use GlobalMaxPooling1D() instead of Flatten()
        pool_M = GlobalMaxPooling1D()(mid)
        convsM.append(pool_M)

    convoluted_mid = Merge(mode='concat')(convsM)
    CONV_DIM = sum(conv_hidden_units)

    ####convoluted_mid, convoluted_left, convoluted_right, CONV_DIM = main_input, left_context, right_context, 300
    #flat_mid = Flatten()(convoluted_mid)
    encode_mid = Dense(300, name='dense-intermediate-mid-encoder')(convoluted_mid)

    #context_encoder_intermediate1 = LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), consume_less='gpu', dropout_W=0.3, dropout_U=0.3, return_sequences=True, stateful=False)
    #context_encoder = LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), consume_less='gpu', dropout_W=0.3, dropout_U=0.3, return_sequences=True, stateful=False)
    #context_encoder_intermediate1 = Bidirectional(LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), consume_less='gpu', dropout_W=0.3, dropout_U=0.3, return_sequences=True, stateful=False), name='BiLSTM-context-encoder-intermediate1', merge_mode='concat')
    #context_encoder = Bidirectional(LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), consume_less='gpu', dropout_W=0.3, dropout_U=0.3, return_sequences=True, stateful=False), name='BiLSTM-context-encoder', merge_mode='concat')
    ####encode_left = context_encoder(context_encoder_intermediate1(convoluted_left))
    
    encode_mid_drop = Dropout(0.2)(encode_mid)

    decoded = Dense(300, name='decoded')(encode_mid_drop)
    decoded_drop = Dropout(0.3, name='decoded_drop')(decoded)
    
    output = Dense(2, activation='sigmoid')(decoded_drop)
    model = Model(input=[main_input], output=output)
    model.layers[1].trainable = TRAINABLE_EMBEDDINGS
    model.compile(loss=w_binary_crossentropy, optimizer='rmsprop', metrics=['accuracy', 'recall'])
    #model.compile(loss=w_binary_crossentropy, optimizer='adadelta', metrics=['accuracy', 'recall'])


    print model.summary(line_length=150, positions=[.46, .65, .77, 1.])
    return model




def batch_gen_consecutive_context_segments_from_big_seq(typeOfBatch, X_with_doc, Y_with_doc, batch_size, one_side_context_size):
    # X should be (no_of_documents, total_sequences, bla, bla, bla ...)
    # Same should be Y
    X_Left_batch, Y_Left_batch, X_Right_batch, Y_Right_batch, X_Mid_batch, Y_Mid_batch = [], [], [], [], [], []
    for X, Y in zip(X_with_doc, Y_with_doc):   # iterate over document
        total_seq = X.shape[0]
        for i in range(total_seq):   # sample count
            # The neighbouring context do not contain the main sentence in their window
            #xL_i, xL_j, sent_i, xR_i, xR_j = i-one_side_context_size, i, i, i+1, i+one_side_context_size+1  # Example:: left: [0: 5] | Mid: [5: 6] | Right: [6: 11]
            #context_mat_size = one_side_context_size

            # The neighbouring context also contains the main sentence in the temporal data to derive some inference uing BLSTM
            xL_i, xL_j, sent_i, xR_i, xR_j = i-one_side_context_size, i+1, i, i, i+one_side_context_size+1  # Example:: left: [0: 6] | Mid: [5: 6] | Right: [5: 11]
            context_mat_size = one_side_context_size + 1

            if typeOfBatch == "train":    # Skip documents only when they are for training, for testing test on all the documents
                if total_seq < 2*one_side_context_size + 1:
                    print "Too Small sequence: Found %d, required %d. DROPPING DOCUMENT" %(total_seq, 2*one_side_context_size+1)
                    break

            # Check if padding for the one_side_context_size required for both LEFT & RIGHT
            X_temp_Left, X_temp_Right, Y_temp_Left, Y_temp_Right = None, None, None, None
            if xL_i < 0:
                X_temp_Left, Y_temp_Left = np.array(X[0: xL_j]), np.array(Y[0: xL_j]) # Create resizable array not view
                #print "_________________"
                #print X_temp_Left.shape
                #print X_temp_Left
                shpX, shpY = list(X_temp_Left.shape), list(Y_temp_Left.shape)
                shpX[0] = shpY[0] = context_mat_size - xL_j
                X_temp_Left, Y_temp_Left = np.concatenate((np.zeros(shpX), X_temp_Left), axis=0), np.concatenate((np.zeros(shpY), Y_temp_Left), axis=0)
                #print X_temp_Left.shape
                #print X_temp_Left
                #print "-----------------"
            else:
                X_temp_Left, Y_temp_Left = X[xL_i: xL_j], Y[xL_i: xL_j]

            if xR_j > total_seq:
                X_temp_Right, Y_temp_Right = np.array(X[xR_i: total_seq]), np.array(Y[xR_i: total_seq]) # Create resizable array not view
                #print "+++++++++++++++++++++++++++"
                #print X_temp_Right.shape
                #print X_temp_Right
                shpX, shpY = list(X_temp_Right.shape), list(Y_temp_Right.shape)
                shpX[0] = shpY[0] = context_mat_size - (total_seq - xR_i)
                X_temp_Right, Y_temp_Right = np.concatenate((X_temp_Right, np.zeros(shpX)), axis=0), np.concatenate((Y_temp_Right, np.zeros(shpY)), axis=0)
                #print X_temp_Right
                #print X_temp_Right.shape
                #print "==========================="
            else:
                X_temp_Right, Y_temp_Right = X[xR_i: xR_j], Y[xR_i: xR_j]

            X_Left_batch.append(X_temp_Left), X_Right_batch.append(X_temp_Right)
            #Y_Left_batch.append(Y_Left_temp), Y_Right_batch.append(Y_Right_temp)       # No need for context groundtruths. May be we can test with that also later?
            X_Mid_batch.append(X[sent_i].reshape((1, -1))), Y_Mid_batch.append(Y[sent_i])  # X's should be TimeDistributed but not Y's
            if len(X_Left_batch) == batch_size:
                yield np.array(X_Left_batch), np.array(X_Mid_batch), np.asarray(X_Right_batch), np.array(Y_Mid_batch)
                X_Left_batch, Y_Left_batch, X_Right_batch, Y_Right_batch, X_Mid_batch, Y_Mid_batch = [], [], [], [], [], []
    if len(X_Left_batch):
        yield np.array(X_Left_batch), np.array(X_Mid_batch), np.asarray(X_Right_batch), np.array(Y_Mid_batch)


def batch_gen_sentences_without_context(X, Y, batch_size, fixed_size=False):
    # X should be (no_of_documents, total_sequences, bla, bla, bla ...)
    # Same should be Y
    # Specially for baselines_models
    total_seq = X.shape[0]
    for i in range(int(math.ceil(total_seq*1.0/batch_size))):   # sample count
        x_i, x_j = i*batch_size, (i+1)*batch_size
        X_temp, Y_temp = X[x_i: x_j], Y[x_i: x_j]
        count_sent = X_temp.shape[0]

        # Check if padding for the no of sentences required
        if fixed_size:
            if X_temp.shape[0] != batch_size:
                X_temp, Y_temp = np.array(X_temp), np.array(Y_temp) # Create resizable array not view
            shpX, shpY = list(X_temp.shape), list(Y_temp.shape)
            shpX[0], shpY[0] = batch_size, batch_size
            X_temp.resize(tuple(shpX)), Y_temp.resize(tuple(shpY))    # Resize to fit the batch size (ZEROES)

        yield np.array(X_temp), np.asarray(Y_temp)



def custom_fit(X_train, Y_train, X_test, Y_test, model, batch_size, epochs=10):

    # Print Train stats
    total_sentences, total_documents = 0, 0
    total_documents = X_train.shape[0]
    total_sentences = sum([doc.shape[0] for doc in X_train])
    print "X-wiki TRAIN stats: Total %d sentences in %d documents" %(total_sentences, total_documents)

    class_weight = None
    if SCALE_LOSS_FUN:
        # Iterate as the no of sentences in each document is different
        # so np.unique() messes up.
        classes, counts = None, []
        for _temp_Yi in Y_train:
            classes, _temp_counts = np.unique(_temp_Yi, return_counts=True)
            counts.append(_temp_counts)
        counts = np.sum(counts, axis=0)
        class_weight = dict(zip(classes.tolist(), counts/float(sum(counts))))
        print class_weight

    train_avg_seg_len = np.mean([helper.compute_avg_seg_len(Yi) for Yi in Y_train], axis=0)
    print ">> Train AVG_SEGMENT_LENGTH:", train_avg_seg_len

    print 'Train...'
    start_epoch = 0
    if LOAD_SAVED_MODEL_AND_CONTINUE_TRAIN:   # If we have saved model, then continue from the last epoch where we stopped
        start_epoch = saved_model_epoch_done  # The epoch count is zero indexed in TRAIN, while the count in saved file is 1 indexed

    print_iter_count = 0
    for epoch in range(start_epoch, epochs):
        mean_tr_acc, mean_tr_loss, mean_tr_rec = [], [], []
        batch_count = 0
        rLoss, rRecall, rAcc = 0,0,0 # Running parameters for printing while training
        for i in range(total_documents):
            X, Y = X_train[i], Y_train[i]
            for (batch_X, batch_Y) in batch_gen_sentences_without_context(X, Y, batch_size, fixed_size=False):
                #pdb.set_trace()

                batch_Y = to_categorical(batch_Y, nb_classes=2) # Convert to output as 2 classes

                start = time.time()
                tr_loss, tr_acc, tr_rec = model.train_on_batch([batch_X], batch_Y)
                speed = time.time() - start

                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
                mean_tr_rec.append(tr_rec)
                #rLoss, rRecall, rAcc = (rLoss*batch_count + tr_loss)/(batch_count + 1), (rRecall*batch_count + tr_rec)/(batch_count + 1), (rAcc*batch_count + tr_acc)/(batch_count + 1)
                #progbar.prog_bar(True, total_sentences, epochs, batch_size, epoch, batch_count, speed=speed, data={ 'rLoss': rLoss, 'rAcc': rAcc, 'rRec': rRecall })
                progbar.prog_bar(True, total_sentences, epochs, batch_size, epoch, batch_count, speed=speed, data={ 'Loss': tr_loss, 'Acc': tr_acc, 'Rec': tr_rec })
                batch_count += 1

        progbar.end()
        if SAVE_MODEL_AFTER_EACH_EPOCH:
            model.save("model_trainable_%s_epoc_%d.h5" %(str(TRAINABLE_EMBEDDINGS), epoch+1))

        print ">> Epoch: %d/%d" %(epoch+1, epochs)
        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('recall training = {}'.format(np.mean(mean_tr_rec)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))

        testing_on_data("Wikipedia(DEVELOPMENT)", X_test, Y_test, model, batch_size, summary_only=True)
        testing_on_data("Clinical", X_cli, Y_cli, model, batch_size, summary_only=True)
        #testing_on_data("Biography", X_bio, Y_bio, model, batch_size)
        testing_on_data("Fiction", X_fic, Y_fic, model, batch_size, summary_only=True)
        testing_on_data("Wikipedia(BENCHMARK)", X_wikitest, Y_wikitest, model, batch_size, summary_only=True)
    
        print('___________________________________')
    
    # Testing
    print "####################################################################"
    print ">> (TEST) >> Testing, X:", X_test.shape, "Y:", Y_test.shape
    mean_te_acc, mean_te_loss, mean_te_rec = [], [], []
    for i in range(X_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        for batch_X, batch_Y in batch_gen_sentences_without_context(X, Y, batch_size, fixed_size=False):
            te_loss, te_acc, te_rec = model.test_on_batch([batch_X], batch_Y)
            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
            mean_te_rec.append(te_rec)

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('recall testing = {}'.format(np.mean(mean_te_rec)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    

def save_predictions(type_of_data, X_test, Y_test, model, batch_size, summary_only=False):
    # Predicting
    print "====================== %s ======================" %(type_of_data)
    print "GET PREDICTIONS... (SEPARATELY FOR EACH DOCUMENT)"
    data = {
        'wd_r': [],
        'wd_e': [],
        'pk': []
    }
    doc_idx = 18
    avg_segment_lengths_across_test_data = [] # Average segment length across the documents
    predictions_return = []
    zipped = zip(X_test, Y_test)
    for i, (Xi_test, Yi_test) in enumerate(zipped):

        if i != doc_idx:
            continue

        print Xi_test.shape
        pred_per_doc = []
        Xi_test, Yi_test = Xi_test.reshape((1,) + Xi_test.shape), Yi_test.reshape((1,) + Yi_test.shape)   # Convert to format of 1 document
        for batch_X_left, batch_X_mid, batch_X_right, batch_Y_mid in batch_gen_consecutive_context_segments_from_big_seq("test", Xi_test, Yi_test, batch_size, ONE_SIDE_CONTEXT_SIZE):
            batch_y_pred = model.predict_on_batch([batch_X_left, batch_X_mid, batch_X_right])
            pred_per_doc.append(batch_y_pred)

        if not len(pred_per_doc): # batch generator might drop a few documents
            continue

        #rounded = np.round(pred_per_doc)
        pred_per_doc = np.concatenate(pred_per_doc, axis=0)
        #return pred_per_doc
        predictions_return.append(pred_per_doc)
        actual_avg_seg_length, result = helper.windiff_and_pk_metric_ONE_SEQUENCE(Yi_test[0], pred_per_doc, window_size=-1, rounded=False, print_individual_stats=not summary_only)
        avg_segment_lengths_across_test_data.append(actual_avg_seg_length)
        data['pk'].append(result['pk'])
        data['wd_r'].append(result['wd_r'])
        data['wd_e'].append(result['wd_e'])

        print "WD: %f, PK: %f" %(result['wd_r'], result['pk'])
        # Save for visualization
        #rounded_per_doc = np.round(pred_per_doc)
        rounded_per_doc = pred_per_doc
        output = ["ref,hyp"]
        for (ref, hyp) in zip(Y_test[doc_idx], rounded_per_doc):
            output.append(str(int(ref[0])) + "," + str(hyp[0]))
        file_name = "prediction_output_save.csv"
        with open(file_name, "a") as f:
            for line in output:
                f.write(line + "\r\n")
        print "Written document index: `%d` to file: `%s`" %(doc_idx, file_name)
        return



def testing_on_data(type_of_data, X_test, Y_test, model, batch_size, summary_only=False, visualize=False):
    # Predicting
    print "====================== %s ======================" %(type_of_data)
    print "Predicting... (SEPARATELY FOR EACH DOCUMENT)"
    data = {
        'wd_r': [],
        'wd_e': [],
        'pk': []
    }
    avg_segment_lengths_across_test_data = [] # Average segment length across the documents
    for Xi_test, Yi_test in zip(X_test, Y_test):
        pred_per_doc = []
        for batch_count, (batch_X, batch_Y) in enumerate(batch_gen_sentences_without_context(Xi_test, Yi_test, batch_size, fixed_size=False)):
            start_time = time.time()
            batch_y_pred = model.predict_on_batch([batch_X])
            #print time.time() - start_time, "sec/batch of size", batch_size
            pred_per_doc.append(batch_y_pred[:,1])

        if not len(pred_per_doc): # batch generator might drop a few documents
            continue

        if len(pred_per_doc) > Xi_test.shape[0]:
            pred_per_doc = pred_per_doc[:Xi_test.shape[0]]

        #rounded = np.round(pred_per_doc)
        pred_per_doc = np.concatenate(pred_per_doc, axis=0)
        if visualize:
            print ">>>>> VISUALIZE <<<<<<"
            pdb.set_trace()
        #pdb.set_trace()

        Yi_test_for_windiff = to_categorical(Yi_test)[:,1]

        actual_avg_seg_length, result = helper.windiff_and_pk_metric_ONE_SEQUENCE(Yi_test_for_windiff, pred_per_doc, window_size=-1, rounded=False, print_individual_stats=not summary_only)
        avg_segment_lengths_across_test_data.append(actual_avg_seg_length)
        data['pk'].append(result['pk'])
        data['wd_r'].append(result['wd_r'])
        data['wd_e'].append(result['wd_e'])

    print ">> Summary (%s):" %(type_of_data)
    print "AVG segment length in test data: %f, std: %f" % (np.mean(avg_segment_lengths_across_test_data), np.std(avg_segment_lengths_across_test_data))
    print "WinDiff metric (Windiff_r):: avg: %f | std: %f | min: %f | max: %f" %(np.mean(data['wd_r']), np.std(data['wd_r']), np.min(data['wd_r']), np.max(data['wd_r']))
    print "WinDiff metric (Windiff_e):: avg: %f | std: %f | min: %f | max: %f" %(np.mean(data['wd_e']), np.std(data['wd_e']), np.min(data['wd_e']), np.max(data['wd_e']))
    print "Pk metric:: avg: %f | std: %f | min: %f | max: %f" %(np.mean(data['pk']), np.std(data['pk']), np.min(data['pk']), np.max(data['pk']))
    print('___________________________________')




def split_data(X, Y, train_split):
    samples = X.shape[0]
    train_samples = int(X.shape[0]*train_split)
    X_train, Y_train = X[:train_samples+1,], Y[:train_samples+1,]
    X_test, Y_test = X[train_samples+1:,], Y[train_samples+1:,]
    print "X_train:", X_train.shape, " | Y_train:", Y_train.shape
    print "X_test:", X_test.shape, " | Y_test:", Y_test.shape
    return X_train, Y_train, X_test, Y_test


def train_model(X, Y, model, train_split=0.8, epochs=10, batch_size=32):
    global X_wiki, Y_wiki, X_cli, Y_cli, X_bio, Y_bio, X_fic, Y_fic, X_wikitest, Y_wikitest

    # This is only for training! (If train_split =1 then only TEST)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)

    #custom_fit(X_train, Y_train, X_test, Y_test, model=model, batch_size=batch_size, epochs=epochs)
    custom_fit(X_train, Y_train, X_test, Y_test, model=model, batch_size=batch_size, epochs=epochs)
    
    testing_on_data("Wikipedia(DEVELOPMENT)", X_test, Y_test, model, batch_size, summary_only=True, visualize=False)
    testing_on_data("Clinical", X_cli, Y_cli, model, batch_size, summary_only=True, visualize=False)
    testing_on_data("Fiction", X_fic, Y_fic, model, batch_size, summary_only=True, visualize=False)
    testing_on_data("Wikipedia(BENCHMARK)", X_wikitest, Y_wikitest, model, batch_size, summary_only=True, visualize=False)
        
    

if __name__ == "__main__":

    # Print parameters
    print "=== SCALE_LOSS_FUN: %d, ONE_SIDE_CONTEXT_SIZE: %d ===" % (int(SCALE_LOSS_FUN), ONE_SIDE_CONTEXT_SIZE)
    print "NOTE: Make sure you have MIN_SENTENCES_IN_DOCUMENT >= 2*context_size + 1"

    # For which_model == 2
    SAMPLE_TYPE_wiki, X_wiki, Y_wiki, trained_sample_handler = get_input(sample_type=2, shuffle_documents=True, pad=False)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_wiki.shape[0], -1, X_wiki[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # For which_model == 2
    # Biography data for training
    #SAMPLE_TYPE_bio, X_bio, Y_bio, trained_sample_handler = get_input(sample_type=5, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    #NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_bio.shape[0], -1, X_bio[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # Only for testing
    SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = get_input(sample_type=4, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    SAMPLE_TYPE_fic, X_fic, Y_fic, trained_sample_handler = get_input(sample_type=6, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    SAMPLE_TYPE_wikitest, X_wikitest, Y_wikitest, trained_sample_handler = get_input(sample_type=7, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    
    #print "Check data type"

    sequence_length = X_wiki[0].shape[-1]

    dictionary_object = trained_sample_handler.dictionary
    embedding_W = dictionary_object.get_embedding_weights()


    # COmment temporarily for context variation experiemnt
    if LOAD_SAVED_MODEL_AND_CONTINUE_TRAIN:
        model = load_saved_model()
    else:
        model = baseline_CNN(-1, EMBEDDING_DIM, embedding_W,  len(dictionary_object.word2id_dic))
        ####model = lstm_model(-1, EMBEDDING_DIM, embedding_W, None)

    #pdb.set_trace()

    train_model(X_wiki, Y_wiki, model, train_split=0.8, epochs=20, batch_size=64)
    ##train_LSTM(X_bio, Y_bio, model, embedding_W, train_split=0.7, epochs=1, batch_size=32)
