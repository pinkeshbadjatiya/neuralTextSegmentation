from keras.models import Sequential, Model
from keras.models import load_model
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
from tabulate import tabulate

#from attention_lstm import AttentionLSTM
#from attention_lstm_without_weights import Attention
from final_attention_layer import Attention
import progbar, time

trained_sample_handler = None
SAMPLE_TYPE_wiki, X_wiki, Y_wiki = None, None, None           # Training + Testing (need to split)
SAMPLE_TYPE_cli, X_cli, Y_cli = None, None, None              # Testing
SAMPLE_TYPE_bio, X_bio, Y_bio = None, None, None              # Testing
SAMPLE_TYPE_fic, X_fic, Y_fic = None, None, None              # Testing

GLOVE_EMBEDDING_DIM = 300
SCALE_LOSS_FUN = True

# SEQUENCES_LENGTH_FOR_TRAINING = 40

ONE_SIDE_CONTEXT_SIZE = 7

LOAD_SAVED_MODEL_AND_CONTINUE_TRAIN = False
SAVE_MODEL_AFTER_EACH_EPOCH = False
TRAINABLE_EMBEDDINGS = True


np.random.seed(12345)   # IMP seed
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
        


def lstm_model(sequences_length_for_training, embedding_dim, embedding_matrix, vocab_size):

    which_model = 2
    # Look back is equal to -INF
    # This model creates a Stateful LSTM with lookback of the whole document
    # Input should be of the format (TOTAL_DOCUMENTS, TOTAL_SEQUENCES, SEQUENCE_DIM)
    # Also train using the custom trainer

    print 'Build MAIN model...'
    #pdb.set_trace()
    ngram_filters = [2, 3, 4, 5]
    conv_hidden_units = [300, 300, 300, 300]
    
    left_context= Input(shape=(ONE_SIDE_CONTEXT_SIZE+1, embedding_dim), dtype='float32', name='left-context')
    main_input = Input(shape=(1, embedding_dim), dtype='float32')
    right_context = Input(shape=(ONE_SIDE_CONTEXT_SIZE+1, embedding_dim), dtype='float32')

    context_embedder = TimeDistributed(Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, weights=[embedding_matrix], input_length=embedding_dim, init='uniform'))
    main_input_embedder = TimeDistributed(Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, weights=[embedding_matrix], input_length=embedding_dim, init='uniform'))
    embedded_input_left, embedded_input_main, embedded_input_right = context_embedder(left_context), main_input_embedder(main_input), context_embedder(right_context)

    convsL, convsM, convsR = [], [], []
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv_layer = Convolution1D(nb_filter=hidden_units,
                             filter_length=n_gram,
                             #border_mode='same',
                             border_mode='valid',
                             activation='tanh', name='Convolution-'+str(n_gram)+"gram")
        lef = TimeDistributed(conv_layer, name='TD-convolution-left-'+str(n_gram)+"gram")(embedded_input_left)
        mid = TimeDistributed(conv_layer, name='TD-convolution-mid-'+str(n_gram)+"gram")(embedded_input_main)
        rig = TimeDistributed(conv_layer, name='TD-convolution-right-'+str(n_gram)+"gram")(embedded_input_right)

        # Use Flatten() instead of MaxPooling()
        #flat_L = TimeDistributed(Flatten(), name='TD-flatten-left-'+str(n_gram)+"gram")(lef)
        #flat_M = TimeDistributed(Flatten(), name='TD-flatten-mid-'+str(n_gram)+"gram")(mid)
        #flat_R = TimeDistributed(Flatten(), name='TD-flatten-right-'+str(n_gram)+"gram")(rig)
        #convsL.append(flat_L), convsM.append(flat_M), convsR.append(flat_R)

        # Use GlobalMaxPooling1D() instead of Flatten()
        pool_L = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-left-'+str(n_gram)+"gram")(lef)
        pool_M = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-mid-'+str(n_gram)+"gram")(mid)
        pool_R = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-right-'+str(n_gram)+"gram")(rig)
        convsL.append(pool_L), convsM.append(pool_M), convsR.append(pool_R)

    convoluted_left, convoluted_mid, convoluted_right = Merge(mode='concat')(convsL), Merge(mode='concat')(convsM), Merge(mode='concat')(convsR)
    CONV_DIM = sum(conv_hidden_units)

    flat_mid = Flatten()(convoluted_mid)
    encode_mid = Dense(512, name='dense-intermediate-mid-encoder')(flat_mid)

    context_encoder = Bidirectional(LSTM(1000, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), consume_less='gpu', dropout_W=0.1, dropout_U=0.1, return_sequences=True, stateful=False), merge_mode='concat')
    encode_left, encode_right = Attention(name='encode-left-attention')(context_encoder(convoluted_left)), Attention(name='encode-right-attention')(context_encoder(convoluted_right))
    encode_left_drop, encode_mid_drop, encode_right_drop = Dropout(0.05)(encode_left), Dropout(0.05)(encode_mid), Dropout(0.05)(encode_right)

    encoded_info = Merge(mode='concat', name='encode_info')([encode_left_drop, encode_mid_drop, encode_right_drop])

    decoded = Dense(600, name='decoded')(encoded_info)
    decoded_drop = Dropout(0.1, name='decoded_drop')(decoded)
    
    output = Dense(1, activation='sigmoid')(decoded_drop)
    model = Model(input=[left_context, main_input, right_context], output=output)
    model.layers[1].trainable = TRAINABLE_EMBEDDINGS
    #model.compile(loss=w_binary_crossentropy, optimizer='rmsprop', metrics=['accuracy', 'recall'])
    model.compile(loss=w_binary_crossentropy, optimizer='adadelta', metrics=['accuracy', 'recall'])


    print model.summary(line_length=150, positions=[.46, .65, .77, 1.])
    return model

def batch_gen_consecutive_context_segments_from_big_seq(X_with_doc, Y_with_doc, batch_size, one_side_context_size):
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

    for epoch in range(start_epoch, epochs):
        mean_tr_acc, mean_tr_loss, mean_tr_rec = [], [], []
        rLoss, rRecall, rAcc = 0,0,0 # Running parameters for printing while training
        for batch_count, (batch_X_left, batch_X_mid, batch_X_right, batch_Y_mid) in enumerate(batch_gen_consecutive_context_segments_from_big_seq(X_train, Y_train, batch_size, ONE_SIDE_CONTEXT_SIZE)):
            #batch_Y_vec = to_categorical_MULTI_DIM(batch_Y, nb_classes=2)
            try:
                start = time.time()
                tr_loss, tr_acc, tr_rec = model.train_on_batch([batch_X_left, batch_X_mid, batch_X_right], batch_Y_mid)
                speed = time.time() - start

                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
                mean_tr_rec.append(tr_rec)
                #rLoss, rRecall, rAcc = (rLoss*batch_count + tr_loss)/(batch_count + 1), (rRecall*batch_count + tr_rec)/(batch_count + 1), (rAcc*batch_count + tr_acc)/(batch_count + 1)
                #progbar.prog_bar(True, total_sentences, epochs, batch_size, epoch, batch_count, speed=speed, data={ 'rLoss': rLoss, 'rAcc': rAcc, 'rRec': rRecall })
                progbar.prog_bar(True, total_sentences, epochs, batch_size, epoch, batch_count, speed=speed, data={ 'Loss': tr_loss, 'Acc': tr_acc, 'Rec': tr_rec })

            except KeyboardInterrupt, SystemExit:
                print ""
                print "########################################################"
                print "######  Pausing execution. Press ENTER to continue #####"
                print "########################################################"
                out = raw_input('Enter "pdb" to get prompt or ENTER to exit.> ')
                if out == "pdb":
                    pdb.set_trace()

        progbar.end()
        if SAVE_MODEL_AFTER_EACH_EPOCH:
            model.save("model_trainable_%s_epoc_%d.h5" %(str(TRAINABLE_EMBEDDINGS), epoch+1))

        print ">> Epoch: %d/%d" %(epoch+1, epochs)
        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('recall training = {}'.format(np.mean(mean_tr_rec)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))

        testing_on_data("Wikipedia", X_test, Y_test, model, batch_size, summary_only=True)
        testing_on_data("Clinical", X_cli, Y_cli, model, batch_size)
        testing_on_data("Biography", X_bio, Y_bio, model, batch_size)
        testing_on_data("Fiction", X_fic, Y_fic, model, batch_size, summary_only=True)
    
        print('___________________________________')
    
    # Testing
    print "####################################################################"
    print ">> (TEST) >> Testing, X:", X_test.shape, "Y:", Y_test.shape
    mean_te_acc, mean_te_loss, mean_te_rec = [], [], []
    for batch_X_left, batch_X_mid, batch_X_right, batch_Y_mid in batch_gen_consecutive_context_segments_from_big_seq(X_test, Y_test, batch_size, ONE_SIDE_CONTEXT_SIZE):
        te_loss, te_acc, te_rec = model.test_on_batch([batch_X_left, batch_X_mid, batch_X_right], batch_Y_mid)
        mean_te_acc.append(te_acc)
        mean_te_loss.append(te_loss)
        mean_te_rec.append(te_rec)

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('recall testing = {}'.format(np.mean(mean_te_rec)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    


def testing_on_data(type_of_data, X_test, Y_test, model, batch_size, summary_only=False):
    # Predicting
    print "====================== %s ======================" %(type_of_data)
    print "Predicting... (SEPARATELY FOR EACH DOCUMENT)"
    data = {
        'wd': [],
        'pk': []
    }
    avg_segment_lengths_across_test_data = [] # Average segment length across the documents
    for Xi_test, Yi_test in zip(X_test, Y_test):
        pred_per_doc = []
        Xi_test, Yi_test = Xi_test.reshape((1,) + Xi_test.shape), Yi_test.reshape((1,) + Yi_test.shape)   # Convert to format of 1 document
        for batch_X_left, batch_X_mid, batch_X_right, batch_Y_mid in batch_gen_consecutive_context_segments_from_big_seq(Xi_test, Yi_test, batch_size, ONE_SIDE_CONTEXT_SIZE):
            batch_y_pred = model.predict_on_batch([batch_X_left, batch_X_mid, batch_X_right])
            pred_per_doc.append(batch_y_pred)

        if not len(pred_per_doc): # batch generator might drop a few documents
            continue

        #rounded = np.round(pred_per_doc)
        pred_per_doc = np.concatenate(pred_per_doc, axis=0)
        actual_avg_seg_length, result = helper.windiff_and_pk_metric_ONE_SEQUENCE(Yi_test[0], pred_per_doc, window_size=-1, rounded=False, print_individual_stats=not summary_only)
        avg_segment_lengths_across_test_data.append(actual_avg_seg_length)
        data['pk'].append(result['pk'])
        data['wd'].append(result['wd'])

    print ">> Summary (%s):" %(type_of_data)
    print "AVG segment length in test data: %f" % (np.mean(avg_segment_lengths_across_test_data))
    print "WinDiff metric:: avg: %f | std: %f | min: %f | max: %f" %(np.mean(data['wd']), np.std(data['wd']), np.min(data['wd']), np.max(data['wd']))
    print "Pk metric:: avg: %f | std: %f | min: %f | max: %f" %(np.mean(data['pk']), np.std(data['pk']), np.min(data['pk']), np.max(data['pk']))
    print('___________________________________')



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
    global X_wiki, Y_wiki, X_cli, Y_cli, X_bio, Y_bio, X_fic, Y_fic

    which_model = 2

    # This is only for training! (If train_split =1 then only TEST)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, train_split=train_split)

    if which_model == 2:
        #custom_fit(X_train, Y_train, X_test, Y_test, model=model, batch_size=batch_size, epochs=epochs)
        custom_fit(X_train, Y_train, X_test, Y_test, model=model, batch_size=batch_size, epochs=epochs)
        
        print ">>>>>> Final Testing <<<<<<"
        print ">> ATTENTION weights:"
        attn_weights = [model.get_layer("encode_left").get_weights(), model.get_layer("encode_right").get_weights()]
        print attn_weights[0]
        print attn_weights[1]

        testing_on_data("Wikipedia", X_test, Y_test, model, batch_size, summary_only=True)
        testing_on_data("Clinical", X_cli, Y_cli, model, batch_size)
        testing_on_data("Biography", X_bio, Y_bio, model, batch_size)
        testing_on_data("Fiction", X_fic, Y_fic, model, batch_size, summary_only=True)

        
#    elif which_modle == 1:
#        # Works for TYPE2 but check for others
#        # Both these lines work for which_model == 1
#        model.fit(X_train, Y_train, shuffle=False, nb_epoch=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
#    
#        # WIkipedia
#        #model.evaluate(X_test, Y_test, batch_size=batch_size)
#        #pred = model.predict(X_test)
#        #rounded = np.round(pred)
#        #result = helper.windiff_metric_NUMPY(Y_test, rounded)
#        #print result
#    
#        
#        # Clinical
#        # Temporary TRUNCATION
#        TRUNCATE_LEN = X_train.shape[1]
#        print "NOTE: Truncating the Test dataset(clinical) from %d sentences to %d sentences." %(X_cli.shape[1], TRUNCATE_LEN)
#        X_cli, Y_cli = X_cli[:,:TRUNCATE_LEN,:], Y_cli[:,:TRUNCATE_LEN,:]
#        model.evaluate(X_cli, Y_cli, batch_size=batch_size)
#        pred = model.predict(X_cli)
#        rounded = np.round(pred)
#        _, result = helper.windiff_metric_NUMPY(Y_cli, rounded, win_size=10, rounded=True)
#        print result


    pdb.set_trace()

    #rounded = [round(x) for x in pred]
    

if __name__ == "__main__":

    # Print parameters
    print "=== SCALE_LOSS_FUN: %d, ONE_SIDE_CONTEXT_SIZE: %d ===" % (int(SCALE_LOSS_FUN), ONE_SIDE_CONTEXT_SIZE)
    print "NOTE: Make sure you have MIN_SENTENCES_IN_DOCUMENT >= 2*context_size + 1"

    # For which_model == 2
    SAMPLE_TYPE_wiki, X_wiki, Y_wiki, trained_sample_handler = get_input(sample_type=2, shuffle_documents=True, pad=False)
    NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_wiki.shape[0], -1, X_wiki[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # For which_model == 2
    # Biography data for training
    SAMPLE_TYPE_bio, X_bio, Y_bio, trained_sample_handler = get_input(sample_type=5, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    #NO_OF_SAMPLES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = X_bio.shape[0], -1, X_bio[0].shape[1]          #MAX_SEQUENCE_LENGTH is is already padded
    
    # Clinical - Only for testing
    SAMPLE_TYPE_cli, X_cli, Y_cli, trained_sample_handler = get_input(sample_type=4, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)
    
    # Fiction - Only for testing
    SAMPLE_TYPE_fic, X_fic, Y_fic, trained_sample_handler = get_input(sample_type=6, shuffle_documents=False, pad=False, trained_sent2vec_model=trained_sample_handler)

    dictionary_object = trained_sample_handler.dictionary
    embedding_W = dictionary_object.get_embedding_weights()

    print "#####################################################################"
    print "VOCAB_SIZE:",  len(dictionary_object.word2id_dic)
    print "#####################################################################"
    if LOAD_SAVED_MODEL_AND_CONTINUE_TRAIN:
        model = load_saved_model()
    else:
        model = lstm_model(-1, EMBEDDING_DIM, embedding_W, len(dictionary_object.word2id_dic))

    train_LSTM(X_wiki, Y_wiki, model, embedding_W, train_split=0.7, epochs=20, batch_size=60)
    #train_LSTM(X_bio, Y_bio, model, embedding_W, train_split=0.7, epochs=1, batch_size=32)
