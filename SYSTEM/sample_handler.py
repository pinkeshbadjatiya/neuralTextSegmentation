import nltk.data
import pdb

from encode_tfidf import TFIDF
from encode_mean import MeanWord2vec
from encode_2d_repr import CustomSent2vec

from helper import unison_shuffled_copies
from parse_xml import DataHandler
from numpy import vstack
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# This is the no of lines in each sample
from parse_xml import INPUT_VECTOR_LENGTH

import time
import load_data
import progbar

AVERAGE_WORDS = 20
STATIC_PAD = 1

def get_input(sample_type, shuffle_documents, pad, trained_sent2vec_model=None):
    # Returns X, Y
    # X: Each row is a sample
    # Y: A 1-D vector for ground truth
    # Also pads the sample input as per the mentioned value of INPUT_VECTOR_LENGTH is needed

    start = time.time()
    data_handler = DataHandler()

    print "==========================================="
    if sample_type == 1:
        # NOT SURE ABOUT THIS TYPE!
        sample_type, samples = data_handler.get_samples()            # Get samples, each sample is a set of INPUT_VECTOR_LENGTH consecutive sentences. No document information captured
    elif sample_type in (2, 3):
        # type2 : Get samples, each sample is a document (a set of sentences resulting in a sequence), or, (NUM_DOCUMENTS, NUM_SENTENCES, SENTENCE)
        # type3 : Same as type2 just merge the samples to remove the sequence information and treat as simple sentence classification problem, i.e. (TOTAL_NUM_SENTENCES, SENTENCE)
        #         This processing will be done in the cnn_clssifier.py itself.
        sample_type, samples = data_handler.get_sequence_samples(sample_type)
        #sample_type, samples = data_handler.get_sequence_samples_PARALLEL()  # Get samples, each sample is a document (a set of sentences resulting in a sequence)
    elif sample_type == 4:
        # type4: Clinical sequence of a single sample
        # X.shape = (1, TOTAL_SENTENCES)
        # Y.shape = (TOTAL_SENTENCES, 1)
        ld = load_data.LoadData()
        sample_type, samples = ld.load_clinical_sequence()
    elif sample_type == 5:
        # type5: Biography sequence of a single sample
        # X.shape = (1, TOTAL_SENTENCES)
        # Y.shape = (TOTAL_SENTENCES, 1)
        ld = load_data.LoadData()
        sample_type, samples = ld.load_biography_sequence()
    elif sample_type == 6:
        # type5: Fiction sequence of a multiple documents
        # X.shape = (NO_OF_BOOKS, TOTAL_SENTENCES)
        # Y.shape = (NO_OF_BOOKS, TOTAL_SENTENCES, 1)
        ld = load_data.LoadData()
        sample_type, samples = ld.load_fiction_sequence()
    else:
        print "NOTE: INVALID SAMPLE_TYPE!"
        return None

    del data_handler
    print "Samples Loading took", time.time() - start, "seconds"

    model = trained_sent2vec_model
    if not trained_sent2vec_model:
        #model = TFIDF()
        #model = MeanWord2vec()
        model = CustomSent2vec()

    X, Y = [], []
    _total_samples,_start_time = len(samples), time.time()
    for _idx, sample in enumerate(samples):
        # Each sample is a document
        # Each sample is a list of tuples with each tuple as (sentence, groundTruth)
        sentences, groundTruths = zip(*sample)        # Unpack a sample

        if not _idx%50:
            progbar.simple_update("Converting doc to martices", _idx+1, _total_samples, time_elapsed=(time.time() - _start_time))

        if sample_type == 1:
            # Correct groundtruth sync problem here
            sentences, groundTruths = model.convert_sample_to_vec(sentences, groundTruths)
        elif sample_type in (2, 3, 4, 5, 6):
            sentences, groundTruths = model.convert_sequence_sample_to_vec(sentences, groundTruths)
        else:
            print "Wrong Sample TYPE"

        if sentences is None:
            continue
        X.append(sentences)            # X[0].shape = matrix([[1,2,3,4.....]])
        Y.append(np.asarray(groundTruths))          # Y[0] = [1, 0, 0, ..... 0, 1, 0, 1....]
    progbar.simple_update("Creating a standalone matrix for samples...", -1, -1)
    X, Y = np.asarray(X), np.asarray(Y)
    progbar.end()
    

    print "Total samples: %d" %(len(X))
    if shuffle_documents: # Shuffle the X's and Y's if required
        X, Y = unison_shuffled_copies(X, Y)
        print "SHUFFLE: Shuffled input document order! (X:",X.shape,", Y:",Y.shape,")"

    if sample_type == 2 and pad == False:
        print "NOTE: Sample type2 requires PADDING!"

    if pad:
        if STATIC_PAD:
            max_len = AVERAGE_WORDS
        else:
            max_len = None  # Uses the max length of the sequences

        doc_lengths = [len(doc) for doc in X]
        print "Padding sequences. Doc-lengths: Mean=%d, Std=%d" %(np.mean(doc_lengths), np.std(doc_lengths))
        # Both of them have to be in unison
        X = pad_sequences(X, padding="post", truncating="post", value=0.0, dtype=np.float32)
        Y = pad_sequences(Y, padding="post", truncating="post", value=0.0, dtype=np.float32)
        print "Size of new X(after padding):", X.shape

    return sample_type, X, Y, model
    #return sample_type, X, Y




if __name__=="__main__":
    get_input()
    pdb.set_trace()
