import nltk.data
import pdb

from encode_tfidf import TFIDF
from encode_mean import MeanWord2vec

from helper import unison_shuffled_copies
from parse_xml import DataHandler
from numpy import vstack
import numpy as np
import pdb

# This is the no of lines in each sample
from parse_xml import INPUT_VECTOR_LENGTH



def get_input(shuffle=False):
    # Returns X, Y
    # X: Each row is a sample
    # Y: A 1-D vector for ground truth
    # Also pads the input as per the mentioned value of INPUT_VECTOR_LENGTH is needed

    data_handler = DataHandler()
    samples = data_handler.get_samples()     # Get samples

    #model = TFIDF()
    model = MeanWord2vec()

    X = []
    Y = []
    for sample in samples:
        # Each sample is a list of tuples with each tuple as (sentence, groundTruth)
        sentences, groundTruths = zip(*sample)        # Unpack a sample
        sentences = model.convert_sample_to_vec(sentences)
        if sentences == None:
            continue
        X.append(sentences)            # X[0].shape = matrix([[1,2,3,4.....]])
        Y.append(groundTruths)          # Y[0] = [1, 0, 0, ..... 0, 1, 0, 1....]

    if shuffle: # Shuffle the X's and Y's if required
        return unison_shuffled_copies(np.asarray(X), np.asarray(Y))
    return np.asarray(X), np.asarray(Y)


# # Might fail if the classification is not binary!!!
# def pad_vector(vec, sample_position, total_samples):
#     # Skip if INPUT_VECTOR_LENGTH is negative as the vector transformation is already fixed size
#     if INPUT_VECTOR_LENGTH < 0:
#         return vec
#
#     if len(vec) >= INPUT_VECTOR_LENGTH:
#         if sample_position == 0:                        # 1st paragraph
#             return vec[-INPUT_VECTOR_LENGTH:]
#         elif sample_position == total_samples - 1:   # Last paragraph
#             return vec[:INPUT_VECTOR_LENGTH]
#         else:                                           # Middle paragraph
#             raise Exception("Dont know how to handle middle paragraphs")
#     raise Exception("The pad_vector() needs vectors that are >=INPUT_VECTOR_LENGTH defined")
#
#
# def convert_sample_to_vec(doc):
#     # Using tfidf
#         for i, paragraph in enumerate(sample):
#     string_doc =
#
#
#     ##################################
#     ## Using other methods if needed
#     ##################################
#     # sent_vec = []
#     # for sentence in doc:
#     #     sent_vec += sentence2vec(sentence)
#     # sent_vec = pad_vector(sent_vec, i, len(sample))
#     # return sent_vec
#     return None
