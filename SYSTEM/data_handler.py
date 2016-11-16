import nltk.data

from encode_tfidf import TFIDF
from encode_mean import MeanWord2vec

from parse_xml import get_samples
from numpy import vstack
import numpy as np
import pdb

# This is for each paragraph, or across a split point
INPUT_VECTOR_LENGTH=100

sentence_tzr = nltk.data.load("tokenizers/punkt/english.pickle")


####################################
### FORMALIZE THIS SECTION
####################################
def get_input():
    # Returns X, Y
    # X: Each row is a sample
    # Y: A 1-D vector for ground truth
    # Also pads the input as per the mentioned value of INPUT_VECTOR_LENGTH is needed

    samples, samples_NEG = get_samples()
    
    #model = TFIDF() 
    model = MeanWord2vec() 
    
    X = []
    Y = []
    for sample in samples:
        # Each sample is a set of consecutive paragraphs, initially it is 2 paragraphs
        # Virtually, each paragraph is like a document
        sample = model.convert_sample_to_vec(sample)
        if sample == None:
            continue
        X.append(sample)
        # X[0].shape = matrix([[1,2,3,4.....]])
        Y += [1]
        # Y[0] = [0, 0, 0, ..... 1, 1, 0, 0....]


    X_neg = []
    Y_neg = []
    for sample in samples_NEG:
        # Each sample is a set of consecutive paragraphs, initially it is 2 paragraphs
        # Virtually, each paragraph is like a document
        sample = model.convert_sample_to_vec(sample)
        if sample == None:
            continue
        X_neg.append(sample)
        # X[0].shape = matrix([[1,2,3,4.....]])
        Y_neg += [0]
        # Y[0] = [0, 0, 0, ..... 1, 1, 0, 0....]

    return vstack(X + X_neg), Y + Y_neg
    #return X, Y, X_neg, Y_neg
    

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
