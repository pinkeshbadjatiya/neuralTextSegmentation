import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as s_words
import pdb
import numpy as np
import string

from parse_xml import MIN_SENTENCES_IN_PARAGRAPH


def isINT(w):
    try:
        w = int(w)
    except ValueError:
        return 0
    return 1


class MeanWord2vec(object):
    def __init__(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def convert_sample_to_vec(self, sample):
        sample_vec = []
        vectorize = lambda x: self.model[x]
        stopwords = set(s_words.words('english') + [w for w in string.punctuation])
        for i, doc in enumerate(sample):
            # Doc is a list of sentences
            doc_vec = []
            for sentence in doc:
                vec = []
                for w in word_tokenize(sentence):
                    if (w not in stopwords) and (not isINT(w)):
                        try:
                            vec.append(self.model[w])
                        except KeyError:
                            # Skip all the words whose vector representation is not present in the word2vec pre-trained model
                            continue
                if len(vec) > 0:
                    doc_vec.append(np.mean(vec, axis=0))

            # Make the sentences in the paragraph equal to MIN_SENTENCES_IN_PARAGRAPH
            remove = len(doc_vec) - MIN_SENTENCES_IN_PARAGRAPH
            if remove > 0:
                if i == 0:                  # Remove from start
                    doc_vec = doc_vec[remove:]
                elif i == len(sample) - 1:      # Remove from end
                    doc_vec = doc_vec[:-remove]
                else:                       # Remove from both the directions
                    doc_vec = doc_vec[remove/2:-remove+remove/2]
            elif remove < 0:
                # Do not have sentences equal to MIN_SENTENCES_IN_PARAGRAPH
                print ">>>>>>>>>>>> Found %d sentences instead of %d" %(len(doc_vec), MIN_SENTENCES_IN_PARAGRAPH)
                return None

            sample_vec.append(np.hstack(doc_vec))

        try:
            if len(sample_vec) != len(sample):
                raise Exception(">>>>>>>>> Found %d paragraph in a sample instead of %d" %(len(sample_vec), len(sample)))
            temp = np.hstack(sample_vec)
        except Exception, e:
            print ">>>>>>>>>>>>>>>", e
            #pdb.set_trace()
            return None
        return temp


if __name__=="__main__":
    word2vec = MeanWord2vec()
    pdb.set_trace()
