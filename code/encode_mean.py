import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as s_words
import pdb
import numpy as np
import string

from parse_xml import MIN_SENTENCES_IN_PARAGRAPH, INPUT_VECTOR_LENGTH
import codecs


def isINT(w):
    try:
        w = int(w)
    except ValueError:
        return 0
    return 1


class MeanWord2vec(object):
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('/home/pinkesh.badjatiya/WORD_EMBEDDINGS/word2vec__GoogleNews_100Bwords__300Dvectors__3M_vocab.bin', binary=True)
        #self.model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.stopwords = set(s_words.words('english') + [w for w in string.punctuation])
        self.vectorize = lambda x: self.model[x]
        self.AVERAGE_WORDS = 20


    def convert_sequence_sample_to_vec(self, sample, groundTruths):
        """ For type2 samples
        """
        # g_ths: Groundtruths
        sample_vec, g_ths = [], []
        for i, sentence in enumerate(sample):
            vec = []
            for w in word_tokenize(codecs.decode(sentence, "utf-8")):
                if (w not in self.stopwords) and (not isINT(w)):
                    try:
                        vec.append(self.model[w])
                    except KeyError:
                        # Skip all the words whose vector representation is not present in the word2vec pre-trained model
                        continue
            if len(vec) > 0:
                sample_vec.append(np.mean(vec, axis=0))
                g_ths.append(groundTruths[i])

        # Check vstack() or hstack()
        return np.vstack(sample_vec), np.asarray(g_ths).reshape((len(g_ths), 1))


    def convert_sample_to_vec(self, sample):
        """ For type1 samples
        """
        sample_vec = []
        for sentence in sample:
            vec = []
            for w in word_tokenize(codecs.decode(sentence, "utf-8")):
                if (w not in self.stopwords) and (not isINT(w)):
                    try:
                        vec.append(self.model[w])
                    except KeyError:
                        # Skip all the words whose vector representation is not present in the word2vec pre-trained model
                        continue
            if len(vec) > 0:
                sample_vec.append(np.mean(vec, axis=0))

        # Make the sentences in the sample equal to INPUT_VECTOR_LENGTH
        remove = len(sample_vec) - INPUT_VECTOR_LENGTH
        if remove > 0:
            print "Vectorized sentence len not equal to min sentences in sample, INPUT_VECTOR_LENGTH"
            return None
        elif remove < 0:
            # Do not have sentences equal to INPUT_VECTOR_LENGTH
            #print ">>>>>>>>>>>> Found %d sentences instead of %d" %(len(sample_vec), INPUT_VECTOR_LENGTH)
            return None

        try:
            if len(sample_vec) != len(sample):
                raise Exception(">>>>>>>>> Found %d sentences in a sample instead of %d. (Skipped sentences while using word2vec)" %(len(sample_vec), len(sample)))
            
            ##########################################################
            ############ Check karle hstack()/vstack() chahiye
            ##########################################################
            temp = np.vstack(sample_vec)
            #temp = np.hstack(sample_vec)
        except Exception, e:
            print ">>>>>>>>>>>>>>>", e
            #pdb.set_trace()
            return None
        return temp


if __name__=="__main__":
    word2vec = MeanWord2vec()
    pdb.set_trace()
