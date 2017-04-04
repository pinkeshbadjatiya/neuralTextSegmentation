import numpy as np
import string

import nltk
nltk.data.path.append("../nltk_data")

from nltk.corpus import stopwords as s_words
from nltk.stem.wordnet import WordNetLemmatizer


class Dictionary:

    def __init__(self, EMBEDDING_DIM, word2vec_model):
        self.id2word_dic = {}
        self.word2id_dic = {}
        
        self.word2id_dic['<UNK>'] = 0
        self.id2word_dic[0] = '<UNK>'
        self.id_to_use_count = 1

        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.word2vec = word2vec_model
        self.ignore_words = ['of', 'and', 'to', 'a']

        self.lemmatizer = WordNetLemmatizer()



    def normalize_word(self, word):
        # Do not lowercase, (`the` VS `The`)
        #word = word.lower()
        
        # Remove puct if needed
        if word in [w for w in string.punctuation]:
            return None

        # Remove stop words if needed
        if w in set(s_words.words('english')):
            return None

        return word


    def word2id(self, word):
        word = self.normalize_word(word)
        if word is None:
            return None

        try:
            return self.word2id_dic[word]
        except KeyError:
            if word not in self.word2vec:
                #if word not in self.ignore_words:       # Print words only if they do not belong to this list
                #    print "WORD2VEC:  `%s` not found" %(word)

                # Try to lemmatize the word and find its embedding
                # If not found then return UNK emb
                lm_word = self.lemmatizer.lemmatize(word)
                if lm_word not in self.word2vec:
                    return self.word2id_dic['<UNK>']
                print "Lemmatized: %s ---> %s" %(word, lm_word)
                word = lm_word
                if word in self.word2id_dic:
                    return self.word2id_dic[word]

            self.word2id_dic[word] = self.id_to_use_count
            self.id2word_dic[self.id_to_use_count] = word
            self.id_to_use_count += 1
            return self.id_to_use_count - 1

    def id2word(self, wid):
        return self.id2word_dic[wid]

    def get_embedding_weights(self):
        embedding = np.zeros((len(self.word2id_dic) + 1, self.EMBEDDING_DIM))
        n = 0
        for k, v in self.word2id_dic.iteritems():
            try:
                embedding[v] = self.word2vec[k]
            except Exception, e:
                print e
                n += 1
                pass
        print "%d embedding missed"%n
        #pdb.set_trace()
        return embedding

