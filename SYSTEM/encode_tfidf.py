from gensim import corpora, models
from pprint import pprint  # pretty-printer
import pdb

WIKIPEDIA_DATASET_PATH="/home/pinkesh/DATASETS/WIKIPEDIA_DATASET/enwiki-latest-pages-articles.xml.bz2"
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


# Tokenize the document & Lowercase it
texts = [[word for word in document.lower().split()] for document in documents]
print(texts)

# remove common words and tokenize
#stoplist = set('for a of the and to in'.split())
#texts = [[word for word in document if word not in stoplist] for document in texts]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
#dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored


#corpus = [dictionary.doc2bow(text) for text in texts]
print "Creating wiki corpus"
wiki = corpora.wikicorpus.WikiCorpus(WIKIPEDIA_DATASET_PATH) # create word->word_id mapping, takes almost 8h
pdb.set_trace()
print "Saving the corpus to file!"
MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format plus file with id->word
#corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
#pprint(corpus)



# Create a TF-IDF transform
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
print(tfidf[new_vec]) # step 2 -- use the model to transform vectors
