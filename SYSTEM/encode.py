import os
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_DIR = "./AAs"

corpuses = [DATASET_DIR + "/" + i for i in os.listdir(DATASET_DIR)]
test_data = ["./test_doc.xml"]


vectorizer = TfidfVectorizer('filename', min_df=1, ngram_range=(1, 2))
#vectorizer = TfidfVectorizer('filename', min_df=1)
#vectorizer = TfidfVectorizer('string', min_df=1)
X_train = vectorizer.fit_transform(corpuses)
pdb.set_trace()
X_test = vectorizer.transform(test_data)
