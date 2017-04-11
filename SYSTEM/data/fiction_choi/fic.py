import os
import nltk
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


tok = nltk.data.load("tokenizers/punkt/english.pickle")
files = os.listdir("./fiction/new")

for fil in files:
    with open("./fiction/new/" + fil) as f:
        print fil
        with open("fic_saved/" + fil, "a") as f2:
            data = f.readlines()
            start_idx = -1
            for i, line in enumerate(data):
                line = line.decode("UTF-8")
                if i == start_idx+1:
                    if i == 0:
                        f2.write("==========\r\n")
                    continue
                elif line[:3] == "===":
                    f2.write("==========\r\n")
                    start_idx = i
                else:
                    line = line.strip().strip(".").strip()
                    if len(line) > 0:
                        sents = tok.tokenize(line)
                        for sen in sents:
                            f2.write(sen + "\r\n")
            
        
