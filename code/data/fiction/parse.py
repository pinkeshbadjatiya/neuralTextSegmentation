import nltk, re, sys

#s = nltk.data.load("tokenizers/punkt/english.pickle")

replace_string = [(r'^ +', r'')]
filename = "/home/grim/Desktop/fiction_cleaned/old/" + sys.argv[1]
outfile = "/home/grim/Desktop/fiction_cleaned/new/" + sys.argv[1]

print filename

with open(filename) as fin, open(outfile, 'a') as fout:
    for line in fin.readlines():
        for (reg_old, reg_new) in replace_string:
            fout.write(re.sub(reg_old, reg_new, line))


#s.tokenize()



