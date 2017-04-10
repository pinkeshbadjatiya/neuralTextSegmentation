import os, pdb

import nltk
nltk.data.path.append("../nltk_data")

class LoadData:
    DIR="./data/"
    SUB_DIRS = {
        "fiction": "fiction/new/",
        "clinical": "clinical/",
        "ai": "ai/",
        "biography": "biography/chapters/",
        "wikipedia": "wiki/"
    }

    def __init__(self):
        # load_XXX_sequence returns [[(s1, gt1),(s2, gt2), (s3, gt3)..., (sN, gtN)], [], ... []]
        self.documents = []
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    ####### WIKIPEDIA DATA ##########################################################################################
    def load_wikipedia_sequence(self):
        SAMPLE_TYPE = 7     # 7th sample type
        doc = self.load_wikipedia()
        documents = []
        for wiki_page in doc:
            sequence = []
            for par in wiki_page:
                for i, line in enumerate(par):
                    sequence.append((line, int(i==0)))
            documents.append(sequence)
        return SAMPLE_TYPE, documents   # Return multiple samples with sequences of sentence (each sample is a chapter)

    
    def load_wikipedia(self):
        dirname = self.DIR + self.SUB_DIRS['wikipedia']
        
        # Use only .ref files
        files = sorted([dirname+fil for fil in os.listdir(dirname) if fil.endswith(".ref")])

        document = []           # Initialize document
        for fil in files:
            page = []
            with open(fil) as f:
                data = f.readlines()
            paragraph = []
            for line in data:
                line = line.strip()
                if line.startswith("======="):
                    if len(paragraph) > 0:
                        page.append(paragraph)
                    paragraph = []
                else:
                    paragraph.append(line)
            if len(paragraph) > 0:
                page.append(paragraph)
            document.append(page)

        print "Total WIKIPEDIA(test) data: %d chapters, %d paragraphs and %d sentences" %(len(document), sum([len(chap) for chap in document]), sum([sum([len(par) for par in chap]) for chap in document]))
        return document

    ####### CLINICAL DATA ##########################################################################################
    def load_clinical_sequence(self):
        SAMPLE_TYPE = 4     # 4th sample type
        doc = self.load_clinical()
        documents = []
        for chapter in doc:
            sequence = []
            for par in chapter:
                for i, line in enumerate(par):
                    sequence.append((line, int(i==0)))
            documents.append(sequence)
        return SAMPLE_TYPE, documents   # Return multiple samples with sequences of sentence (each sample is a chapter)

    
    def load_clinical(self):
        dirname = self.DIR + self.SUB_DIRS['clinical']
        
        # Use only .ref files
        files = sorted([dirname+fil for fil in os.listdir(dirname) if fil.endswith(".ref")])

        document = []           # Initialize document
        for fil in files:
            chapter = []
            with open(fil) as f:
                data = f.readlines()
            paragraph = []
            for line in data:
                line = line.strip()
                if line.startswith("======="):
                    if len(paragraph) > 0:
                        chapter.append(paragraph)
                    paragraph = []
                else:
                    paragraph.append(line)
            if len(paragraph) > 0:
                chapter.append(paragraph)
            document.append(chapter)

        print "Total CLINICAL data: %d chapters, %d paragraphs and %d sentences" %(len(document), sum([len(chap) for chap in document]), sum([sum([len(par) for par in chap]) for chap in document]))
        return document

    ####### BIOGRAPHY DATA ##########################################################################################
    def load_biography_sequence(self):
        SAMPLE_TYPE = 5     # 5th sample type
        doc = self.load_biography()
        sequence = []
        for chapter in doc:
            for par in chapter:
                for i, line in enumerate(par):
                    sequence.append((line, int(i==0)))
        return SAMPLE_TYPE, [sequence]   # Return a single sample with sequences of sentence

    
    def load_biography(self):
        dirname = self.DIR + self.SUB_DIRS['biography']
        
        # Use only .txt files
        files = sorted([dirname+fil for fil in os.listdir(dirname) if fil.endswith(".txt")])

        document = []           # Initialize document
        for fil in files:
            chapter = []
            with open(fil) as f:
                data = f.readlines()
            paragraph = []
            for line in data:
                line = line.strip().strip(".")      # Remove full-stop as well
                if line.startswith("======="):
                    if len(paragraph) > 0:
                        chapter.append(paragraph)
                    paragraph = []
                else:
                    if len(line) > 0:
                        paragraph.append(line)
            if len(paragraph) > 0:
                chapter.append(paragraph)
            document.append(chapter)

        print "Total BIOGRAPHY data: %d chapters, %d paragraphs and %d sentences" %(len(document), sum([len(chap) for chap in document]), sum([sum([len(par) for par in chap]) for chap in document]))
        return document

    
    ####### FICTION DATA ##########################################################################################
    def load_fiction_sequence(self):
        SAMPLE_TYPE = 6     # 6th sample type
        doc_data = self.load_fiction()
        documents = []
        for book in doc_data:
            sequence = []
            for par in book:
                for i, line in enumerate(par):
                    sequence.append((line, int(i==0)))
            documents.append(sequence)
        return SAMPLE_TYPE, documents   # Return multiple sample with sequences of sentences. Each sample is a book

    
    def load_fiction(self, datatype='testing'):

        if datatype is 'testing':
            print "##### Loading FICTION: testing data ######"
        else:
            print "##### Loading FICTION: development data ######"

        extension = '.dev' if datatype is 'development' else '.ref'
        dirname = self.DIR + self.SUB_DIRS['fiction']
        
        # Use appropriate file type for data
        files = sorted([dirname+fil for fil in os.listdir(dirname) if fil.endswith(extension)])

        document = []           # Initialize document
        for fil in files:
            book = []
            with open(fil) as f:
                data = f.readlines()
            paragraph = []
            for line in data:
                line = line.strip().strip(".")      # Remove full-stop as well
                if line.startswith("======="):
                    if len(paragraph) > 0:
                        book.append(paragraph[1:])  # Do not add the 1st line as it is the TITLE of the section
                    paragraph = []
                else:
                    if len(line) > 0:
                        # Each line does not have a single sentence. There are multiple sentences in a single line as well.
                        #paragraph.extend(self.sent_tokenizer.tokenize(line))    # Tokenizer splits a chunk of sentences into 1 or more sentences, so we need to extend the main list.
                        paragraph.append(line)
            if len(paragraph) > 0:
                book.append(paragraph[1:])      # Do not add the 1st line as it is the TITLE of the section
            document.append(book)

        print "Total FICTION data: %d chapters, %d paragraphs and %d sentences" %(len(document), sum([len(chap) for chap in document]), sum([sum([len(par) for par in chap]) for chap in document]))
        return document

    

if __name__=="__main__":
    ld = LoadData()
    SAMPLE_TYPE_cli, clinical_data = ld.load_clinical_sequence()
    SAMPLE_TYPE_bio, biography_data = ld.load_biography_sequence()
    SAMPLE_TYPE_fic, fiction_data = ld.load_fiction_sequence()
    pdb.set_trace()
