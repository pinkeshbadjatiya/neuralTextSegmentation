import os
import pdb

class LoadData:
    DIR="./data/"
    SUB_DIRS = {
        "fiction": "fiction/",
        "clinical": "clinical/",
        "ai": "ai/"
    }

    def __init__(self):
        self.documents = []

    def load_clinical_sequence(self):
        SAMPLE_TYPE = 4     # 4th sample type
        doc = self.load_clinical()
        sequence = []
        for chapter in doc:
            for par in chapter:
                for i, line in enumerate(par):
                    sequence.append((line, int(i==0)))
        return SAMPLE_TYPE, [sequence]   # Return a single sample with sequences of sentence

    
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

    


if __name__=="__main__":
    ld = LoadData()
    clinical_data = ld.load_clinical_sequence()
    pdb.set_trace()
