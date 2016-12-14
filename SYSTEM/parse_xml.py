import re
import pdb
import nltk.data
import os

# Skip that condition if the value is -1
MIN_SENTENCES_IN_DOCUMENT = -1
MIN_SENTENCES_IN_SECTION = 1
MIN_SECTIONS = 2    # Exlcuding the 1st section
MIN_SENTENCES_IN_PARAGRAPH = 5      # Using the nltk tokenizer to get the approximate sentence count in a paragraph

INPUT_VECTOR_LENGTH = 100



class DataHandler:
    def __init__(self):
        ################ Constants #################
        self.REGEX_heading = re.compile(r'<(h[0-9])>(.*)\.</\1>')      # <h2>heading.</h>
        self.REGEX_document_start = re.compile(r'<doc *(id="([0-9]{1,})")? *(url=".*")? *(title=".*")?>')
        self.REGEX_document_end = re.compile(r'</doc>')

        self.WIKI_DOCS = "/home/pinkesh/DATASETS/WIKIPEDIA_DATASET/extracted_WIKIPEDIA/"
        if self.WIKI_DOCS[-1]!="/":
            raise Exception("Check the directory name")

        self.sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sample_creator = SampleCreator()

        ################## VARIABLES ###################
        self.document_id_to_title = {}  # Map for the document ID's
        self.documents = []             # Collects "Raw" documents extracted from the dataset
                                        # Document = [section in sections] where, section = [paragraph in paragraphs] where paragraph = [line in lines] (Tokenized using a tokenizer)

        self.required_samples = []           # Samples which are actually split segments
        self.required_samples_NEG = []       # Samples which are not split segments
        self.best_docs = []
        self._queue = []          # Handles the tokenised sentence in a paragraph, a temp variable



    def _create_structured_document(self, file_name):
        with open(file_name) as f:
            data = f.readlines()
    
        sections = []   # sections = [paragraph in paragraphs], where paragraph = [line in lines] after tokenizing using a tokenizer
        docID = None
        start_line = None
        paragraph = []
        for i, line in enumerate(data):
            line = line.decode("UTF-8")
    
            # Skip the blank lines or which have only one character.
            if len(line.strip()) < 1:
                continue
    
            lno = i + 1
            _doc_start, _doc_end, _heading = self.REGEX_document_start.match(line), self.REGEX_document_end.match(line), self.REGEX_heading.match(line)
            if _doc_start:
                if len(paragraph) > 0 or len(sections) > 0 or (docID is not None) or (start_line is not None):
                    print "SERIOUS PROBLEM !!"
                paragraph, sections, docID, start_line = [], [], _doc_start.group(2), lno
                print "docID: %s" % (docID)
            elif _doc_end:
                # Flush old data & reset
                self.documents.append((docID, sections))
                docID, sections, paragraph, start_line = None, [], [], None
            elif not start_line and lno == start_line + 1:   # Doc title
                document_id_to_title[docID] = line
            elif _heading:
    
                # Do this only for the top sections as we are using the paragraphs
                # to learn split points
                if len(paragraph) == 0:
                    continue
    
                sections.append(paragraph)
                paragraph = []
            else:
                paragraph.append(self.sentence_tokenizer(line))


    
    def filter_docs(self):
        best_docs = []
        for (docID, sections) in self.documents:
            # Remove the 1st section as it might be very complex in structure
            sections = sections[1:]
    
            # Remove documents with less than 2 sections
            if MIN_SECTIONS != -1:
                if len(sections) <= MIN_SECTIONS:
                    print docID, ": Fails at MIN_SECTIONS (", len(sections), "/", MIN_SECTIONS, ")"
                    continue
    
            sentence_counts = [[len(par) for par in section] for section in sections]
    
            # Remove documents that have less than MIN_SENTENCES_IN_DOCUMENT.
            if MIN_SENTENCES_IN_DOCUMENT != -1:
                count = sum([sum(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_DOCUMENT:
                    print docID, ": Fails at MIN_SENTENCES_IN_DOCUMENT (", count, "/", MIN_SENTENCES_IN_DOCUMENT,")"
                    continue
    
            # Remove documents that have less than MIN_SENTENCES_IN_SECTION
            if MIN_SENTENCES_IN_SECTION != -1:
                count = min([sum(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_SECTION:
                    print docID, ": Fails at MIN_SENTENCES_IN_SECTION (", count,"/", MIN_SENTENCES_IN_SECTION,")"
                    continue
                    
            if MIN_SENTENCES_IN_PARAGRAPH != -1:
                count = min([min(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_PARAGRAPH:
                    print docID, ": Fails at MIN_SENTENCES_IN_PARAGRAPH (", count, "/", MIN_SENTENCES_IN_PARAGRAPH,")"
                    continue
                
    
            best_docs.append(docID)

        # Skip the bad documents
        best_docs = set(best_docs)
        new_docs = [doc for doc in self.documents if doc[0] in best_docs]
        return new_docs


    def get_samples(self):
        #PROCESS_MAX_FILES = 600
        PROCESS_MAX_FILES = 100
        #PROCESS_MAX_FILES = 50
        files_processed = 0
        for fil in os.listdir(self.WIKI_DOCS):
            self._create_structured_document(self.WIKI_DOCS + fil)
            files_processed += 1
            if files_processed > PROCESS_MAX_FILES:
                print ">>>> Breaking the process loop. Processed %d files" %(files_processed)
                break

        self.documents = self.filter_docs()
        samples = self.sample_creator.create_samples(self.documents)
        return samples



class SampleCreator:
    def __init__(self):
        self.queue = []
        self.samples = []
        #self.REQUIRED_CONSECUTIVE_PARAGRAPH = 2  # For the sample, each sample is (1 paragraph, split-end, 1 paragraph)

    def create_samples(self, documents):
        print "Creating Samples...."
        for (docID, sections) in document:
            # Iterate over section
            for section in sections:
                for paragraph in section:
                    self.queue += [((sentence, int(not count))) for count, sentence in enumerate(paragraph)]    # GroundTruth is 1 for the splitting sentence else 0
        self._process_queue()
        return self.samples
    

    def _process_queue(self):
        """ Unpack and create individual samples from the common long queue """
        if not len(self.queue):
            return
        
        for i in range(len(self.queue)):
            temp = self.queue[i: i+INPUT_VECTOR_LENGTH]
            if len(temp) != INPUT_VECTOR_LENGTH:
                continue
            self.samples.append(temp)


if __name__ == "__main__":
    data_handler = DataHandler()
    data_handler.get_samples()
    import pdb; pdb.set_trace()
