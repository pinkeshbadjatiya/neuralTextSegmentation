import re
import pdb
import nltk.data
import os
import codecs

from multiprocessing import Process, Lock, Queue
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


# Skip that condition if the value is -1
MIN_SENTENCES_IN_DOCUMENT = -1
MIN_SENTENCES_IN_SECTION = 1
MIN_SECTIONS = 2    # Exlcuding the 1st section
MIN_SENTENCES_IN_PARAGRAPH = 5      # Using the nltk tokenizer to get the approximate sentence count in a paragraph

INPUT_VECTOR_LENGTH = 10       # Similar to K as discussed with litton



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
        
        #self.PROCESS_MAX_FILES = -1
        #self.PROCESS_MAX_FILES = 5000
        self.PROCESS_MAX_FILES = 1500
        #self.PROCESS_MAX_FILES = 750
        #self.PROCESS_MAX_FILES = 50
        #self.PROCESS_MAX_FILES = 30
        #self.PROCESS_MAX_FILES = 10

        ################## VARIABLES ###################
        self.document_id_to_title = {}  # Map for the document ID's
        self.documents = []             # Collects "Raw" documents extracted from the dataset
                                        # Document = [section in sections] where, section = [paragraph in paragraphs] where paragraph = [line in lines] (Tokenized using a tokenizer)

        self.required_samples = []           # Samples which are actually split segments
        self.best_docs = []
        self._queue = []          # Handles the tokenised sentence in a paragraph, a temp variable


    def _create_structured_document_PARALLEL(self, files, _queue):
        return_documents = []
        for file_name in files:
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
                    #print "docID: %s" % (docID)
                elif _doc_end:
                    # Flush old data & reset
                    return_documents.append((docID, sections))
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
                    paragraph.append(self.sentence_tokenizer.tokenize(line))

        _queue.put(return_documents)


    def _create_structured_documents(self, filenameS):
        # Read all data in one Go
        all_data = []
        for file_name in filenameS:
            with open(file_name) as f:
                all_data.append(f.readlines())

        # Now process all data
        for data in all_data:
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
                    #print "docID: %s" % (docID)
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
                    paragraph.append(self.sentence_tokenizer.tokenize(line))



    def filter_docs(self):
        print "Filtering GOOD documents..."
        best_docs = []
        for (docID, sections) in self.documents:
            # Remove the 1st section as it might be very complex in structure
            sections = sections[1:]

            # Remove documents with less than 2 sections
            if MIN_SECTIONS != -1:
                if len(sections) <= MIN_SECTIONS:
                    #print docID, ": Fails at MIN_SECTIONS (", len(sections), "/", MIN_SECTIONS, ")"
                    continue

            sentence_counts = [[len(par) for par in section] for section in sections]

            # Remove documents that have less than MIN_SENTENCES_IN_DOCUMENT.
            if MIN_SENTENCES_IN_DOCUMENT != -1:
                count = sum([sum(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_DOCUMENT:
                    #print docID, ": Fails at MIN_SENTENCES_IN_DOCUMENT (", count, "/", MIN_SENTENCES_IN_DOCUMENT,")"
                    continue

            # Remove documents that have less than MIN_SENTENCES_IN_SECTION
            if MIN_SENTENCES_IN_SECTION != -1:
                count = min([sum(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_SECTION:
                    #print docID, ": Fails at MIN_SENTENCES_IN_SECTION (", count,"/", MIN_SENTENCES_IN_SECTION,")"
                    continue

            if MIN_SENTENCES_IN_PARAGRAPH != -1:
                count = min([min(section) for section in sentence_counts])
                if count < MIN_SENTENCES_IN_PARAGRAPH:
                    #print docID, ": Fails at MIN_SENTENCES_IN_PARAGRAPH (", count, "/", MIN_SENTENCES_IN_PARAGRAPH,")"
                    continue


            best_docs.append(docID)

        # Skip the bad documents
        best_docs = set(best_docs)
        new_docs = [doc for doc in self.documents if doc[0] in best_docs]
        return new_docs

    def get_sequence_samples_PARALLEL(self):
        """ Type2 samples, parallel
        """
        print "Going PARALLEL!"
        SAMPLE_TYPE = 2
        files = [self.WIKI_DOCS+fil for fil in os.listdir(self.WIKI_DOCS)]
        if self.PROCESS_MAX_FILES != -1:
            files = files[:self.PROCESS_MAX_FILES]
            print "NOTE: Processing %d files and breaking" %(self.PROCESS_MAX_FILES)
        else:
            print "NOTE: Processing a total of %d files" %(self.PROCESS_MAX_FILES)

        processes = []
        PARALLEL_PROCESSES = 5
        _out_queue = Queue()
        chunk = [files[i::PARALLEL_PROCESSES] for i in xrange(PARALLEL_PROCESSES) ]
        #assert(sum([len(chk) for chk in chunk]), len(files))

        procs = []
        for i in range(PARALLEL_PROCESSES):
            p = Process(
                    target=self._create_structured_document_PARALLEL,
                    args=(chunk[i],
                          _out_queue))
            procs.append(p)

        # Update the documents
        for i in range(PARALLEL_PROCESSES):
            self.documents += _out_queue.get()

        # Wait for all worker processes to finish
        for p in procs:
            p.join()
        print "Pool Ended!"
        logging.debug('Total documents: %d', len(self.documents))

        self.documents = self.filter_docs()
        sequence_samples = self.sample_creator.create_sequence_samples(self.documents)
        return SAMPLE_TYPE, sequence_samples

    def get_sequence_samples(self, sample_type):
        """ Type2 samples
        """
        assert sample_type in (2, 3)
        files = [self.WIKI_DOCS+fil for fil in os.listdir(self.WIKI_DOCS)]
        if self.PROCESS_MAX_FILES != -1:
            files = files[:self.PROCESS_MAX_FILES]
            print "NOTE: Processing %d files and breaking" %(self.PROCESS_MAX_FILES)
        else:
            print "NOTE: Processing a total of %d files" %(self.PROCESS_MAX_FILES)

        self._create_structured_documents(files)
        self.documents = self.filter_docs()
        sequence_samples = self.sample_creator.create_sequence_samples(self.documents)
        return sample_type, sequence_samples


    def get_samples(self):
        """ Type1 samples
        """
        SAMPLE_TYPE = 1
        files_processed = 0
        for fil in os.listdir(self.WIKI_DOCS):
            print "Processed file %s." %(fil)
            if self.PROCESS_MAX_FILES != -1 and files_processed >= self.PROCESS_MAX_FILES:
                print "NOTE: Breaking the process loop. Processed %d files" %(files_processed)
                break
            self._create_structured_documents([self.WIKI_DOCS + fil])
            files_processed += 1

        self.documents = self.filter_docs()
        samples = self.sample_creator.create_samples(self.documents)
        return SAMPLE_TYPE, samples



class SampleCreator:
    def __init__(self):
        self.queue = []
        self.samples = []
        #self.REQUIRED_CONSECUTIVE_PARAGRAPH = 2  # For the sample, each sample is (1 paragraph, split-end, 1 paragraph)

    def create_sequence_samples(self, document):
        print "Creating Samples for each document (Document is a sequence of sentences) (NOT separating as paragraph splitting)...."
        self.samples = []
        for (docID, sections) in document:
            # Iterate over section
            queue = []
            for section in sections:
                for paragraph in section:
                    queue += [(codecs.encode(sentence, "utf-8"), int(not count)) for count, sentence in enumerate(paragraph)]    # GroundTruth is 1 for the splitting sentence else 0
                    ###########################################################
                    #####     Encoding is shit! (but it is nice :D)     #######
                    #####     It is more complicated then you think     #######
                    ###########################################################
            self.samples.append(queue)
        return self.samples

    def create_samples(self, document):
        print "Creating Samples...."
        for (docID, sections) in document:
            # Iterate over section
            for section in sections:
                for paragraph in section:
                    self.queue += [(codecs.encode(sentence, "utf-8"), int(not count)) for count, sentence in enumerate(paragraph)]    # GroundTruth is 1 for the splitting sentence else 0
                    ###########################################################
                    #####     Encoding is shit! (but it is nice :D)     #######
                    #####     It is more complicated then you think     #######
                    ###########################################################
        self._process_queue()
        return self.samples


    def _process_queue(self):
        """ Unpack and create individual samples from the common long queue """
        if not len(self.queue):
            return

        for i in range(len(self.queue)):
            temp = self.queue[i: i+INPUT_VECTOR_LENGTH]
            if len(temp) != INPUT_VECTOR_LENGTH:
                break
            self.samples.append(temp)


if __name__ == "__main__":
    data_handler = DataHandler()
    #data_handler.get_sequence_samples()
    data_handler.get_samples()
    #import pdb; pdb.set_trace()
