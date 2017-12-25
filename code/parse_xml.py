import re
import pdb

import nltk
nltk.data.path.append("../nltk_data")

import os
import codecs

from multiprocessing import Process, Lock, Queue
from helper import compute_avg_seg_len
#import logging
#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-10s) %(message)s',
#                    )


# Skip that condition if the value is -1
MIN_SENTENCES_IN_DOCUMENT = 21  # currently equal to 2*context + 1
MIN_SENTENCES_IN_SECTION = 1
MIN_SECTIONS = 2    # Exlcuding the 1st section
MIN_SENTENCES_IN_PARAGRAPH = -1      # Using the nltk tokenizer to get the approximate sentence count in a paragraph

INPUT_VECTOR_LENGTH = 10       # Similar to K as discussed with litton, not required if fetching document as a single sequence


MIN_TRAIN_AVG_SEGMENT_LENGTH = 20


class DataHandler:
    def __init__(self):
        ################ Constants #################
        self.REGEX_heading = re.compile(r'<(h[0-9])>(.*)\.</\1>')      # <h2>heading.</h>
        self.REGEX_document_start = re.compile(r'<doc *(id="([0-9]{1,})")? *(url=".*")? *(title=".*")?>')
        self.REGEX_document_end = re.compile(r'</doc>')

        #self.WIKI_DOCS = "/home/pinkesh/DATASETS/WIKIPEDIA_DATASET/extracted_WIKIPEDIA/"
        self.WIKI_DOCS = "/home/pinkesh.badjatiya/TopicSegmentation_ECIR/extracted_WIKIPEDIA/"
        if self.WIKI_DOCS[-1]!="/":
            raise Exception("Check the directory name")

        self.sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sample_creator = SampleCreator()

        self.PROCESS_MAX_FILES = -1
        #self.PROCESS_MAX_FILES = 5000
        #self.PROCESS_MAX_FILES = 4400
        #self.PROCESS_MAX_FILES = 4000
        #self.PROCESS_MAX_FILES = 1500
        #self.PROCESS_MAX_FILES = 900
        #self.PROCESS_MAX_FILES = 800
        #self.PROCESS_MAX_FILES = 400
        self.PROCESS_MAX_FILES = 200
        #self.PROCESS_MAX_FILES = 100
        #self.PROCESS_MAX_FILES = 50
        #self.PROCESS_MAX_FILES = 20
        #self.PROCESS_MAX_FILES = 10
        #self.PROCESS_MAX_FILES = 5
        #self.PROCESS_MAX_FILES = 1

        ################## VARIABLES ###################
        self.document_id_to_title = {}  # Map for the document ID's
        self.documents = []             # Collects "Raw" documents extracted from the dataset
                                        # Document = [section in sections] where, section = [paragraph in paragraphs] where paragraph = [line in lines] (Tokenized using a tokenizer)

        self.required_samples = []           # Samples which are actually split segments
        self.best_docs = []
        self._queue = []          # Handles the tokenised sentence in a paragraph, a temp variable

        # Print var status
        print "==== MIN_SENTENCES_IN_DOCUMENT: %d, MIN_SENTENCES_IN_SECTION: %d, MIN_SECTIONS(excluding 1st): %d, MIN_SENTENCES_IN_PARAGRAPH: %d ===" %(MIN_SENTENCES_IN_DOCUMENT, MIN_SENTENCES_IN_SECTION, MIN_SECTIONS, MIN_SENTENCES_IN_PARAGRAPH)

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
        print "Filtering GOOD documents using MIN_(SENT/DOC/SEC..) filters ...."
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


    def get_sequence_samples(self, sample_type):
        """ Type2 samples
        """
        assert sample_type in (2, 3)
        files = [self.WIKI_DOCS+fil for fil in os.listdir(self.WIKI_DOCS)]
        if self.PROCESS_MAX_FILES != -1:
            files = files[:self.PROCESS_MAX_FILES]
            print "NOTE: Processing %d files and breaking" %(self.PROCESS_MAX_FILES)
        else:
            self.PROCESS_MAX_FILES = len(files)
            print "NOTE: Processing a total of %d files" %(self.PROCESS_MAX_FILES)

        self._create_structured_documents(files)
        self.documents = self.filter_docs()
        sequence_samples = self.sample_creator.create_sequence_samples(self.documents)
        sequence_samples = self.sample_creator.filter_low_segment_documents(sequence_samples)
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

        #self.SPLIT_TYPE = "paragraph"             # should be in [section, paragraph]
        self.SPLIT_TYPE = "section"             # should be in [section, paragraph]
        assert self.SPLIT_TYPE in ["section", "paragraph"]
        print "#### Using split type:", self.SPLIT_TYPE

        #self.REQUIRED_CONSECUTIVE_PARAGRAPH = 2  # For the sample, each sample is (1 paragraph, split-end, 1 paragraph)

    def _get_groundtruth(self, section_idx, paragraph_idx, sentence_idx):
        if self.SPLIT_TYPE == "section":        # if the sentence is from the 1st paragraph and its index is 0 then its groundtruth is 1.
            if (not paragraph_idx) and (not sentence_idx):
                return True
        elif self.SPLIT_TYPE == "paragraph":    # if the sentence is the 1st sentence of the paragraph
            if not sentence_idx:
                return True
        else:
            print ">>>>>>>>>>>>> INVALID SPLIT TYPE <<<<<<<<<<"

        # Not a split point
        return False
            
    def filter_low_segment_documents(self, documents):
        new_documents = []
        skipped = 0
        for sample in documents:
            sentences, groundTruths = zip(*sample)
            avg_seg = compute_avg_seg_len(groundTruths)
            if avg_seg >= MIN_TRAIN_AVG_SEGMENT_LENGTH:
                new_documents.append(sample)
            else:
                skipped += 1
        #    pdb.set_trace()
        print "Skipped %d documents due to MIN_TRAIN_AVG_SEGMENT_LENGTH=%d" %(skipped, MIN_TRAIN_AVG_SEGMENT_LENGTH)
        return new_documents

    def create_sequence_samples(self, document):
        print "Creating Samples for each document (Document is a sequence of sentences) (NOT separating as paragraph splitting)...."
        self.samples = []
        for (docID, sections) in document:
            # Iterate over section
            queue = []
            for sec_idx_in_doc, section in enumerate(sections):
                for par_idx_in_sec, paragraph in enumerate(section):
                    for sent_idx_in_par, sentence in enumerate(paragraph):
                        queue.append((codecs.encode(sentence, "utf-8"), self._get_groundtruth(sec_idx_in_doc, par_idx_in_sec, sent_idx_in_par)))
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
            for sec_idx_in_doc, section in enumerate(sections):
                for par_idx_in_sec, paragraph in enumerate(section):
                    for sent_idx_in_par, sentence in enumerate(paragraph):
                        self.queue.append((codecs.encode(sentence, "utf-8"), self._get_groundtruth(sec_idx_in_doc, par_idx_in_sec, sent_idx_in_par)))
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
