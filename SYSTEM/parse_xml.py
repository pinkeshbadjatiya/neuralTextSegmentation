import re
import pdb
import nltk.data
import os

REGEX_heading = re.compile(r'<(h[0-9])>(.*)\.</\1>')      # <h2>heading.</h>
REGEX_document_start = re.compile(r'<doc *(id="([0-9]{1,})")? *(url=".*")? *(title=".*")?>')
REGEX_document_end = re.compile(r'</doc>')

WIKI_DOCS = "/home/pinkesh/DATASETS/WIKIPEDIA_DATASET/extracted_WIKIPEDIA/"
if WIKI_DOCS[-1]!="/":
    raise Exception("Check the directory name")

# Skip that condition if the value is -1
MIN_SENTENCES_IN_DOCUMENT = -1
MIN_SENTENCES_IN_SECTION = 1
MIN_SECTIONS = 2    # Exlcuding the 1st section

MIN_SENTENCES_IN_PARAGRAPH = 5      # Using the nltk tokenizer to get the approximate sentence count in a paragraph
REQUIRED_CONSECUTIVE_PARAGRAPH = 2  # For the sample, each sample is (1 paragraph, split-end, 1 paragraph)


document_id_to_title = {}
documents = []


def create_structured_document(file_name):
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
        _doc_start, _doc_end, _heading = REGEX_document_start.match(line), REGEX_document_end.match(line), REGEX_heading.match(line)
        if _doc_start:
            if len(paragraph) > 0 or len(sections) > 0 or (docID is not None) or (start_line is not None):
                print "SERIOUS PROBLEM !!"
            paragraph, sections, docID, start_line = [], [], _doc_start.group(2), lno
            print "docID: %s" % (docID)
        elif _doc_end:
            # Flush old data & reset
            documents.append((docID, sections))
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
            paragraph.append(line)


required_samples = []           # Samples which are actually split segments
required_samples_NEG = []       # Samples which are not split segments
best_docs = []
queue = []          # Handles the tokenised sentence in a paragraph


def process_sample(invalid_paragraph_encountered=False):
    global queue
    if not len(queue):
        return

    # For NEGATIVE samples
    # SPLIT it into 2 if twice the length of the paragraph 
    if len(queue[-1]) >= 2*MIN_SENTENCES_IN_PARAGRAPH:
        print "########## Found a NEGATIVE split point with len: %d" %(len(queue[-1]))
        leng = len(queue[-1])
        required_samples_NEG.append([queue[-1][:leng/2], queue[-1][leng/2:]])

    # For POSITIVE samples
    if len(queue) >= REQUIRED_CONSECUTIVE_PARAGRAPH:
        required_samples.append(queue)
        queue = queue[1:]
        return
    else:
        pass
    if invalid_paragraph_encountered:
        queue = []

def get_best_documentIDs():
    global queue
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    for (docID, sections) in documents:
        # Remove the 1st section as it might be very complex in structure
        sections = sections[1:]

        # Remove documents with less than 2 sections
        if MIN_SECTIONS != -1:
            if len(sections) <= MIN_SECTIONS:
                print docID, ": Fails at MIN_SECTIONS (", len(sections),"/",MIN_SECTIONS,")"
                process_sample(invalid_paragraph_encountered=True)
                continue

        sentence_counts = [[len(sentence_tokenizer.tokenize(par)) for par in section] for section in sections]

        # Remove documents that have less than MIN_SENTENCES_IN_DOCUMENT.
        if MIN_SENTENCES_IN_DOCUMENT != -1:
            count = sum([sum(section) for section in sentence_counts])
            if count < MIN_SENTENCES_IN_DOCUMENT:
                print docID, ": Fails at MIN_SENTENCES_IN_DOCUMENT (", count,"/",MIN_SENTENCES_IN_DOCUMENT,")"
                process_sample(invalid_paragraph_encountered=True)
                continue

        # Remove documents that have less than MIN_SENTENCES_IN_SECTION
        if MIN_SENTENCES_IN_SECTION != -1:
            count = min([sum(section) for section in sentence_counts])
            if count < MIN_SENTENCES_IN_SECTION:
                print docID, ": Fails at MIN_SENTENCES_IN_SECTION (", count,"/",MIN_SENTENCES_IN_SECTION,")"
                process_sample(invalid_paragraph_encountered=True)
                continue

        # Iterate over section to classify the section as good or bad
        for section in sections:
            for paragraph in section:
                # Length of paragraph is wrong as the sentences are merges and need to be split using nltk sentence tokeniser
                paragraph = sentence_tokenizer.tokenize(paragraph)
                le = len(paragraph)
                if le < MIN_SENTENCES_IN_PARAGRAPH:
                    process_sample(invalid_paragraph_encountered=True)
                    continue

                # Make the sentences in the paragraph equal to MIN_SENTENCES_IN_PARAGRAPH
                #remove = le - MIN_SENTENCES_IN_PARAGRAPH
                #if remove:
                #    paragraph = paragraph[remove/2:-remove+remove/2]
                queue.append(paragraph)
                process_sample(invalid_paragraph_encountered=False)

        best_docs.append(docID)

    return best_docs

def get_samples():
    PROCESS_MAX_FILES = 600
    files_processed = 0
    for fil in os.listdir(WIKI_DOCS):
        create_structured_document(WIKI_DOCS + fil)
        files_processed += 1
        if files_processed > PROCESS_MAX_FILES:
            print ">>>> Breaking the process loop. Processed %d files" %(files_processed)
            break
    #create_structured_document("./AAs/AA_wiki_01")
    best_docs = get_best_documentIDs()
    print "Total best documents:", len(best_docs)
    return required_samples, required_samples_NEG


if __name__ == "__main__":
    get_samples()
    import pdb; pdb.set_trace()
