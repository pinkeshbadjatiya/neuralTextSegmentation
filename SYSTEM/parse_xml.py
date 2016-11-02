import re
import nltk.data

REGEX_heading = re.compile(r'<(h[0-9])>(.*)\.</\1>')      # <h2>heading.</h>
REGEX_document_start = re.compile(r'<doc *(id="([0-9]{1,})")? *(url=".*")? *(title=".*")?>')
REGEX_document_end = re.compile(r'</doc>')


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

    sections = []   # sections = [paragraph in paragraphs], where paragraph = [line in lines]
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


required_samples = []
best_docs = []
queue = []


def process_sample(invalid_paragraph_encountered=False):
    global queue
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
                if len(sentence_tokenizer.tokenize(paragraph)) < MIN_SENTENCES_IN_PARAGRAPH:
                    process_sample(invalid_paragraph_encountered=True)
                    continue
                queue.append(paragraph)
                process_sample(invalid_paragraph_encountered=False)

        best_docs.append(docID)

    return best_docs

if __name__ == "__main__":
    create_structured_document("./AAs/AA_wiki_01")
    best_docs = get_best_documentIDs()
    print "Total best documents:", len(best_docs)
    import pdb; pdb.set_trace()
