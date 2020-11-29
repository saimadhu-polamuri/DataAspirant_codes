"""
===============================================
Objective: Implementing TF-IDF in python
Author: Nikhil Kumar Jha
Blog: https://dataaspirant.com
Date: 2020-11-28
===============================================
"""

class TfIdf:
    def __init__(self, corpus_filename = None, stopword_filename = None,
               DEFAULT_IDF = 1.5):
        """
           Initialize the idf dictionary.

           Reads the idf dictionary from corpus file, in the format of:
             # of total documents
             term: # of documents containing the term

           If a stopword file is specified, reads the stopword list from it, in
           the format of one stopword per line.

           The DEFAULT_IDF value is returned when a query term not found
        """

        self.num_docs = 0
        self.term_num_docs = {}     # term : num_docs_containing_term
        self.stopwords = set([])
        self.idf_default = DEFAULT_IDF

        if corpus_filename:
            self.merge_corpus_document(corpus_filename)

        if stopword_filename:
          stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
          self.stopwords = set([line.strip() for line in stopword_file])

    def merge_corpus_document(self, corpus_filename):
        """
            Slurp in a corpus document, adding it to the existing corpus model
        """
        corpus_file = codecs.open(corpus_filename, "r", encoding='utf-8')

        ## Load number of documents.
        line = corpus_file.readline()
        self.num_docs += int(line.strip())

        ## Reads "term:frequency" from each subsequent line in the file.
        for line in corpus_file:
          tokens = line.rsplit(":",1)
          term = tokens[0].strip()
          try:
              frequency = int(tokens[1].strip())
          except IndexError, err:
              if line in ("","\t"):
                  #catch blank lines
                  print "line is blank"
                  continue
              else:
                  raise
          if self.term_num_docs.has_key(term):
            self.term_num_docs[term] += frequency
          else:
            self.term_num_docs[term] = frequency


    def add_input_document(self, input):
        """
            Add terms in the specified document to the idf dictionary.
        """
        self.num_docs += 1
        words = set(self.get_tokens(input))
        for word in words:
          if word in self.term_num_docs:
            self.term_num_docs[word] += 1
          else:
            self.term_num_docs[word] = 1

    def save_corpus_to_file(self, idf_filename, stopword_filename,
                          STOPWORD_PERCENTAGE_THRESHOLD = 0.01):
        """
            Save the idf dictionary and stopword list to the specified file.
        """
        output_file = codecs.open(idf_filename, "w", encoding='utf-8')

        output_file.write(str(self.num_docs) + "\n")
        for term, num_docs in self.term_num_docs.items():
          output_file.write(term + ": " + str(num_docs) + "\n")

        sorted_terms = sorted(self.term_num_docs.items(), key=itemgetter(1),
                              reverse=True)
        stopword_file = open(stopword_filename, "w")
        for term, num_docs in sorted_terms:
          if num_docs < STOPWORD_PERCENTAGE_THRESHOLD * self.num_docs:
            break

          stopword_file.write(term + "\n")

    def get_idf(self, term):
        """
            Retrieve the IDF for the specified term.

            This is computed by taking the logarithm of ((number of
            documents in corpus) divided by (number of documents
            containing this term) ).
         """
        if term in self.stopwords:
          return 0

        if not term in self.term_num_docs:
          return self.idf_default

        return math.log(float(1 + self.get_num_docs()) /
          (1 + self.term_num_docs[term]))


    def get_doc_keywords(self, curr_doc):
        """
            Retrieve terms and corresponding tf-idf for the specified document.

            The returned terms are ordered by decreasing tf-idf.
        """
        tfidf = {}
        tokens = self.get_tokens(curr_doc)
        tokens_set = set(tokens)
        for word in tokens_set:
          mytf = float(tokens.count(word)) / len(tokens_set)
          myidf = self.get_idf(word)
          tfidf[word] = mytf * myidf
          print(
          f'{word}\t::\tTF = {mytf}\t::\tIDF = {myidf}\t::\tTF-IDF ={tfidf[word]}')


        return sorted(tfidf.items(), key=itemgetter(1), reverse=True)


## For words not in corpus
idf_default = 1.5

## Create your TF-IDF model instance
my_tfidf = TfIdf("tfidf_testcorpus.txt",
"tfidf_teststopwords.txt", DEFAULT_IDF = idf_default)

vocab = []

print('**Reading Corpus file**')
with open('tfidf_testcorpus.txt', 'r') as ff:
    corp = ff.read()
    print(corp)
    corp_split = corp.split(':')
    for i in range(len(corp_split)-1):
        vocab.append(corp_split[i].split('\n')[1])

print('\n**Stopwords**')
with open('tfidf_teststopwords.txt', 'r') as ff:
    print(ff.read())

## Calculate and store the TF-IDF values
tfidf_vals = [my_tfidf.get_doc_keywords(word)[0] for word in vocab]

## dataaspirant-tf-idf-example-calculation.py
