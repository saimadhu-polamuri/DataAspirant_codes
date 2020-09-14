"""
===============================================
Objective: List of 20+ text preprocessing techniques implementation in python
Author: Sharmila.Polamuri
Blog: https://dataaspirant.com
Date: 2020-09-14
===============================================
"""


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of lower case conversion

def lower_case_convertion(text):
	"""
	Input :- string
	Output :- lowercase string
	"""
	lower_text = text.lower()
	return lower_text


ex_lowercase = "This is an example Sentence for LOWER case conversion"
lowercase_result = lower_case_convertion(ex_lowercase)
print(lowercase_result)

## Output:: this is an example sentence for lower case conversion

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# HTML tags removal Implementation using regex module

import re
def remove_html_tags(text):
	"""
	Return :- String without Html tags
	input :- String
	Output :- String
	"""
	html_pattern = r'<.*?>'
	without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
	return without_html

ex_htmltags = """ <body>
<div>
<h1>Hi, this is an example text with Html tags. </h1>
</div>
</body>
"""
htmltags_result = remove_html_tags(ex_htmltags)
print(f"Result :- \n {htmltags_result}")

## Output:: Hi, this is an example text with Html tags.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of Removing HTML tags using bs4 library

from bs4 import BeautifulSoup
def remove_html_tags_beautifulsoup(text):
	"""
	Return :- String without Html tags
	input :- String
	Output :- String
	"""
	parser = BeautifulSoup(text, "html.parser")
	without_html = parser.get_text(separator = " ")
	return without_html

ex_htmltags = """ <body>
<div>
<h1>Hi, this is an example text with Html tags. </h1>
</div>
</body>
"""
htmltags_result = remove_html_tags_beautifulsoup(ex_htmltags)
print(f"Result :- \n {htmltags_result}")

## Output:: Hi, this is an example text with Html tags.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of Removing URLs  using python regex

import re
def remove_urls(text):
	"""
	Return :- String without URLs
	input :- String
	Output :- String
	"""
	url_pattern = r'https?://\S+|www\.\S+'
	without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
	return without_urls

# example text which contain URLs in it
ex_urls = """
This is an example text for URLs like http://google.com & https://www.facebook.com/ etc.
"""

# calling removing_urls function with example text (ex_urls)
urls_result = remove_urls(ex_urls)
print(f"Result after removing URLs from text :- \n {urls_result}")


## Output:: This is an example text for URLs like   &   etc.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of Removing numbers  using python regex

import re
def remove_numbers(text):
	"""
	Return :- String without numbers
	input :- String
	Output :- String
	"""
	number_pattern = r'\d+'
	without_number = re.sub(pattern=number_pattern,
 repl=" ", string=text)
	return without_number

# example text which contain numbers in it
ex_numbers = """
This is an example sentence for removing numbers like 1, 5,7, 4 ,77 etc.
"""
# calling remove_numbers function with example text (ex_numbers)
numbers_result = remove_numbers(ex_numbers)
print(f"Result after removing number from text :- \n {numbers_result}")

## Output:: This is an example sentence for removing numbers like  ,  , ,   ,  etc.
""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

# Implementation of Converting numbers to words using python num2words library

from num2words import num2words

# function to convert numbers to words
def num_to_words(text):
	"""
	Return :- text which have all numbers or integers in the form of words
	Input :- string
	Output :- string
	"""
	# splitting text into words with space
	after_spliting = text.split()

	for index in range(len(after_spliting)):
		if after_spliting[index].isdigit():
			after_spliting[index] = num2words(after_spliting[index])

    # joining list into string with space
	numbers_to_words = ' '.join(after_spliting)
	return numbers_to_words

# example text which contain numbers in it
ex_numbers = """
This is an example sentence for converting numbers to words like 1 to one, 5 to five, 74 to seventy-four, etc.
"""
# calling remove_numbers function with example text (ex_numbers)
numners_result = num_to_words(ex_numbers)
print(f"Result after converting numbers to its words from text :- \n {numners_result}")

## Output:: This is an example sentence for converting numbers to words like one to one, five to five, seventy-four to seventy-four, etc.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of spelling correction using python pyspellchecker library

from spellchecker import SpellChecker

spell_corrector = SpellChecker()

# spelling correction using spellchecker
def spell_correction(text):
	"""
	Return :- text which have correct spelling words
	Input :- string
	Output :- string
	"""
	# initialize empty list to save correct spell words
	correct_words = []
	# extract spelling incorrect words by using unknown function of spellchecker
	misSpelled_words = spell_corrector.unknown(text.split())

	for each_word in text.split():
		if each_word in misSpelled_words:
			right_word = spell_corrector.correction(each_word)
			correct_words.append(right_word)
		else:
			correct_words.append(each_word)

	# joining correct_words list into single string
	correct_spelling = ' '.join(correct_words)
	return correct_spelling

#example text with mis spelling words
ex_misSpell_words = """
This is an example sentence for spell corecton
"""
spell_result = spell_correction(ex_misSpell_words)
print(f"Result after spell checking :- \n{spell_result}")

## Output:: This is an example sentence for spell correction

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

# Implementation of spelling correction using python autocorrect library

from autocorrect import Speller
from nltk import word_tokenize

# spelling correction using spellchecker
def spell_autocorrect(text):
	"""
	Return :- text which have correct spelling words
	Input :- string
	Output :- string
	"""
	correct_spell_words = []

	# initialize Speller object for english language with 'en'
	spell_corrector = Speller(lang='en')
	for word in word_tokenize(text):
		# correct spell word
		correct_word = spell_corrector(word)
		correct_spell_words.append(correct_word)

	correct_spelling = ' '.join(correct_spell_words)
	return correct_spelling

# another example text with misSpelling words
ex_misSpell_words_1 = """
This is anoter exapl for spell correction
"""
spell_result = spell_autocorrect(ex_misSpell_words_1)
print(f"Result :- \n{spell_result}")

## Output:: This is another example for spell correction

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of accented text to ASCII converter in python

import unidecode

def accented_to_ascii(text):
	"""
	Return :- text after converting accented characters
	Input :- string
	Output :- string
	"""
	# apply unidecode function on text to convert
	# accented characters to ASCII values
	text = unidecode.unidecode(text)
	return text

# example text with accented characters
ex_accented = """
This is an example text with accented characters like dèèp lèarning ánd cömputer vísíön etc.
"""
accented_result = accented_to_ascii(ex_accented)
print(f"Result after converting accented characters to their ASCII values \n{accented_result}")

## Output:: This is an example text with accented characters like deep learning and computer vision etc.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Converting chat conversion words to normal words

def short_to_original(text):
	"""
	Return :- text after converting short_form words to original
	Input :- string
	Output :- string
	"""
	new_text = []
	for w in text.split():
		if w.upper() in chat_words_list:
			new_text.append(chat_words_map_dict[w.upper()])
		else:
			new_text.append(w)
	return " ".join(new_text)


# example text for chat conversation short-form words
ex_chat = """
omg this is an example text for chat conversation.
"""
# open short_form file and then read sentences from text file using read())
short_form_list = open('short_forms.txt', 'r')
chat_words_str = short_form_list.read()

chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
	if line != "":
		cw = line.split("=")[0]
		cw_expanded = line.split("=")[1]
		chat_words_list.append(cw)
		chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)


# calling function
chat_result = short_to_original(ex_chat)
print(f"Result {chat_result}")

## Output :: Result oh my god this is an example text for chat conversation.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
## Implementation of expanding contractions

from contraction import CONTRACTION_MAP

def expand_contractions(contraction):
	# take matching contraction in the text
	match = contraction.group(0)
	# first char from matching contraction (D for Doesn't)
	first_char = match[0]
	if contraction_mapping.get(match):
		expanded_contraction = contraction_mapping.get(match)
	else:
		expanded_contraction = contraction_mapping.get(match.lower())
	expanded_contraction = first_char+expanded_contraction[1:]

	return expanded_contraction

# expending contractions
contraction_mapping = CONTRACTION_MAP
# take all key values from contraction_mapping
contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)

# example text with contractions
ex_contractions = """
Sometimes our mind doesn't work properly.
"""
# substitute result of function in the text
expanded_text = contractions_pattern.sub(expand_contractions, ex_contractions)
# replacing apostrophe with empty string (to remove apostrophe)
expanded_text = re.sub("'", "", expanded_text)
print(f"Result :- \n{expanded_text}")

## Output:: Sometimes our mind does not work properly.

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

# Implementation of Stemming using PorterStemming from nltk library

from nltk.stem import PorterStemmer

def porter_stemmer(text):
	"""
	Result :- string after stemming
	Input :- String
	Output :- String
	"""
	# word tokenization
	tokens = word_tokenize(text)

	for index in range(len(tokens)):
		# stem word to each word
		stem_word = stemmer.stem(tokens[index])
		# update tokens list with stem word
		tokens[index] = stem_word

	# join list with space separator as string
	return ' '.join(tokens)

# initialize porter stemmer object
stemmer = PorterStemmer()
# example text for stemming technique
ex_stem = "Programers program with programing languages"
stem_result = porter_stemmer(ex_stem)
print(f"Result after stemming technique :- \n{stem_result}")

## Output:: program program with program languag

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
## Implementation of lemmatization using nltk

from nltk.stem import WordNetLemmatizer

def lemmatization(text):
	"""
	Result :- string after stemming
	Input :- String
	Output :- String
	"""
	# word tokenization
	tokens = word_tokenize(text)

	for index in range(len(tokens)):
		# lemma word
		lemma_word = lemma.lemmatize(tokens[index])
		tokens[index] = lemma_word

	return ' '.join(tokens)

# initialize lemmatizer object
lemma = WordNetLemmatizer()
# example text for lemmatization
ex_lemma = """
Programers program with programing languages
"""
lemma_result = lemmatization(ex_lemma)
print(f"Result of lemmatization \n{lemma_result}")

## Output:: Programers program with programing language

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

# Implementation of emoji removing

def remove_emojis(text):
	"""
	Result :- string without any emojis in it
	Input :- String
	Output :- String
	"""
	emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

	without_emoji = emoji_pattern.sub(r'',text)
	return without_emoji


# example text for emoji removing technique
ex_emoji = """
This is a test 😻 👍🏿
"""
# calling function
emoji_result = remove_emojis(ex_emoji)
print(f"Result text after removing emojis :- \n{emoji_result}")

## Output :: This is a test

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of removing of emoticons

from emoticons_list import EMOTICONS
def remove_emoticons(text):
	"""
	Return :- string after removing emoticons
	Input :- string
	Output :- string
	"""
	emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

	without_emoticons = emoticon_pattern.sub(r'',text)
	return without_emoticons

# example sentence for removing emoticons
ex_emoticons = """
Hello this is a sentence with these 2 emoticons :-) & :-)
"""
emoticons_result = remove_emoticons(ex_emoticons)
print(f"After removing emoticons :- \n{emoticons_result}")

## Ouput:: Hello this is a sentence with these 2 emoticons  &

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of converting emoji to words using python

from emoticons_list import EMO_UNICODE

def emoji_words(text):
	for emot in UNICODE_EMO:
		emoji_pattern = r'('+emot+')'
		# replace
		emoji_words = UNICODE_EMO[emot]
		replace_text = emoji_words.replace(",","")
		replace_text = replace_text.replace(":","")
		replace_text_list = replace_text.split()
		emoji_name = '_'.join(replace_text_list)
		text = re.sub(emoji_pattern, emoji_name, text)
	return text


# convert emo_unicode to unicode_emo
UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}
# example text for converting emojis to words
ex_emoji = """
This is a test 😻 👍🏿
"""

emoji_result = emoji_words(ex_emoji)
print(f"Result after converting emojis to corresponding words :- \n{emoji_result}")

## Output:: This is a test smiling_cat_face_with_heart-eyes thumbs_updark_skin_tone

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

# Implementation of converting emoticons to words

from emoticons_list import EMOTICONS

def emoticons_words(text):
	for emot in EMOTICONS:
		emoticon_pattern = r'('+emot+')'
		# replace
		emoticon_words = EMOTICONS[emot]
		replace_text = emoticon_words.replace(",","")
		replace_text = replace_text.replace(":","")
		replace_text_list = replace_text.split()
		emoticon_name = '_'.join(replace_text_list)
		text = re.sub(emoticon_pattern, emoticon_name, text)
	return text


# example sentence for converting  emoticons to words
ex_emoticons = """
Hello this is a sentence with these 2 emoticons :-) & :-)
"""
emoticons_result = emoticons_words(ex_emoticons)
print(f"After converting emoticons to words :- \n{emoticons_result}")

## Output:: Hello this is a sentence with these 2 emoticons Happy_face_smiley & Happy_face_smiley
""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of removing punctuations using string library

from string import punctuation

def remove_punctuation(text):
	"""
	Return :- String after removing punctuations
	Input :- String
	Output :- String
	"""
	return text.translate(str.maketrans('', '', punctuation))


# example text for removing punctuations
ex_punct = """
this is an example text for punctuations like .?/*
"""
punct_result = remove_punctuation(ex_punct)
print(f"Result after removing punctuations :- \n{punct_result}")

## Output:: this is an example text for punctuations like
""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of removing stopwords using all stop words from nltk, spacy, gensim

from nltk.corpus import stopwords
import spacy
import gensim


def remove_stopwords(text):
	"""
	Return :- String after removing stopwords
	Input :- String
	Output :- String
	"""
	text_without_sw = []
	# tokenization
	text_tokens = word_tokenize(text)
	for word in text_tokens:
		# checking word is stopword or not
		if word not in all_stopwords:
			text_without_sw.append(word)

	# joining all tokens after removing stop words
	without_sw = ' '.join(text_without_sw)
	return without_sw


# list of stopwords from nltk
stopwords_nltk = list(stopwords.words('english'))
sp = spacy.load('en_core_web_sm')
# list of stopwords from spacy
stopwords_spacy = list(sp.Defaults.stop_words)
# list of stopwords from gensim
stopwords_gensim = list(gensim.parsing.preprocessing.STOPWORDS)

# unique stopwords from all stopwords
all_stopwords = []
all_stopwords.extend(stopwords_nltk)
all_stopwords.extend(stopwords_spacy)
all_stopwords.extend(stopwords_gensim)
# all unique stop words
all_stopwords = list(set(all_stopwords))
print(f"Total number of Stopwords :- {len(all_stopwords)}")

# example text for stop words removing
ex_sw = """
this is an example text for stopwords such as a, an, the etc.
"""
sw_result = remove_stopwords(ex_sw)

print(f"Result after removing stopwords :- \n{sw_result}")

## Output:: example text stopwords , , .

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
## Implementation of frequent words removing

from collections import Counter

def freq_words(text):
	"""
	Return :- Most frequent words
	Input :- string
	Output :-
	"""
	# tokenization
	tokens = word_tokenize(text)
	for word in tokens:
		counter[word]= +1

	FrequentWords = []
	# take top 10 frequent words
	for (word, word_count) in counter.most_common(10):
		FrequentWords.append(word)

	return FrequentWords

def remove_fw(text, FrequentWords):
	"""
	Return :- String after removing frequent words
	Input :- String
	Output :- String
	"""

	tokens = word_tokenize(text)
	without_fw = []
	for word in tokens:
		if word not in FrequentWords:
			without_fw.append(word)

	without_fw = ' '.join(without_fw)
	return without_fw


# initiate object for counter
counter = Counter()
# some random text on machine learning
ex_fw = """
Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.
For example, one kind of algorithm is a classification algorithm. It can put data into different groups. The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code. It's the same algorithm but it's fed different training data so it comes up with different classification logic.
Two kinds of Machine Learning Algorithms
You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning. The difference is simple, but really important.
Supervised Learning
Let's say you are a real estate agent. Your business is growing, so you hire a bunch of new trainee agents to help you out. But there's a problem -- you can glance at a house and have a pretty good idea of what a house is worth, but your trainees don't have your experience so they don't know how to price their houses.
To help your trainees (and maybe free yourself up for a vacation), you decide to write a little app that can estimate the value of a house in your area based on it's size, neighborhood, etc, and what similar houses have sold for.
So you write down every time someone sells a house in your city for 3 months. For each house, you write down a bunch of details -- number of bedrooms, size in square feet, neighborhood, etc. But most importantly, you write down the final sale price:
This is called supervised learning. You knew how much each house sold for, so in other words, you knew the answer to the problem and could work backwards from there to figure out the logic.
To build your app, you feed your training data about each house into your machine learning algorithm. The algorithm is trying to figure out what kind of math needs to be done to make the numbers work out.
This kind of like having the answer key to a math test with all the arithmetic symbols erased:
"""

# calling count_fw to calculate frequent words
FrequentWords = freq_words(ex_fw)
print(f"Top 10 Frequent Words from our example text :- \n{FrequentWords}")


# calling remove_fw to remove frequent words from example text
fw_result = remove_fw(ex_fw, FrequentWords)

print(f"Result after removing frequent words :-\n{fw_result}")


"""
Output::

can tell you something interesting about a set of data without you having to write any custom code specific to problem . Instead of writing code , you feed data to algorithm and it builds its own logic based on data . For example , one kind of algorithm a classification algorithm . It can put data into different groups . The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code . It ' s same algorithm but it ' s fed different training data so it comes up with different classification logic . Two kinds of Learning Algorithms You can think of machine as falling into one of two main categories -- supervised and unsupervised . The difference simple , but really important . Supervised Learning Let ' s say you a real estate agent . Your business growing , so you hire a bunch of new trainee agents to help you out . But ' s a problem -- you can glance at a house and have a pretty good of what a house worth , but your trainees don ' t have your experience so they don ' t know how to price their houses . To help your trainees ( and maybe free yourself up for a vacation ) , you decide to write a little app can estimate value of a house in your area based on it ' s size , neighborhood , etc , and what similar houses have sold for . So you write down every time someone sells a house in your city for 3 months . For each house , you write down a bunch of details -- number of bedrooms , size in square feet , neighborhood , etc . But most importantly , you write down final sale price : This called supervised . You knew how much each house sold for , so in other words , you knew answer to problem and could work backwards from to figure out logic . To build your app , you feed your training data about each house into your machine algorithm . The algorithm trying to figure out what kind of math needs to be done to make numbers work out . This kind of like having answer key to a math test with all arithmetic symbols erased
"""

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Implementation of rare words removing

from collections import Counter

def rare_words(text):
	"""
	Return :- Most Rare words
	Input :- string
	Output :- list of rare words
	"""
	# tokenization
	tokens = word_tokenize(text)
	for word in tokens:
		counter[word]= +1

	RareWords = []
	number_rare_words = 10
	# take top 10 frequent words
	frequentWords = counter.most_common()
	for (word, word_count) in frequentWords[:-number_rare_words:-1]:
		RareWords.append(word)

	return RareWords

def remove_rw(text, RareWords):
	"""
	Return :- String after removing frequent words
	Input :- String
	Output :- String
	"""

	tokens = word_tokenize(text)
	without_rw = []
	for word in tokens:
		if word not in RareWords:
			without_rw.append(word)

	without_rw = ' '.join(without_fw)
	return without_rw


# initiate object for counter
counter = Counter()
# some random text on machine learning
ex_fw = """
Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.
For example, one kind of algorithm is a classification algorithm. It can put data into different groups. The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code. It's the same algorithm but it's fed different training data so it comes up with different classification logic.
Two kinds of Machine Learning Algorithms
You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning. The difference is simple, but really important.
Supervised Learning
Let's say you are a real estate agent. Your business is growing, so you hire a bunch of new trainee agents to help you out. But there's a problem -- you can glance at a house and have a pretty good idea of what a house is worth, but your trainees don't have your experience so they don't know how to price their houses.
To help your trainees (and maybe free yourself up for a vacation), you decide to write a little app that can estimate the value of a house in your area based on it's size, neighborhood, etc, and what similar houses have sold for.
So you write down every time someone sells a house in your city for 3 months. For each house, you write down a bunch of details -- number of bedrooms, size in square feet, neighborhood, etc. But most importantly, you write down the final sale price:
This is called supervised learning. You knew how much each house sold for, so in other words, you knew the answer to the problem and could work backwards from there to figure out the logic.
To build your app, you feed your training data about each house into your machine learning algorithm. The algorithm is trying to figure out what kind of math needs to be done to make the numbers work out.
This kind of like having the answer key to a math test with all the arithmetic symbols erased:
"""

# calling rare_words to calculate rare words
RareWords = rare_words(ex_fw)
print(f"Top 10 Rarer Words from our example text :- \n{RareWords}\n")

# calling remove_fw to remove rare words from example text
rw_result = remove_fw(ex_fw, RareWords)

print(f"Result after removing rare words :-\n{rw_result}")

"""
Output::

Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem . Instead of writing code , you feed data to the generic algorithm and it builds its own logic based on the data . For example , one kind of algorithm is a classification algorithm . It can put data into different groups . The same classification algorithm used to recognize handwritten numbers could also be used to classify emails into spam and not-spam without changing a line of code . It ' s the same algorithm but it ' s fed different training data so it comes up with different classification logic . Two kinds of Machine Learning Algorithms You can think of machine learning algorithms as falling into one of two main categories -- supervised learning and unsupervised learning . The difference is simple , but really important . Supervised Learning Let ' s say you are a real estate agent . Your business is growing , so you hire a bunch of new trainee agents to help you out . But there ' s a problem -- you can glance at a house and have a pretty good idea of what a house is worth , but your trainees don ' t have your experience so they don ' t know how to price their houses . To help your trainees ( and maybe free yourself up for a vacation ) , you decide to write a little app that can estimate the value of a house in your area based on it ' s size , neighborhood , etc , and what similar houses have sold for . So you write down every time someone sells a house in your city for 3 months . For each house , you write down a bunch of details -- number of bedrooms , size in square feet , neighborhood , etc . But most importantly , you write down the final sale price : This is called supervised learning . You knew how much each house sold for , so in other words , you knew the answer to the problem and could work backwards from there to figure out the logic . To build your app , you feed your training data about each house into your machine learning algorithm . The algorithm is trying to figure out what kind of math needs to be to the numbers work out . This kind of having the answer to a math with the
"""

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
## Remove single characters

def remove_single_char(text):
	"""
	Return :- string after removing single characters
	Input :- string
	Output:- string
	"""
	single_char_pattern = r'\s+[a-zA-Z]\s+'
	without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
	return without_sc

# example text for removing single characters
ex_sc = """
this is an example of single characters like a , b , and c .
"""
# calling remove_sc function to remove single characters
sc_result = remove_single_char(ex_sc)
print(f"Result :-\n{sc_result}")

## Output:: this is an example of single characters like , , and .

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
# Removing Extra Whitespaces

import re

def remove_extra_spaces(text):
	"""
	Return :- string after removing extra whitespaces
	Input :- String
	Output :- String
	"""
	space_pattern = r'\s+'
	without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
	return without_space


# example text for removing extra spaces
ex_space = """
this      is an


extra spaces        .
"""

space_result = remove_extra_spaces(ex_space)
print(f"Result :- \n{space_result}")

## Output:: this is an extra spaces .
""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
