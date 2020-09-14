import re
from bs4 import BeautifulSoup
from num2words import num2words
from spellchecker import SpellChecker
from autocorrect import Speller
from nltk import word_tokenize
import unidecode
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from emoticons_list import EMOTICONS 
from emoticons_list import EMO_UNICODE
from string import punctuation
from nltk.corpus import stopwords
import spacy
import gensim
from collections import Counter
from contraction import CONTRACTION_MAP
from progress.bar import Bar


spell_corrector = SpellChecker()
# expending contractions 
contraction_mapping = CONTRACTION_MAP
# take all key values from contraction_mapping 
contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)

# initialize porter stemmer object
stemmer = PorterStemmer()
# initializa lemmatizer object
lemma = WordNetLemmatizer()
# convert emo_unicode to unicode_emo 
UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}
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
# initiate object for counter
counter = Counter()


class Preprocess:
	def __init__(self):
		self.data = None
		self.techniques = None

	def lower_case_convertion(self, text):
		"""
		Input :- string
		Output :- lowercase string
		"""
		lower_text = text.lower()
		return lower_text

	def remove_html_tags(self, text):
		"""
		Return :- String without Html tags
		input :- String
		Output :- String
		"""
		html_pattern = r'<.*?>'
		without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
		return without_html

	def remove_urls(self, text):
		"""
		Return :- String without URLs 
		input :- String
		Output :- String
		"""
		url_pattern = r'https?://\S+|www\.\S+'
		without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
		return without_urls

	def remove_numbers(self, text):
		"""
		Return :- String without numbers 
		input :- String
		Output :- String
		"""
		number_pattern = r'\d+'
		without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
		return without_number

	def num_to_words(self, text):
		"""
		Return :- text which have all numbers or integers in the form of words
		Input :- string
		Output :- string
		"""
		# spliting text into words with space 
		after_spliting = word_tokenize(text)

		for index in range(len(after_spliting)):
			if after_spliting[index].isdigit():
				after_spliting[index] = num2words(after_spliting[index])

		# joining list into string with space
		numbers_to_words = ' '.join(after_spliting)
		return numbers_to_words

	def spell_correction(self, text):
		"""
		Return :- text which have correct spelling words
		Input :- string
		Output :- string
		"""
		# initialize empty list to save correct spell words
		correct_words = []
		# extract spelling incorrect words by uing unknown function of spellchecker
		misSpelled_words = spell_corrector.unknown(word_tokenize(text))

		for each_word in word_tokenize(text):
			if each_word in misSpelled_words:
				right_word = spell_corrector.correction(each_word)
				correct_words.append(right_word)
			else:
				correct_words.append(each_word)

		# joining correct_words list into single string
		correct_spelling = ' '.join(correct_words)
		return correct_spelling

	def accented_to_ascii(self, text):
		"""
		Return :- text after converting accented characters
		Input :- string
		Output :- string
		"""
		# apply unidecode function on text to convert
		# accented characters to ASCII values
		text = unidecode.unidecode(text)
		return text

	def short_to_original(self, text):
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

	def expand_contractions(self, contraction):
		# take maching contraction in the text
		match = contraction.group(0)
		# first char from matching contrcation (D for Doesn't)
		first_char = match[0]
		if contraction_mapping.get(match):
			expanded_contraction = contraction_mapping.get(match)
		else:
			expanded_contraction = contraction_mapping.get(match.lower())
		expanded_contraction = first_char+expanded_contraction[1:]

		return expanded_contraction


	def porter_stemmer(self, text):
		"""
		Result :- string after stemming
		Input :- String
		Ouput :- String
		"""
		# word tokenization
		tokens = word_tokenize(text)

		for index in range(len(tokens)):
			# stem word to each word
			stem_word = stemmer.stem(tokens[index])
			# uodate tokens list with stem word
			tokens[index] = stem_word

		# join list with space separator as string
		return ' '.join(tokens)

	def lemmatization(self, text):
		"""
		Result :- string after stemming
		Input :- String
		Ouput :- String
		"""
		# word tokenization
		tokens = word_tokenize(text)

		for index in range(len(tokens)):
			# lemma word 
			lemma_word = lemma.lemmatize(tokens[index])
			tokens[index] = lemma_word

		return ' '.join(tokens)

	def remove_emojis(self, text):
		"""
		Result :- string without any emojis in it
		Input :- String
		Ouput :- String
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

	def remove_emoticons(self, text):
		"""
		Return :- string after removing emoticons
		Input :- string
		Output :- string
		"""
		emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

		without_emoticons = emoticon_pattern.sub(r'',text)
		return without_emoticons

	def emoji_words(self, text):
		"""
		Return :- string after converting emojis to words
		Input :- String
		Output :- String
		"""
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

	def emoticons_words(self, text):
		"""
		Return :- string after converting emoticons to words
		Input :- String
		Output :- String
		"""
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

	def remove_punctuation(self, text):
		"""
		Return :- String after removing punctuations
		Input :- String
		Output :- String 
		"""
		return text.translate(str.maketrans('', '', punctuation))

	def remove_stopwords(self, text):
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

		# joining all tokens afetr removing stopwords
		without_sw = ' '.join(text_without_sw)
		return without_sw

	def freq_words(self, text):
		"""
		Return :- Most frequent words
		Input :- string
		Output :- list of frequent words
		"""
		# tokenization
		tokens = word_tokenize(text)
		for word in tokens:
			counter[word]= +1

		FrequentWords = []
		# take top 10 frequent words
		for (word, word_count) in counter.most_common(10):
			FrequentWords.append(word)

		return set(FrequentWords)

	def remove_fw(self, text, FrequentWords):
		"""
		Return :- String after removing frequent words
		Input :- String
		Output :- String 
		"""
		# FrequentWords = self.freq_words(text)

		for index in range(len(text)):
			tokens = word_tokenize(text[index])
			without_fw = []
			for word in tokens:
				if word not in FrequentWords:
					without_fw.append(word)
			without_fw = ' '.join(without_fw)
			text[index] = without_fw
		return text

	def rare_words(self, text):
		"""
		Return :- Most Rarer words
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

		return set(RareWords)

	def remove_rw(self,text, RareWords):
		"""
		Return :- String after removing frequent words
		Input :- String
		Output :- String 
		"""

		# RareWords = self.rare_words(text)
		for index in range(len(text)):
			tokens = word_tokenize(text[index])
			without_rw = []
			for word in tokens:
				if word not in RareWords:
					without_rw.append(word)

			without_rw = ' '.join(without_rw)
			text[index] = without_rw

		return text

	def remove_single_char(self,text):
		"""
		Return :- string after removing single charcters
		Input :- string
		Output:- string
		"""
		single_char_pattern = r'\s+[a-zA-Z]\s+'
		without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
		return without_sc

	def remove_extra_spaces(self,text):
		"""
		Return :- string after removing extra whitespaces
		Input :- String
		Output :- String
		"""
		space_pattern = r'\s+'
		without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
		return without_space

	def remove_null_sent(self, text):
		"""
		Return :- removing all null text or length of 0 or 1 length of sent
		Input :- list of string
		Output :- list of string
		"""
		sent_list = []
		for sent in text:
			if len(sent)>2:
				sent_list.append(sent.strip())
		return sent_list

	# # Print iterations progress
	# def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
	#     """
	#     Call in a loop to create terminal progress bar
	#     @params:
	#         iteration   - Required  : current iteration (Int)
	#         total       - Required  : total iterations (Int)
	#         prefix      - Optional  : prefix string (Str)
	#         suffix      - Optional  : suffix string (Str)
	#         decimals    - Optional  : positive number of decimals in percent complete (Int)
	#         length      - Optional  : character length of bar (Int)
	#         fill        - Optional  : bar fill character (Str)
	#         printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	#     """
	#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	#     filledLength = int(length * iteration // total)
	#     bar = fill * filledLength + '-' * (length - filledLength)
	#     print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	#     # Print New Line on Complete
	#     if iteration == total: 
	#         print()

	def preprocessing(self, data, techniques):
		self.data = data 
		self.techniques = techniques
		
		"""
		lcc = lower case convertion
		rht = Removing HTML tags
		rurls = Revoing Urls 
		rn = Removing Numbers
		ntw = convert numbers to words
		sc = Spelling Correction
		ata = convert accented to ASCII code
		sto = short_to_original
		ec = Expanding Contractions
		ps = Stemming (Porter Stemming)
		l = Lemmatization
		re = Removing Emojis
		ret = Removing Emoticons
		ew = Convert Emojis to words
		etw = Convert Emoticons to words
		rp = Removing Punctuations
		rs = Removing Stopwords
		rfw = Removing Frequent Words
		rrw = Removing Rare Words
		rsc = Removing Single characters
		res = Removing Extra Spaces
		"""
		techniques_dict = {
							'lcc':self.lower_case_convertion, 
							'rht':self.remove_html_tags, 
							'rurls':self.remove_urls, 
							'rn':self.remove_numbers, 
							'ntw':self.num_to_words, 
							'sc':self.spell_correction,
							'ata':self.accented_to_ascii,
							'sto':self.short_to_original,
							'ec':self.expand_contractions,
							'ps':self.porter_stemmer,
							'l':self.lemmatization,
							're':self.remove_emojis,
							'ret':self.remove_emoticons,
							'ew':self.emoji_words,
							'etw':self.emoticons_words,
							'rp':self.remove_punctuation,
							'rs':self.remove_stopwords,
							'rfw':self.remove_fw,
							'rrw':self.remove_rw,
							'rsc':self.remove_single_char,
							'res':self.remove_extra_spaces
							}
		total_iterations = len(techniques)*len(data)
		# Initial call to print 0% progress
		# printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = total_iterations)
		bar = Bar('Technique Processing', max=len(techniques))
		i=1
		for method in techniques:
			print(f"method *************:- {method}")
			bar_1 = Bar(str(i)+" "+'text Processing', max=len(data))
			if method == 'rfw' or method == 'rrw':
				print(f"method *************:- {method}")
				if method == 'rfw':
					list_words = self.freq_words(' '.join(data))
				else:
					list_words = self.rare_words(' '.join(data))
				data = techniques_dict[method](data, list_words)
				bar_1.next()
			else:
				for index in range(len(data)):
					sent = data[index]
					data[index] = techniques_dict[method](sent)
					bar_1.next()
					# Update Progress Bar
					# printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = total_iterations)
				bar_1.finish()
			i +=1
			bar.next()
		bar.finish()

		data = self.remove_null_sent(data)
		
		return data
