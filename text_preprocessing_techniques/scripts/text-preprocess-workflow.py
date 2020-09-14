"""
===============================================
Objective: Text preprocessing techniques workflow in python
Author: Sharmila.Polamuri
Blog: https://dataaspirant.com
Date: 2020-09-14
===============================================
"""



import pandas as pd
from preprocessing import Preprocess

dataset = pd.read_table('SMSSpamCollection', header=None, encoding="utf-8")
# take text column
text_column = dataset[1]

sentences = text_column.tolist()
# below technique list
"""
lcc = Lower case conversion
rurls = Removing URLs
ntw = convert numbers to words
res = Removing Extra Spaces
"""
techniques = ["lcc", "rurls","sc", "ntw", "res"]

# remaining techniques
"""
	lcc = lower case conversion
	rht = Removing HTML tags
	rurls = Removing Urls
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
print(f"******** Before preprocessing technique ******* ")
for sent in sentences[:5]:
	print(sent)
# initiate Preprocess object
preprocessing = Preprocess()

preprocessed_text = preprocessing.preprocessing(sentences, techniques)
print(f"******** After preprocessing ****************")
for sent in preprocessed_text[:5]:
	print(sent)


"""
Output::
******** Before preprocessing technique *******
Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
Ok lar... Joking wif u oni...
Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
U dun say so early hor... U c already then say...
Nah I don't think he goes to usf, he lives around here though
1 text Processing |################################| 5572/5572
2 text Processing |################################| 5572/5572
3 text Processing |################################| 5572/5572
4 text Processing |################################| 5572/5572
Technique Processing |################################| 4/4
******** After preprocessing ****************
go until jurong point , crazy.. available only in bugis n great world la e buffet ... cine there got amore wat ...
ok lar ... joking wif u oni ...
free entry in two a wkly comp to win fa cup final tkts 21st may 2005. text fa to eighty-seven thousand, one hundred and twenty-one to receive entry question ( std txt rate ) t & c 's apply 08452810075over18 's
u dun say so early hor ... u c already then say ...
nah i do n't think he goes to usf , he lives around here though
"""

dataaspirant-text-preprocessing-techniques-work-flow.py
