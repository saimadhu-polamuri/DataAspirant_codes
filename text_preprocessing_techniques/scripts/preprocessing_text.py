import pandas as pd
from preprocessing import Preprocess

dataset = pd.read_table('SMSSpamCollection', header=None, encoding="utf-8")
# take text column
text_column = dataset[1]

sentences = text_column.tolist()
# below technique list 
"""
lcc = Lower case convertion
rurls = Revoing URLs
ntw = convert numbers to words
res = Removing Extra Spaces
"""
# techniques = ["lcc", "rht", "rurls","rn", "sc", "ntw","ata","sto","ec","ps","l","re","ret","ew","etw","rp","rs","rfw","rrw","rsc", "res"]
techniques = ["rfw","rrw","rsc", "res"]

# remaining techniques 
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
print(f"******** Before preprocessing technique ******* ")
for sent in sentences[:5]:
	print(sent)
preprocessing = Preprocess()

preprocessed_text = preprocessing.preprocessing(sentences, techniques)
print(f"******** After preprocessing ****************")
for sent in preprocessed_text[:5]:
	print(sent)
