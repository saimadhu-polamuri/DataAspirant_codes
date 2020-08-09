"""
===============================================
Objective: Handling imbalance data
Author: Jaiganesh Nagidi
Blog: https://dataaspirant.com
Date: 2020-08-09
===============================================
"""


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
#nltk.download('stopwords') for downloading first time uncomment this one
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from collections import Counter


# Load dataset
df = pd.read_csv('spam.csv',encoding='latin-1')
df.head()


#drop unnecessary columns
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.columns=['label','text'] #renaming columns
df.head()

# Data preprocessing
df['label']=df.replace({'ham':0,'spam':1}) #you can also use other techniques

ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i]) #removing useless symbols except alphabets
    review = review.lower() #lowering the text
    review = review.split() #splitting the text

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()
y = df.label
y = y.astype('int') #converting into int type


# Model Building
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_test)


# Randomoversampler
from imblearn.over_sampling import RandomOverSampler
ovs = RandomOverSampler(random_state=42)
x_res, y_res = ovs.fit_sample(x,y)


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))

# SMOTETomek
from imblearn.combine import SMOTETomek
smk = SMOTETomek()
x_res,y_res = smk.fit_sample(x,y)


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# Model on random oversampler data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res,
test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_test)


# Under sampling with randomundersample
from imblearn.under_sampling import RandomUnderSampler
ous = RandomUnderSampler(random_state=42)
x_res,y_res = ous.fit_sample(x,y)

# Under sampling with nearmiss
from imblearn.under_sampling import NearMiss
nm = NearMiss()
x_res,y_res = nm.fit_sample(x,y)


# model with under sampling data
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res,
test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
print("accuracy score:",accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
