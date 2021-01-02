"""
===============================================
Objective: Catboost algorithm implementation
Author: Samuel Adebayo
Blog: https://dataaspirant.com
Date: 2021-01-02
===============================================
"""


## import the libraries needed
import pandas as pd
import numpy as np

# Here we import our dataset from the CatBoost dataset library
from catboost.datasets import titanic

## The titanic dataset is made up of the train and test set, so we have to separate the data
titanic_train, titanic_test = titanic()

## Here we create a list to sort the columns so that the "Survived" column comes last
## This is because "Survived" is the target
column_sort = [ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
'Fare', 'Cabin', 'Embarked','Survived']

## Now we apply the sorted columns to the train data
train = titanic_train[column_sort]
train.set_index('Pclass') ## Not necessary just to get of the default index

test = titanic_test
train.head()

## Remember the target column - "Survived" - we identified in the cell above,
## it is missing in the test set
## To solve this problem, we would create a column and fill it with dummy values,
## let's say '2' so it is not dormant and we can merge the DataFrame later
test['Survived'] = 2  ## The numpy background of pandas allows this to work
test.sample(5) ## shows five random rows in the dataset

## We would combine the train and test set into one DataFrame,
## so we do not have to repeat the same process for the two sets
df = pd.concat([train,test],ignore_index = False)

## Some features (such as Name, and Age) are irrelevant so we delete them
df = df.drop(['Name', 'Age'], axis=1)

## The data is not clean so we check all the columns for missing values
df.isnull().sum(axis=0)


## "Fare", "Cabin", "Embarked", and "PassengerId" have missing values, we have to fix this
df['Embarked'] = df['Embarked'].fillna('S') ## The missing values in Embarked is filled with "S" (for Southampton), the most common value observed in that column
df['Cabin'] = df['Cabin'].fillna('Undefined')
df.fillna(-999, inplace=True)

## Now that the data looks good, we have to separate the train from the test set
train = df[df.Survived != 2]

test = df[df.Survived == 2]
test = test.drop(['Survived'], axis=1) ## drop the placeholder we created earlier in the test set

## Pop out the training features from the target variable
target = train.pop('Survived')
target.head()

## Let's ensure the model is trained and fit well
cat_features_index = np.where(train.dtypes != float)[0]

## Split the data into a train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, target,
train_size=0.85, random_state=1234)

## Import the CatBoostClassifier to fit the model and run a prediction
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42)

## Set the metric for evaluation
model = CatBoostClassifier(eval_metric='Accuracy',
use_best_model=True,  random_seed=42)

model.fit(X_train, y_train, cat_features=cat_features_index,
eval_set=(X_test, y_test))


from catboost import cv
from sklearn.metrics import accuracy_score

print('the test accuracy is :{:.6f}'.format(accuracy_score(
y_test, model.predict(X_test))))
