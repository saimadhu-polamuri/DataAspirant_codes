"""
===============================================
Objective: Explain how to build the lasso regression with sklearn
Author: Aparna Moorthi
Blog: https://dataaspirant.com
Date: 2020-11-26
===============================================
"""


## Load requried packages
import pandas as pd
Import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


## load dataset
link = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = pd.read_csv(link, header=None)
# summarize shape
print(dataframe.shape)
# get information about the dataset
print(dataframe.describe())

"""
Output:

(506, 14)
       0 1 2 ... 11 12 13
count 506.000000 506.000000 506.000000 ... 506.000000 506.000000 506.000000
mean 3.613524 11.363636 11.136779 ... 356.674032 12.653063 22.532806
std 8.601545 23.322453 6.860353 ... 91.294864 7.141062 9.197104
min 0.006320 0.000000 0.460000 ... 0.320000 1.730000 5.000000
25% 0.082045 0.000000 5.190000 ... 375.377500 6.950000 17.025000
50% 0.256510 0.000000 9.690000 ... 391.440000 11.360000 21.200000
75% 3.677082 12.500000 18.100000 ... 396.225000 16.955000 25.000000
max 88.976200 100.000000 27.740000 ... 396.900000 37.970000 50.000000

[8 rows x 14 columns]

"""


## Train and test dataset creation
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30,
random_state=40)
print(Xtrain.shape)
print(Xtest.shape)


"""
Output:

(354, 13)
(152, 13)

"""

## Build the lasso model with alpha

model_lasso = Lasso(alpha=1)
model_lasso.fit(Xtrain, ytrain)
pred_train_lasso= model_lasso.predict(Xtrain)
pred_test_lasso= model_lasso.predict(Xtest)


## Evaluate the lasso model
print(np.sqrt(mean_squared_error(ytrain,pred_train_lasso)))
print(r2_score(ytrain, pred_train_lasso))
print(np.sqrt(mean_squared_error(ytest,pred_test_lasso)))
print(r2_score(ytest, pred_test_lasso))

"""
Output:

4.887113841773082
0.6657249068677625
6.379797782769904
0.6439373929767929
"""


## Tunning lasso regression model

from numpy import arange
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.linear_model import LassoCV

## load the dataset
link = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataset = pd.read_csv(link, header=None)
dataframe = dataset.values
X, y = dataframe [:, :-1], dataframe [:, -1]

## define model evaluation method
cross_validation = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

## define model

lasso_model = LassoCV(alphas=arange(0, 1, 0.02), cv=cross_validation , n_jobs=-1)

## fit model
lasso_model .fit(X, y)
## summarize chosen configuration
print('alpha: %f' % lasso_model .alpha_)

pred_train_lasso= lasso_model .predict(X_train)
pred_test_lasso= lasso_model .predict(X_test)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso)))
print(r2_score(y_test, pred_test_lasso))


"""
Output:

alpha: 0.000000
4.3376084680597495
0.7366703249573084
5.391470274340004
0.7457113531104524
"""
