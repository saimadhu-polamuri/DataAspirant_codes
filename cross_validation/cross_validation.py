"""
===============================================
Objective: Cross validation implementation
Author: Anber Arif
Blog: https://dataaspirant.com
Date: 2020-12-03
===============================================
"""


## k-fold cross validation demonstration by scikit-learn library
from numpy import array
from sklearn.model_selection import KFold

## supposed data sample
data = array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

## cross validation preparation
kfold = KFold(3, True, 1)

## enumeration of the splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))
