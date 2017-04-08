#!/usr/bin/env python
# python_pickle.py
# Author : Saimadhu
# Date: 12-Feb-2017
# About: Examples on How to pickle the python object

# Required Python Packages

import pickle
import pandas as pd
# from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier


# pickle list object


def pickle_list_object():
    numbers_list = [1, 2, 3, 4, 5]
    list_pickle_path = 'list_pickle.pkl'

    # Create an variable to pickle and open it in write mode
    list_pickle = open(list_pickle_path, 'wb')
    pickle.dump(numbers_list, list_pickle)
    list_pickle.close()

    # unpickling the list object

    # Need to open the pickled list object into read mode

    list_unpickle = open(list_pickle_path, 'rb')

    # load the unpickle object into a variable
    numbers_list = pickle.load(list_unpickle)

    print "Numbers List :: ", numbers_list


# Load the dataset
balance_scale_data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', sep=',', header=None)
# print "Dataset Length:: ", len(balance_scale_data)
# print "Dataset Shape:: ", balance_scale_data.shape

# Split the dataset into train and test dataset
X = balance_scale_data.values[:, 1:5]
Y = balance_scale_data.values[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Decision model with Gini index critiria
decision_tree_model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
decision_tree_model.fit(X_train, y_train)
# print "Decision Tree classifier :: ", decision_tree_model

# Dump the trained decision tree classifier with Pickle
decision_tree_pkl_filename = 'decision_tree_classifier_20170212.pkl'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(decision_tree_model, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()

# Loading the saved decision tree model pickle
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)
print "Loaded Decision tree model :: ", decision_tree_model
