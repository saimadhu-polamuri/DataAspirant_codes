#!/usr/bin/env python
# logistic_regression.py
# Author : Saimadhu
# Date: 19-March-2017
# About: Implementing Logistic Regression Classifier to predict to whom the voter will vote.

# Required Python Packages
import pandas as pd
import numpy as np
import pdb
import plotly.plotly as py
import plotly.graph_objs as go

# import plotly.plotly as py
# from plotly.graph_objs import *
py.sign_in('dataaspirant', 'RhJdlA1OsXsTjcRA0Kka')

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Files
DATA_SET_PATH = "../Inputs/anes_dataset.csv"


def dataset_headers(dataset):
    """
    To get the dataset header names
    :param dataset: loaded dataset into pandas DataFrame
    :return: list of header names
    """
    return list(dataset.columns.values)


def unique_observations(dataset, header, method=1):
    """
    To get unique observations in the loaded pandas DataFrame column
    :param dataset:
    :param header:
    :param method: Method to perform the unique (default method=1 for pandas and method=0 for numpy )
    :return:
    """
    try:
        if method == 0:
            # With Numpy
            observations = np.unique(dataset[[header]])
        elif method == 1:
            # With Pandas
            observations = pd.unique(dataset[header].values.ravel())
        else:
            observations = None
            print "Wrong method type, Use 1 for pandas and 0 for numpy"
    except Exception as e:
        observations = None
        print "Error: {error_msg} /n Please check the inputs once..!".format(error_msg=e.message)
    return observations


def feature_target_frequency_relation(dataset, f_t_headers):

    """
    To get the frequency relation between targets and the unique feature observations
    :param dataset:
    :param f_t_headers: feature and target header
    :return: feature unique observations dictionary of frequency count dictionary
    """

    feature_unique_observations = unique_observations(dataset, f_t_headers[0])
    unique_targets = unique_observations(dataset, f_t_headers[1])

    frequencies = {}
    for feature in feature_unique_observations:
        frequencies[feature] = {unique_targets[0]: len(
            dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[0])]),
            unique_targets[1]: len(
                dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[1])])}
    return frequencies


def feature_target_histogram(feature_target_frequencies, feature_header):
    """

    :param feature_target_frequencies:
    :param feature_header:
    :return:
    """
    keys = feature_target_frequencies.keys()
    y0 = [feature_target_frequencies[key][0] for key in keys]
    y1 = [feature_target_frequencies[key][1] for key in keys]

    trace1 = go.Bar(
        x=keys,
        y=y0,
        name='Clinton'
    )
    trace2 = go.Bar(
        x=keys,
        y=y1,
        name='Dole'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group',
        title='Feature :: ' + feature_header + ' Clinton Vs Dole votes Frequency',
        xaxis=dict(title="Feature :: " + feature_header + " classes"),
        yaxis=dict(title="Votes Frequency")
    )
    fig = go.Figure(data=data, layout=layout)
    # plot_url = py.plot(fig, filename=feature_header + ' - Target - Histogram')
    py.image.save_as(fig, filename=feature_header + '_Target_Histogram.png')


def train_logistic_regression(train_x, train_y):
    """
    Training logistic regression model with train dataset features(train_x) and target(train_y)
    :param train_x:
    :param train_y:
    :return:
    """

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model


def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score


def main():
    """
    Logistic Regression classifier main
    :return:
    """
    # Load the data set for training and testing the logistic regression classifier
    dataset = pd.read_csv(DATA_SET_PATH)
    print "Number of Observations :: ", len(dataset)

    # Get the first observation
    print dataset.head()

    headers = dataset_headers(dataset)
    print "Data set headers :: {headers}".format(headers=headers)

    training_features = ['TVnews', 'PID', 'age', 'educ', 'income']
    target = 'vote'

    # Train , Test data split
    train_x, test_x, train_y, test_y = train_test_split(dataset[training_features], dataset[target], train_size=0.7)
    print "train_x size :: ", train_x.shape
    print "train_y size :: ", train_y.shape

    print "test_x size :: ", test_x.shape
    print "test_y size :: ", test_y.shape

    print "edu_target_frequencies :: ", feature_target_frequency_relation(dataset, [training_features[3], target])

    for feature in training_features:
        feature_target_frequencies = feature_target_frequency_relation(dataset, [feature, target])
        feature_target_histogram(feature_target_frequencies, feature)

    # Training Logistic regression model
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)

    train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)

    # Testing the logistic regression model
    test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)

    print "Train Accuracy :: ", train_accuracy
    print "Test Accuracy :: ", test_accuracy

if __name__ == "__main__":
    main()