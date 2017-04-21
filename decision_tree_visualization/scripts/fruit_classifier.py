#!/usr/bin/env python
# fruit_classifier.py
# Author : Saimadhu Polamuri
# Date: 18-April-2017
# About: Fruit classifier with decision tree algorithm

# Required Python Packages
import pandas as pd
import numpy as np
from sklearn import tree


# creating dataset for modeling Apple / Orange classification
fruit_data_set = pd.DataFrame()
fruit_data_set["fruit"] = np.array([1, 1, 1, 1, 1,      # 1 for apple
                                    0, 0, 0, 0, 0])     # 0 for orange
fruit_data_set["weight"] = np.array([170, 175, 180, 178, 182,
                                     130, 120, 130, 138, 145])
fruit_data_set["smooth"] = np.array([9, 10, 8, 8, 7,
                                     3, 4, 2, 5, 6])

fruit_classifier = tree.DecisionTreeClassifier()
fruit_classifier.fit(fruit_data_set[["weight", "smooth"]], fruit_data_set["fruit"])

print ">>>>> Trained fruit_classifier <<<<<"
print fruit_classifier

# fruit data set 1st observation
test_features_1 = [[fruit_data_set["weight"][0], fruit_data_set["smooth"][0]]]
test_features_1_fruit = fruit_classifier.predict(test_features_1)
print "Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][0], predicted_fruit=test_features_1_fruit)

# fruit data set 3rd observation
test_features_3 = [[fruit_data_set["weight"][2], fruit_data_set["smooth"][2]]]
test_features_3_fruit = fruit_classifier.predict(test_features_3)
print "Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][2], predicted_fruit=test_features_3_fruit)

# fruit data set 8th observation
test_features_8 = [[fruit_data_set["weight"][7], fruit_data_set["smooth"][7]]]
test_features_8_fruit = fruit_classifier.predict(test_features_8)
print "Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][7], predicted_fruit=test_features_8_fruit)


with open("fruit_classifier.txt", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f)

# converting into the pdf file
with open("fruit_classifier.dot", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f)



