"""
===============================================
Objective: Implementing confusion matrix in different ways.
Author: saimadhu.polamuri
Blog: https://dataaspirant.com
Date: 2020-08-02
===============================================
"""


# Confusion matrix with sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
actuals = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
predictions = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])

print("Confusion matrix \n", confusion_matrix(actuals, predictions))


# Confusion matrix with tensorflow
import tensorflow as tf
actuals = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
predictions = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1]

con = tf.confusion_matrix(labels=actuals, predictions=predictions )
sess = tf.Session()
with sess.as_default():
        print(sess.run(con))



# Creating the confusion matrix graphs
import seaborn as sns
from matplotlib import pyplot as plt
cf_train_matrix = confusion_matrix(actuals, predictions)
plt.figure(figsize=(10,8))
sns.heatmap(cf_train_matrix, annot=True, fmt='d')
