"""
===============================================
Objective: Explain how to build the classification model using the gradient boosting algorithm
Author: Shaik Sameeruddin
Blog: https://dataaspirant.com
Date: 2020-11-22
===============================================
"""


## Dataset of test classification
from sklearn.datasets import make_classification
## defining dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
## summarizing the dataset
print(X.shape, y.shape)
## Output
## (1000, 20) (1000,)


## Evaluate classification gradient enhancement algorithm
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
## defining dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
## defining the model
model = GradientBoostingClassifier()
## defining the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
## Assess the dataset model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
## production of the study
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
## Output
## Mean Accuracy: 0.899 (0.030)


## Predict using classification gradient boosting
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
## define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
n_redundant=5, random_state=7)
## define the model
model = GradientBoostingClassifier()
## Fit the entire dataset model
model.fit(X, y)
## make a single prediction
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719,
 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717,
 -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633,
 -2.82646737, 0.44916808]

yhat = model.predict([row])
## summarize prediction
print('Predicted Class: %d' % yhat[0])
