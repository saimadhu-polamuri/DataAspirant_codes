"""
===============================================
Objective: Explain various optimization algorithms in deep learning
Author: Anber Arif
Blog: https://dataaspirant.com
Date: 2020-11-11
===============================================
"""


## Requried Python Packages
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge


## Create datasets
A, b, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)


alpha = 1
n, m = A.shape

I = np.identity(m)
w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + alpha * I), A.T), b)

## Outputs
## w = array([87.37153533])
## coefficients = array(90.34019153)

plt.scatter(A, b)
plt.plot(A, w*A, c='red')


## ridge with alpha 0.5
rr = Ridge(alpha=0.5)
rr.fit(A, b)
w = rr.coef_

## Output w = array([88.31917399])

plt.scatter(A, b)
plt.plot(A, w*A, c='red')

## ridge with alpha 10
rr = Ridge(alpha=10)
rr.fit(A, b)
w = rr.coef_[0]
plt.scatter(A, b)
plt.plot(A, w*A, c='red')

## ridge with alpha 100
rr = Ridge(alpha=100)
rr.fit(A, b)
w = rr.coef_[0]
plt.scatter(A, b)
plt.plot(A, w*A, c='red')
