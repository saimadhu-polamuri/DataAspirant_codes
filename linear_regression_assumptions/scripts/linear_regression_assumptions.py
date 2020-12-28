"""
===============================================
Objective: Linear Regression Assumptions Codes
Author: Saumya Awasthi
Blog: https://dataaspirant.com
Date: 2020-12-28
===============================================
"""


## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Load dataset
df = pd.read_csv('../inputs/student_score.csv')
df.plot(x='Hours',y='Scores',style='o')

## Plot the graph
plt.title("Hours vs Percentage")    ## For linear regression assumptions
plt.xlabel("Hours studied")
plt.ylabel("Percentage Score")
plt.show()


## For checking the distrbution

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import norm

## input is the number of hours devoted
X = df.iloc[:,:-1].values
## output is percentage scored
y = df.iloc[:,1].values
## splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X, y,
test_size=0.3, random_state=21)

## training model
model = LinearRegression()
model.fit(X_train,y_train)
## making predictions
y_pred = model.predict(X_test)
residual = y_test-y_pred
sns.distplot(residual , fit=norm)
plt.title("Normal probability plot of residuals")
plt.xlabel("residuals")
plt.ylabel("frequency")


## For checking the distrbution with Q-Q plots
import statsmodels.api as sm
import pylab as py
# Randomly generating data points
data = np.random.normal(0,1,50)
sm.qqplot(data, line ='s')
py.title("Residuals distribution using Q-Q plots")
py.show()


## Creating heatmap

import seaborn as sns
# visualizing the relation between student's score and marks scored
sns.heatmap(df.corr())


## importing statsmodels
import numpy as np
from statsmodels.stats.stattools import durbin_watson

a = np.array([1, 2, 3,6,7,8])
## using statsmodels.durbin_watson()
d = durbin_watson(a)
print(d)
