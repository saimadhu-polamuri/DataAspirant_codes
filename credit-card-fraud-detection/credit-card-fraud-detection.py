"""
===============================================
Objective: Building credit card fraud detection models with classification algorithms
Author: Sharmila.Polamuri
Blog: https://dataaspirant.com
Date: 2020-09-21
===============================================
"""


import pandas as pd

# load dataset
fraud_df = pd.read_csv("project/Dataset/creditcard.csv")

print(f"Dataset Shape :- \n {fraud_df.shape}")

## Output
"""
Dataset Shape:-
(284807, 31)
"""

print(f"Columns or Feature names :- \n {fraud_df.columns}")

## Output
"""
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
"""

print(f"Unique values of target variable :- \n {fraud_df['Class'].unique()}")

## Output
"""
Unique values of target variable
array([0, 1])
""""

print(f"Number of samples under each target value :- \n {fraud_df['Class'].value_counts()}")

## Output
"""
Number of samples under each target value
0    284315
1       492
Name: Class, dtype: int64
"""

# make sure which features are useful & which are not
# we can remove irrelevant features
fraud_df = fraud_df.drop(['Time'], axis=1)
print(f"list of feature names after removing Time column :- \n{fraud_df.columns}")

## Output
"""
list of feature names after removing Time column :-
Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'],
      dtype='object')
"""

print(f"Dataset info :- \n {fraud_df.info()}")

## Output
"""

Dataset info :-
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 30 columns):
 #   Column  Non-Null Count   Dtype
---  ------  --------------   -----
 0   V1      284807 non-null  float64
 1   V2      284807 non-null  float64
 2   V3      284807 non-null  float64
 3   V4      284807 non-null  float64
 4   V5      284807 non-null  float64
 5   V6      284807 non-null  float64
 6   V7      284807 non-null  float64
 7   V8      284807 non-null  float64
 8   V9      284807 non-null  float64
 9   V10     284807 non-null  float64
 10  V11     284807 non-null  float64
 11  V12     284807 non-null  float64
 12  V13     284807 non-null  float64
 13  V14     284807 non-null  float64
 14  V15     284807 non-null  float64
 15  V16     284807 non-null  float64
 16  V17     284807 non-null  float64
 17  V18     284807 non-null  float64
 18  V19     284807 non-null  float64
 19  V20     284807 non-null  float64
 20  V21     284807 non-null  float64
 21  V22     284807 non-null  float64
 22  V23     284807 non-null  float64
 23  V24     284807 non-null  float64
 24  V25     284807 non-null  float64
 25  V26     284807 non-null  float64
 26  V27     284807 non-null  float64
 27  V28     284807 non-null  float64
 28  Amount  284807 non-null  float64
 29  Class   284807 non-null  int64
dtypes: float64(29), int64(1)
memory usage: 65.2 MB
"""

print(f"few values of Amount column :- \n {fraud_df['Amount'][0:4]}")

## Output
"""
few values of Amount column :-
 0    149.62
1      2.69
2    378.66
3    123.50
Name: Amount, dtype: float64
"""

# data preprocessing
from sklearn.preprocessing import StandardScaler
fraud_df['norm_amount'] = StandardScaler().fit_transform(
fraud_df['Amount'].values.reshape(-1,1))
fraud_df = fraud_df.drop(['Amount'], axis=1)
print(f"few values of Amount column after applying StandardScaler:- \n {fraud_df['norm_amount'][0:4]}")

## Output

"""
few values of Amount column after applying StandardScaler:-
 0    0.244964
1   -0.342475
2    1.160686
3    0.140534
Name: norm_amount, dtype: float64
"""

## Features and target creations
X = fraud_df.drop(['Class'], axis=1)
y = fraud_df[['Class']]


# splitting dataset to train & test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

## Output
"""
(199364, 29)
(85443, 29)
(199364, 1)
(85443, 1)
"""

## Building decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def decision_tree_classification(X_train, y_train, X_test, y_test):
    # initialize object for DecisionTreeClassifier class
    dt_classifier = DecisionTreeClassifier()
    # train model by using fit method
    print("Model training starts........")
    dt_classifier.fit(X_train, y_train.values.ravel())
    print("Model training completed")
    acc_score = dt_classifier.score(X_test, y_test)
    print(f'Accuracy of model on test dataset :- {acc_score}')
    # predict result using test dataset
    y_pred = dt_classifier.predict(X_test)
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")



# calling decision_tree_classification method to train and evaluate model
decision_tree_classification(X_train, y_train, X_test, y_test)

## Output
"""
Model training starts........
Model training completed
Accuracy of model on test dataset :- 0.9992275552122467
Confusion Matrix :-
 [[85266    30]
 [   36   111]]
Classification Report :-
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     85296
           1       0.79      0.76      0.77       147

    accuracy                           1.00     85443
   macro avg       0.89      0.88      0.89     85443
weighted avg       1.00      1.00      1.00     85443
"""

## Model with randomforest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def random_forest_classifier(X_train, y_train, X_test, y_test):
     # initialize object for DecisionTreeClassifier class
     rf_classifier = RandomForestClassifier(n_estimators=50)
     # train model by using fit method
     print("Model training starts........")
     rf_classifier.fit(X_train, y_train.values.ravel())
     acc_score = rf_classifier.score(X_test, y_test)
     print(f'Accuracy of model on test dataset :- {acc_score}')
     # predict result using test dataset
     y_pred = rf_classifier.predict(X_test)
     # confusion matrix
     print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
     # classification report for f1-score
     print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")


# calling random_forest_classifier
random_forest_classifier(X_train, y_train, X_test, y_test)

"""
Model training starts........
Accuracy of model on test dataset :- 0.9994967405170698
Confusion Matrix :-
 [[85289     7]
 [   36   111]]
Classification Report :-
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     85296
           1       0.94      0.76      0.84       147

    accuracy                           1.00     85443
   macro avg       0.97      0.88      0.92     85443
weighted avg       1.00      1.00      1.00     85443
"""

## Target class distribution
class_val = fraud_df['Class'].value_counts()
print(f"Number of samples for each class :- \n {class_val}")
non_fraud = class_val[0]
fraud = class_val[1]
print(f"Non Fraudulent Numbers :- {non_fraud}")
print(f"Fraudulent Numbers :- {fraud}")

##Output
"""

Number of samples for each class :-
 0    284315
1       492
Name: Class, dtype: int64
Non Fraudulent Numbers :- 284315
Fraudulent Numbers :- 492

"""

## Equal both the target samples to the same level
# take indexes of non fraudulent
nonfraud_indexies = fraud_df[fraud_df.Class == 0].index
fraud_indices = np.array(fraud_df[fraud_df['Class'] == 1].index)
# take random samples from non fraudulent that are equal to fraudulent samples
random_normal_indexies = np.random.choice(nonfraud_indexies, fraud, replace=False)
random_normal_indexies = np.array(random_normal_indexies)


## Undersampling techniques

# concatenate both indices of fraud and non fraud
under_sample_indices = np.concatenate([fraud_indices, random_normal_indexies])

#extract all features from whole data for under sample indices only
under_sample_data = fraud_df.iloc[under_sample_indices, :]

# now we have to divide under sampling data to all features & target
x_undersample_data = under_sample_data.drop(['Class'], axis=1)
y_undersample_data = under_sample_data[['Class']]
# now split dataset to train and test datasets as before
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
x_undersample_data, y_undersample_data, test_size=0.2, random_state=0)


## DecisionTreeClassifier after applying undersampling technique

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def decision_tree_classification(X_train, y_train, X_test, y_test):
 # initialize object for DecisionTreeClassifier class
 dt_classifier = DecisionTreeClassifier()
 # train model by using fit method
 print("Model training start........")
 dt_classifier.fit(X_train, y_train.values.ravel())
 print("Model training completed")
 acc_score = dt_classifier.score(X_test, y_test)
 print(f'Accuracy of model on test dataset :- {acc_score}')
 # predict result using test dataset
 y_pred = dt_classifier.predict(X_test)
 # confusion matrix
 print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
 # classification report for f1-score
 print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")
 print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

# calling decision tree classifier function
decision_tree_classification(X_train_sample, y_train_sample,
X_test_sample, y_test_sample)

"""
Model training starts........
Model training completed
Accuracy of model on test dataset :- 0.9035532994923858
Confusion Matrix :-
 [[91 15]
 [ 4 87]]
Classification Report :-
               precision    recall  f1-score   support

           0       0.96      0.86      0.91       106
           1       0.85      0.96      0.90        91

    accuracy                           0.90       197
   macro avg       0.91      0.91      0.90       197
weighted avg       0.91      0.90      0.90       197

AROC score :-
 0.9072672610408461
"""

## RandomForestClassifier after apply the undersampling techniques

from sklearn.tree import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def random_forest_classifier(X_train, y_train, X_test, y_test):
 # initialize object for DecisionTreeClassifier class
 rf_classifier = RandomForestClassifier(n_estimators=50)
 # train model by using fit method
 print("Model training start........")
 rf_classifier.fit(X_train, y_train.values.ravel())
 acc_score = rf_classifier.score(X_test, y_test)
 print(f'Accuracy of model on test dataset :- {acc_score}')
 # predict result using test dataset
 y_pred = rf_classifier.predict(X_test)
 # confusion matrix
 print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
 # classification report for f1-score
 print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")
 # area under roc curve
 print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

random_forest_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

## Model accuracy details after applyig the undersampling techniques
"""
Model training starts........
Accuracy of model on test dataset :- 0.949238578680203
Confusion Matrix :-
 [[100   6]
 [  4  87]]
Classification Report :-
               precision    recall  f1-score   support

           0       0.96      0.94      0.95       106
           1       0.94      0.96      0.95        91

    accuracy                           0.95       197
   macro avg       0.95      0.95      0.95       197
weighted avg       0.95      0.95      0.95       197

AROC score :-
 0.9497200912295253
 """
