"""
===============================================
Objective: R-squared Vs Adjusted R-squared comparision
Author: Saimadhu.Polamuri
Blog: https://dataaspirant.com
Date: 2020-10-14
===============================================
"""

## Requried Python Packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

## Paths
data_path = "../data/sales_data.csv"

## Residual sum of squares
def rss_value(actuals, forecasted):

    residuals = actuals - forecasted
    ## Squared each residual
    squared_residuals = [np.power(residual, 2) for residual in residuals]
    rss = sum(squared_residuals)
    return rss


## Total sum of square
def tss_value(actuals):

    ## Calcuate mean
    actual_mean = actuals.mean()
    ## Squared mean difference value
    mean_difference_squared = [np.power(
    (actual - actual_mean), 2) for actual in actuals]
    tss = sum(mean_difference_squared)
    return tss


## R-squared value
def get_r_squared_value(actuals, forecasted):

    rss = rss_value(actuals, forecasted)
    tss = tss_value(actuals)
    ## Calculating R-squared value
    r_squared_value = 1 - (rss/float(tss))
    return round(r_squared_value, 2)


## Adjusted R-squared value
def get_adjusted_r_squared_value(actuals, forecasted, features_size, flag=0):

    observations_size = len(actuals)
    ## Get the r squared value
    r_squared_value = get_r_squared_value(actuals, forecasted)
    numerator = (1 - r_squared_value) * (observations_size - 1)
    denominator = observations_size - features_size - 1
    adjusted_r_squared_value = round(1 - numerator/float(denominator), 2)

    if flag:
        return r_squared_value, adjusted_r_squared_value
    else:
        return adjusted_r_squared_value


## Template for modeling regression algorithm
def regression_model(data, features, target):

    regresser = LinearRegression().fit(data[features], data[target])
    return regresser


def main():

    ## Load dataset
    data = pd.read_csv(data_path)

    ## Creating features set
    features_set_1 = ['email campaign spend', 'google adwords spend']
    features_set_2 = ['google adwords spend', 'season']
    features_set_3 = ['email campaign spend', 'google adwords spend', 'season']
    target = 'sales'

    ## Building 3 models
    model_1 = regression_model(data, features_set_1, target)
    model_2 = regression_model(data, features_set_2, target)
    model_3 = regression_model(data, features_set_3, target)

    ## Prediction for the 3 models
    predictions_1 = model_1.predict(data[features_set_1])
    predictions_2 = model_2.predict(data[features_set_2])
    predictions_3 = model_3.predict(data[features_set_3])

    ## Capturing each model r-squared and adjsuted r-squared value
    r_sq_1, ad_r_sq_1 = get_adjusted_r_squared_value(
    data[target], predictions_1, len(features_set_1), flag=1)

    r_sq_2, ad_r_sq_2 = get_adjusted_r_squared_value(
    data[target], predictions_2, len(features_set_2), flag=1)

    r_sq_3, ad_r_sq_3 = get_adjusted_r_squared_value(
    data[target], predictions_3, len(features_set_3), flag=1)

    print("****************Model 01****************")
    print("\n R-Squared Value: {}, Adjusted R-Squared Value: {}".format(
    r_sq_1, ad_r_sq_1))

    print("\n****************Model 02****************")
    print("\n R-Squared Value: {}, Adjusted R-Squared Value: {}".format(
    r_sq_2, ad_r_sq_2))

    print("\n****************Model 03****************")
    print("\n R-Squared Value: {}, Adjusted R-Squared Value: {}".format(
    r_sq_3, ad_r_sq_3))


if __name__ == "__main__":
    main()


## Output
"""
****************Model 01****************

 R-Squared Value: 0.3, Adjusted R-Squared Value: 0.1

****************Model 02****************

 R-Squared Value: 0.09, Adjusted R-Squared Value: -0.17

****************Model 03****************

 R-Squared Value: 0.32, Adjusted R-Squared Value: -0.02
"""
