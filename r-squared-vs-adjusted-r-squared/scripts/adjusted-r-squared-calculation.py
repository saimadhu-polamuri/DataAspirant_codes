"""
===============================================
Objective: R-squared Vs Adjusted R-squared comparision
Sub Objective: Calculating Adjusted R-squared
Author: Saimadhu.Polamuri
Blog: https://dataaspirant.com
Date: 2020-10-14
===============================================
"""

## Requried Python Packages
import pandas as pd
import numpy as np


## Paths
data_path = "../data/sales_data.csv"

## UDF's Functions

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

def get_adjusted_r_squared_value(actuals, forecasted, features_size):

    observations_size = len(actuals)
    ## Get the r squared value
    r_squared_value = get_r_squared_value(actuals, forecasted)
    numerator = (1 - r_squared_value) * (observations_size - 1)
    denominator = observations_size - features_size - 1

    return 1 - numerator/float(denominator)


def main():

    ## Load dataset
    data = pd.read_csv(data_path)
    # print(data.head())

    ## Calculating residual squared value
    rss = rss_value(data["sales"], data["dummy_forecasted_sales"])
    print("Calculated residual sum of squares :: {}".format(rss))

    ## Calculating total squared value
    tss = tss_value(data["sales"])
    print("Calculated total sum of squares value :: {}".format(tss))

    ## Calculating R-Squared value
    r_squared_value = get_r_squared_value(data["sales"],
    data["dummy_forecasted_sales"])
    print("Calculated R Squared Value :: {}".format(r_squared_value))

    ## Calculating Adjusted R-Squared value
    features_size = 3
    adjusted_r_squared_value = get_adjusted_r_squared_value(data["sales"],
    data["dummy_forecasted_sales"], features_size)
    print("Calculated Adjusted R Squared Value :: {}".format(
    adjusted_r_squared_value))


if __name__ == "__main__":
    main()


## Output
"""
Calculated residual sum of squares :: 189
Calculated total sum of squares value :: 1704.4
Calculated R Squared Value :: 0.89
Calculated Adjusted R Squared Value :: 0.835
"""

## dataaspirant-adjusted-r-squared-calculation.py
