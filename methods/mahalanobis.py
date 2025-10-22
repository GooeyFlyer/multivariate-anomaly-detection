# code adapted from https://www.geeksforgeeks.org/python/how-to-calculate-mahalanobis-distance-in-python
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

from data.get_data import *
from pot import pot_1D


def mahalanobis(file_path: str):
    """Creates dataframe with mahalanobis distance for each data point.
    Anomalies can be seen by looking at high mahalanobis distances
    Also creates p value, but it doesn't work and I don't really understand it"""

    df = preprocess_data(pd.read_csv(file_path, sep=";"))

    # calculate mahalanobis distance
    y_mu = df - np.mean(df)
    cov = np.cov(df.values.T)
    inv_cov_matrix = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_cov_matrix)
    mahal = np.dot(left, y_mu.T)

    df["Mahalanobis"] = mahal.diagonal()

    # calculate p value for each mahalanobis distance
    # used for statistical significance
    # same as the p-value that belongs to the Chi-Square statistic of the Mahalanobis distance
    # having degrees of freedom equal to k-1 where k = number of variables
    df["p"] = 1 - chi2.cdf(df["Mahalanobis"], len(df)-1)

    print(df["Mahalanobis"])
    pot_1D(df["Mahalanobis"].to_numpy(), 0.99)


if __name__ == "__main__":
    file_path = sys.argv[1]
    mahalanobis(file_path)
