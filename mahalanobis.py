# code from https://www.geeksforgeeks.org/python/how-to-calculate-mahalanobis-distance-in-python

import numpy as np
from scipy.stats import chi2

from data.get_data import get_data


def mahalanobis():
    num_datapoints = 10
    df = get_data(num_datapoints, 3)

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
    df["p"] = 1 - chi2.cdf(df["Mahalanobis"], num_datapoints-1)

    print(df)
    print("The last datapoint has a lower p value, indicating an anomaly")


if __name__ == "__main__":
    mahalanobis()
