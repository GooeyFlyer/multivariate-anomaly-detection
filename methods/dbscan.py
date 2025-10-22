# code from https://www.reneshbedre.com/blog/dbscan-python.html

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
# from sklearn.datasets import make_blobs

from data.get_data import preprocess_data, get_data, pd
import sys


def epsilon_graph(data: pd.DataFrame) -> None:
    """Use the knee of this line as the epsilon for DBSCAN"""
    nbrs = NearestNeighbors(n_neighbors = 5).fit(data)
    neigh_dist, neigh_ind = nbrs.kneighbors(data)
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 4]

    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (4th NN)")
    plt.show()


def dbscan(file_path: str):
    """Clusters data using DBSCAN, then shows alone anomaly data points and draws scatter graph labeling anomalies.
    Has the chance of clustering (therefore not highlighting) anomalies"""

    # example does not use multivariate data to make it easier to visualise clusters
    data = preprocess_data(pd.read_csv(file_path, sep=";"))

    epsilon_graph(data)

    # building model
    # numpy array of all the clustering labels assigned to each data point
    # min_samples should almost always be 2 * no. of features
    # epsilon can be found by running epsilon_graph() and using the approximate knee of the graph
    db_default = DBSCAN(eps=8, min_samples=2*data.shape[1]).fit(data)
    labels = db_default.labels_
    # print(labels)

    print(len(set(labels)))
    labels[labels > 0] = 0

    p = sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue=labels, legend="full", palette="deep")
    sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1), title="Clusters")
    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    dbscan(file_path)
