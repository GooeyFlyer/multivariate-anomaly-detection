#

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

from data.get_data import get_data


def hierarchical_clustering():

    data = get_data(num_datapoints=60, num_features=3)

    # standardise data
    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(data)

    # Perform clustering
    linked = linkage(data_scaled, method="ward")

    print(data)

    # plot dendrogram
    plt.figure(figsize=(8, 5))
    dendrogram(linked, labels=data.index.tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Datapoint index (name)")
    plt.ylabel("Distance")
    plt.show()


if __name__ == "__main__":
    hierarchical_clustering()
