import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import numpy as np

from data.get_data import get_data
import utils


def hierarchical_clustering():
    """Performs hierarchical clustering of data, then shows alone anomaly data points and draws dendrogram.
    Dendrogram efficiently highlights all anomalies.
    Linkage matrix needs work to be able to show clustered anomalies."""

    data = get_data(num_normal_data=10, num_features=4, num_anomalies=1)

    # standardise data
    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(data)

    # Perform clustering
    linked = linkage(data_scaled, method="ward")

    # linkage matrix explanation:
    # for each row:
    #   index 0. cluster number 1 - can be an original data point (single cluster)
    #   index 1. cluster number 2
    #   index 2. distance between clusters
    #   index 3. number of original observations in the newly formed cluster

    # stats of distances
    distances = linked[:, 2]
    q3 = np.percentile(distances, 75)
    iqr = np.percentile(distances, 25) - q3

    # checks every distance to see if it's an outlier, then adds both clusters to anomalies
    anomaly_indices = []
    for item in linked:
        if item[2] > q3 - 1.5 * iqr:  # only uses upper quartile, as we are looking for distances on the outer extreme

            # filters out clusters not in original dataframe
            cluster1 = int(item[0])
            cluster2 = int(item[1])
            for cluster in [cluster1, cluster2]:
                if cluster < len(data):
                    anomaly_indices.append(cluster)

    utils.print_anomalies_from_indices(data, anomaly_indices)

    # plot dendrogram
    plt.figure(figsize=(8, 5))
    dendrogram(linked, labels=data.index.tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Datapoint index (name)")
    plt.ylabel("Distance")
    plt.show()


if __name__ == "__main__":
    hierarchical_clustering()
