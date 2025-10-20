import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

from data.get_data import get_data
import utils


def dbscan():
    """Clusters data using DBSCAN, then shows alone anomaly data points and draws scatter graph labeling anomalies.
    Has the chance of clustering (therefore not highlighting) anomalies"""

    # example does not use multivariate data to make it easier to visualise clusters
    data = get_data(500, 2, 5)

    # building model
    # numpy array of all the clustering labels assigned to each data point
    db_default = DBSCAN(eps=1.0, min_samples=2).fit(data)
    labels = db_default.labels_
    # print(labels)

    anomaly_indices = []
    for index in range(len(labels)):
        if labels[index] == -1:
            anomaly_indices.append(index)

    utils.print_anomalies_from_indices(data, anomaly_indices)

    p = sns.scatterplot(data=data, x="Feature1", y="Feature2", hue=db_default.labels_, legend="full", palette="deep")
    sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1), title="Clusters")
    plt.show()


if __name__ == "__main__":
    dbscan()
