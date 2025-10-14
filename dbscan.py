import matplotlib.pyplot as plt
from data.get_data import get_data
import seaborn as sns
from sklearn.cluster import DBSCAN


def dbscan():
    # example does not use multivariate data to make it easier to visualise clusters
    data = get_data(500, 2)

    # building model
    # numpy array of all the clustering labels assigned to each data point
    db_default = DBSCAN(eps=1, min_samples=1).fit(data)
    labels = db_default.labels_
    # print(labels)

    p = sns.scatterplot(data=data, x="Feature1", y="Feature2", hue=db_default.labels_, legend="full", palette="deep")
    sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1), title="Clusters")
    plt.show()


if __name__ == "__main__":
    dbscan()
