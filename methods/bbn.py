# code from https://medium.com/data-science-collective/the-starters-guide-to-casual-structure-learning-with-bayesian-methods-in-python-e3b90f49c99c
import random

import bnlearn as bn
import numpy as np


def bbn_basic():

    df = bn.import_example('sprinkler')

    # DAG describes structure of the data
    auto_DAG = bn.structure_learning.fit(df)  # create DAG from data

    edges = [("Cloudy", "Sprinkler"),
             ("Cloudy", "Rain"),
             ("Sprinkler", "Wet_Grass"),
             ("Rain", "Wet_Grass")]

    custom_DAG = bn.make_DAG(edges)  # make DAG from defined casual dependencies

    print(custom_DAG["adjmat"])  # adjacency matrix
    G = bn.plot(custom_DAG)

    # Conditional Probability Table (CPT) describes statistical relationship between each node and its parents.
    model = bn.parameter_learning.fit(custom_DAG, df, methodtype="ml")  # using maximum likelihood, can use bayes
    bn.print_CPD(model)

    # probability of rain, given sprinkler is False and cloudy is True
    q = bn.inference.fit(model, variables=["Rain"], evidence={"Sprinkler": 0, "Cloudy": 1})


def bbn_predict():

    df = bn.import_example("asia")

    edges = [("smoke", "lung"),
             ("smoke", "bronc"),
             ("lung", "xray"),
             ("bronc", "xray")]

    DAG = bn.make_DAG(edges)
    # bn.plot(DAG)

    model = bn.parameter_learning.fit(DAG, df, verbose=3)

    # create some data based on learned model
    Xtest = bn.sampling(model, n=1000)

    # single inference
    print("inference query:")
    query = bn.inference.fit(DAG, variables=["bronc", "xray"], evidence={"smoke": 1, "lung": 1}, verbose=3)

    print("\n----\n")

    # predict function
    # easily access df for specified variables, without providing evidence
    # inference on dataset performed sample-wise by using all nodes as evidence (except nodes we are predicting)
    # states with the highest probability are returned
    print("prediction:")
    pout = bn.predict(model, Xtest, variables=["bronc", "xray"])
    print(pout)


def bbn_anomaly_detection():
    """An example of anomaly detection for labelled data (data marked as anomalous or not)"""

    df = bn.import_example("sprinkler")

    # add anomalous scenarios
    # df columns: Cloudy, Sprinkler, Rain, Wet_Grass
    num_anomalous_data = 60  # number of anomalous data for each anomaly case
    for x in range(num_anomalous_data):
        df.loc[len(df)] = [random.choice([0,1]), 0, 0, 1]
    for x in range(num_anomalous_data//3):
        df.loc[len(df)] = [random.choice([0,1]), 1, 0, 0]
    for x in range(num_anomalous_data//3):
        df.loc[len(df)] = [random.choice([0,1]), 0, 1, 0]
    for x in range(num_anomalous_data//3):
        df.loc[len(df)] = [random.choice([0,1]), 1, 1, 0]

    def label_anomaly(r):
        if (r["Rain"] == 0) and (r["Sprinkler"] == 0) and (r["Wet_Grass"] == 1):
            return 1

        elif ((r["Rain"] == 1) or (r["Sprinkler"] == 1)) and (r["Wet_Grass"] == 0):
            return 1

        return 0

    # sets anomaly to be 1 on anomalous scenarios
    df["Anomaly"] = df.apply(label_anomaly, axis=1)
    print(df[df["Anomaly"] == 1])  # filter df to show anomaly data correctly added

    edges = [("Cloudy", "Sprinkler"),
             ("Cloudy", "Rain"),
             ("Sprinkler", "Wet_Grass"),
             ("Rain", "Wet_Grass"),
             ("Wet_Grass", "Anomaly"),
             ("Rain", "Anomaly"),
             ("Sprinkler", "Anomaly")]

    DAG = bn.make_DAG(edges)
    # bn.plot(DAG)

    model = bn.parameter_learning.fit(DAG, df, verbose=3)

    # Xtest = bn.sampling(model, n=1000)
    # print("\nprediction:")
    # pout = bn.predict(model, Xtest, variables=["Wet_Grass", "Anomaly"])
    # print(print(pout[pout["Anomaly"] == 1]))

    print("\nRain but no Wet_Grass:")
    q1 = bn.inference.fit(model, variables=["Anomaly"], evidence={"Sprinkler": 0, "Rain": 1, "Wet_Grass": 0})

    print("\nWet_Grass with no rain or sprinkler:")
    q2 = bn.inference.fit(model, variables=["Anomaly"], evidence={"Sprinkler": 0, "Rain": 0, "Wet_Grass": 1})


if __name__ == "__main__":
    bbn_anomaly_detection()
