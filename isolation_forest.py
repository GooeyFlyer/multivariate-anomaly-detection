# code from https://www.geeksforgeeks.org/machine-learning/what-is-isolation-forest/
# uses multivariate data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def isolation_forest():

    # read csv file
    # standardise features, excluding target variable 'Class' using StandardScalar
    credit_data = pd.read_csv("data/creditcard.csv", nrows=40000)
    scalar = StandardScaler().fit_transform(credit_data.loc[:,credit_data.columns!='Class'])
    scaled_data = scalar[0:40000]
    df = pd.DataFrame(data=scaled_data)
    X = credit_data.drop(columns=['Class'])
    y = credit_data['Class']

    # calculate fraction of outliers by looking at the number of fraudulent transactions
    outlier_fraction = len(credit_data[credit_data['Class'] == 1])/float(len(credit_data[credit_data['Class'] == 0]))

    # create and fit isolation forest model with outlier fraction
    # n_estimators sets no. of base estimators (tress) in the ensemble which helps to improve robustness and accuracy
    # random_state used for reproducibility which ensures that the results are consistent across different runs
    model = IsolationForest(n_estimators=100, contamination=outlier_fraction, random_state=42)
    model.fit(df)

    # calculate accuracy based on anomaly scores
    scores_prediction = model.decision_function(df)
    y_pred = model.predict(df)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    print("Accuracy in finding anomaly:", accuracy_score(y,y_pred))

    # plot Amount feature to visualise distinction between normal and outlier
    # Amount can be changed for any other feature to visualise results
    y_feature = credit_data['Amount']
    credit_data['predicted_class'] = y_pred

    plt.figure(figsize=(7,4))
    sns.scatterplot(x=credit_data.index, y=y_feature, hue=credit_data['predicted_class'], palette={0: 'blue', 1: 'red'}, s=50)
    plt.title('Visualization of Normal vs Anomalous Transactions')
    plt.xlabel('Data points')
    plt.ylabel(y_feature.name)
    plt.legend(title='Predicted Class', loc='best')
    plt.show()


if __name__ == "__main__":
    isolation_forest()
