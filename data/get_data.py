import random
import pandas as pd
import os


def get_data(num_normal_data: int, num_features: int, num_anomalies: int = 1) -> pd.DataFrame:
    """Creates pandas Dataframe with num_normal_data data points with random values 1-20,
    and num_anomalies (default 1) data points with random values 35-45"""

    # remember variable names are the features of a datapoint. Each index in the lists is a datapoint
    # random data then append datapoint with anomaly values
    data = {}
    for num in range(1, num_features+1):
        data[f"Feature{str(num)}"] = [random.randint(1, 20) for _ in range(num_normal_data)] + [random.randint(35, 45) for _ in range(num_anomalies)]

    return pd.DataFrame(data)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drops Date_Time column and formats data
    Returns:
        data (pd.DataFrame)"""
    data = data.drop(columns="Date_Time", axis=1)

    for col in data.columns:
        m_neg = data[col].str.startswith("-")
        data[col] = data[col].str.strip("-")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "."))
        data.loc[m_neg, col] += -1

    print(data.shape)

    return data


if __name__ == "__main__":

    # print(get_data(10, 3, 2))

    print(preprocess_data(load_csv_file("FeatureDataSel.csv")).head())
