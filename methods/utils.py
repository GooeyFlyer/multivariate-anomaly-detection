import pandas as pd


def print_anomalies_from_indices(data: pd.DataFrame, anomaly_indices: list[int]) -> None:
    print("Anomalies:")
    for index in anomaly_indices:
        print(str(data.iloc[index]) + "\n")
