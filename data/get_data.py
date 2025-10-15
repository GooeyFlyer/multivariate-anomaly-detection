import random
import pandas as pd


def get_data(num_datapoints: int, num_features: int) -> pd.DataFrame:
    """Creates pandas Dataframe with random values 1-20, then appends anomaly (with value 40) as last data point"""

    # remember variable names are the features of a datapoint. Each index in the lists is a datapoint
    # random data then append datapoint with anomaly values
    data = {}
    for num in range(1, num_features+1):
        data[f"Feature{str(num)}"] = [random.randint(1, 20) for _ in range(num_datapoints)] + [40]

    return pd.DataFrame(data)


if __name__ == "__main__":
    print(get_data(10, 3))
