# Code from https://stackoverflow.com/questions/60401716/finding-peaks-above-threshold
# Does not use multivariate data

from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


def pot_1D(x: np.ndarray, threshold_multiplier: float):
    """Creates graph showing peaks in an array
    Doesn't use multivariate data."""

    # find_peaks is POT algorithm
    peaks_indices = find_peaks(x)[0]

    peaks = np.array(list(zip(peaks_indices, x[peaks_indices])))

    # for multivariate data, we need a way of finding a threshold
    threshold = threshold_multiplier * max(x[peaks_indices])

    filtered_peaks_indices = [index for index, value in peaks if value > threshold]
    filtered_peaks_values = [value for index, value in peaks if value > threshold]

    plt.plot(range(len(x)), x)
    plt.ylabel("Value")
    plt.xlabel("Datapoint index")

    plt.axhline(threshold, color="r")
    plt.scatter(filtered_peaks_indices, filtered_peaks_values, s=200, color="r")
    plt.show()


if __name__ == "__main__":
    pot_1D(np.array([1,2,3,2,1,2,3,2,1,2,3,4,3,2,1,2,3,4,7,4,3,2,1]), 0.5)
