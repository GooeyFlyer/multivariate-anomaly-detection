# Basic implementations of tools for multivariate anomaly detection

## Methods using multivariate data:
- GRU (Gated Recurrent Units) - timeseries data, temperature_data.csv available from link at top of python code
- Mahalanobis Distance - uses data.get_data()
- dbscan - example uses 2D data for better visualisation, uses data.get_data()
- isolation forest - creditcard.csv available from link at top of python code
- hierarchical_clustering - uses data.get_data()
- BBN (Bayes Belief Network) - uses data from bnlearn package

## Methods using only 1D data:
- POT (Peak Over Threshold)