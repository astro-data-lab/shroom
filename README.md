# Mismatch Detection in Tabular Data

Several tests exist to quantify how well two data distributions match.
But if they don't match well, it's difficult in high-dimensional spaces to find prototypical examples of discrepant samples.
Our method performs a dimensionality reduction by a Self-Organzing Map, finds regions of significant differences in the maps for both data sets, and aggregates samples from those regions as examples of over/under-represented features between the data sets.

The setup is simple:

```python

from shroom import get_som, analyze

# load two samples (train and test) as numpy arrays
# ...

# get SOM based on the training data
# the default uses best-guess parameters of the SOM
# optimization is possible but slow
som = get_som(train, optimize=False)

# make frequency maps for train and test as well as their difference
# connect regions with significant differences and aggregate samples therein
# set plot=True for visualization (labels correspond to names of fields in train/test)
aggregates_above, aggregates_below = analyze(som, train, test, plot=True, labels=labels)
```
