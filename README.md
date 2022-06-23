# Discrepancy Detection in Tabular Data

Several tests exist to quantify how well two data distributions match.
But if they don't match well, it's difficult to find prototypical examples of discrepancies, especially in high dimensions.
Our method performs a dimensionality reduction by a Self-Organzing Map, finds regions of significant differences in the maps for both data sets, and selects or aggregates samples from those regions as examples of over/under-represented features between the data sets.

![download-1](https://user-images.githubusercontent.com/1463403/175424123-6f06e22c-38d7-4535-b941-aa86ac5cc318.png)

The setup is simple:

```python


from shroom import Shroom

# load two samples (train and test) as numpy arrays
# ...

# make SOM based on the training data, apply to test data
# the default SOM uses best-guess parameters; optimization is possible but slow
m = Shroom(train, test)

# If you already have a SOM you want to work with
m = Shroom(train, test, som=som)

# find all groups of SOM cells with higher or lower density in test than in train
# significance and minimum cell number per group are configurable
groups_above, groups_below = m.find_discrepancies()

# show a over/under-dense group in data space
m.show(groups_above[0])

# combine all samples in `test` that fall in the cells of `group`
# optional: show the position of aggregate in data space
m.aggregate(groups_above[0], test, plot=True, labels=labels)

# select `N` examples from `test` that fall in the cells of `group`
m.pick(groups_above[0], test, N=10, method="sample")
```
