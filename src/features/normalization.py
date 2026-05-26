import numpy as np


def minmax_norm(x):
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min == 0:
        return x

    return (x - x_min) / (x_max - x_min)


def zscore_norm(x):
    mean = x.mean()
    std = x.std()

    if std == 0:
        return x

    return (x - mean) / std