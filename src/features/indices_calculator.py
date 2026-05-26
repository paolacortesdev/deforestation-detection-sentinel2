import numpy as np


def ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)


def nbr(nir, swir):
    return (nir - swir) / (nir + swir + 1e-8)