import numpy as np

__all__ = ["average_along_cv","compute_histogram"]

def average_along_cv(value, cv, bins):
    h1,x = np.histogram(cv,bins=bins,weights=value)
    h2,_ = np.histogram(cv,bins=bins)

    mean = h1/h2
    x = (x[:-1]+x[1:])/2

    return x,mean

def compute_histogram(value, bins, threshold = None):
    h1,x = np.histogram(value,bins=bins)
    x = (x[:-1]+x[1:])/2

    if threshold is not None:
        h2,_ = np.histogram(value[(value>threshold)],bins=bins)
        return x,(h1,h2)
    else:
        return x,h1
    