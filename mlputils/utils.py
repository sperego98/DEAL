import numpy as np

__all__ = ["average_along_cv","compute_histogram","plumed_to_pandas"]

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
    
import pandas as pd 

def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=" ",
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df
    