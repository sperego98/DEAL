import numpy as np

__all__ = ["filter_by_uncertainty", "filter_by_cv", "sort_traj"]

def filter_by_uncertainty(traj,uncertainty,threshold,max_threshold=0.5):
    """
    Filter atoms objects in `traj` by uncertainty.

    Parameters
    ----------
    traj : list
        List of atoms objects
    uncertainty : array_like
        Uncertainty of each frame in `traj`
    threshold : float
        Min uncertainty to include
    max_threshold : float, optional
        Max uncertainty to include. Default is 0.5

    Returns
    -------
    filtered_traj : list
        List of atoms objects with uncertainty > `threshold`
        and uncertainty < `max_threshold`
    """
    sel = (uncertainty>threshold) & (uncertainty<max_threshold)
    return [atoms for s,atoms in zip(sel,traj) if s]

def filter_by_cv(traj,cv,bins,max_samples_per_bin=1000, verbose=True):
    """
    Filter atoms objects in `traj` by binning in collective variable `cv`.

    Parameters
    ----------
    traj : list
        List of atoms objects
    cv : array_like
        Collective variable for each frame in `traj`
    bins : array_like
        Bins for CV
    max_samples_per_bin : int, optional
        Maximum number of samples to keep in each bin. Default is 1000
    verbose : bool, optional
        Print out the distribution of CV. Default is True

    Returns
    -------
    filtered_traj : list
        List of atoms objects, with `max_samples_per_bin` samples per bin
    """
    indexes = []
    if verbose: 
        print( 'CV distribution' )
    for i in range(len(bins)-1):
        sel = (cv>bins[i]) & (cv<bins[i+1])
        index = np.argwhere(sel).ravel()
        
        if len(index) > max_samples_per_bin:
            index = np.random.choice(index,max_samples_per_bin,replace=False)

        if verbose:
            print(f'{bins[i]:.2f} < cv < {bins[i+1]:.2f} : {np.sum(sel)} --> {len(index)}')

        indexes.extend(index)

    return [traj[i] for i in indexes]

def sort_traj(traj,mode,seed=None, uncertainty=None,return_idx=False):
    """
    Sort the trajectory `traj` according to `mode`.

    Parameters
    ----------
    traj : list
        List of atoms objects
    mode : str
        Sorting mode. Options are:
            - None: No sorting
            - 'shuffle': Shuffle the trajectory
            - 'uncertainty': Sort by uncertainty in descending order
    seed : int, optional
        Random seed for shuffling. Only used if mode='random'
    uncertainty : array_like, optional
        Uncertainty for each frame in `traj`. Only used if mode='uncertainty'
    return_idx : bool, optional
        If True, return the index of the sorted trajectory. Default is False

    Returns
    -------
    sorted_traj : list
        Sorted list of atoms objects
    """
    idx = np.arange(len(traj))
    
    if mode == "shuffle":
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx)

    elif mode == "uncertainty":
        if uncertainty is None:
            raise ValueError('Must provide uncertainty')
        idx = np.argsort(uncertainty)[::-1]
    
    traj = [traj[i] for i in idx]

    if return_idx:
        return traj, idx
    else:
        return traj

