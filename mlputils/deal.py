import numpy as np
import yaml
import os
from ase import Atoms
from ase.io import read,write
from pathlib import Path

__all__ = ["filter_by_uncertainty", "filter_by_cv", "sort_traj", "create_deal_input"]

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

def sort_traj(traj,mode,seed=None, uncertainty=None):
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

    Returns
    -------
    sorted_traj : list
        Sorted list of atoms objects
    """
    if mode == None:
        return traj
    
    if mode == "shuffle":
        if seed is not None:
            np.random.seed(seed)
        idx = np.arange(len(traj))
        np.random.shuffle(idx)

    elif mode == "uncertainty":
        if uncertainty is None:
            raise ValueError('Must provide uncertainty')
        idx = np.argsort(uncertainty)[::-1]
    
    traj = [traj[i] for i in idx]

    return traj

def create_deal_input(trajectory,
                      folder='./deal/',
                      config_path='./configs/deal.yaml',
                      threshold=0.1,
                      update_threshold=0.05,
                      cutoff=5.,
                      atomic_numbers=None,
                      pretrain=None,
                      copy_traj=True):
    """
    Create input files for DEAL.

    Parameters
    ----------
    trajectory : list of ase.Atoms or str
        Trajectory of atoms objects or path to an xyz file
    folder : str, optional (default='./deal/')
        Folder where the input files will be created
    config_path : str, optional (default='./configs/deal.yaml')
        Path of the configs file 
    threshold : float, optional (default=0.1)
        Uncertainty threshold for selecting the configuration (whenever the GP uncertainty on one of the local environments is >= threshold the configuration will be added to the GP)
    update_threshold : float, optional (default=0.05)
        When a new configuration is found, all local environments with threshold >= update_threshold will be added to the GP
    cutoff : float or dict, optional (default=5.)
        Cutoff for the descriptors. Can be a single value or a dictionary where the keys are tuples of atomic numbers
    atomic_numbers : list of int, optional
        List of atomic numbers to consider. If None, it will be extracted from the trajectory
    pretrain : str, optional
        Path of the pre-trained model. If None, a new GP will be trained with the given cutoff
    copy_traj : bool, optional
        If True, copy the trajectory to the folder. If False, only the path will be modified in the input file. Considered only if trajectory is a path to an xyz file

    Returns
    -------
    None
    """
    # create folder if it does not exist
    Path(folder).mkdir(parents=True, exist_ok=True)

    # if traj is a list of atoms, save it to traj-selection.xyz inside folder
    if isinstance(trajectory,list) & isinstance(trajectory[0],Atoms):
        filename = 'traj-selection.xyz'
        traj = trajectory
        write(folder+filename,traj)
    # elif is a path, check that exists and copy eventually  it inside folder
    elif isinstance(trajectory,str):
        traj = read(trajectory,index=':')
        # check if the trajectory is in xyz format
        if os.path.splitext(trajectory)[-1] != '.xyz':
            copy_traj = True
        if copy_traj:
            # write to xyz
            filename = os.path.basename(trajectory)
            write(folder+filename,traj)
        elif not copy_traj:
            # check if trajectory is absolute path if not make it relative to folder
            if not os.path.isabs(trajectory):
                filename = os.path.relpath(trajectory, folder)

    # get species
    if atomic_numbers is None:
        atomic_numbers = sorted(list(set(traj[0].get_atomic_numbers())))

    # build cutoff matrix 
    if isinstance(cutoff,dict):
        cutoff_matrix = [[cutoff[i][j] for j in atomic_numbers] for i in atomic_numbers]
        max_cutoff = float(np.asarray(cutoff_matrix).max())
    else:
        cutoff_matrix = [[cutoff for _ in atomic_numbers] for i in atomic_numbers]
        max_cutoff = cutoff

    # check that update_threshold is not higher than threshold
    if update_threshold > threshold:
        raise ValueError(f'update_threshold ({update_threshold}) must be lower than threshold ({threshold})')

    # load default configs
    with open(config_path) as file:
        config = (yaml.load(file, Loader=yaml.FullLoader))

    # update config file
    section = 'supercell'
    config[section]['file'] = filename

    section = 'flare_calc'
    if pretrain is not None:
        config[section]['file'] = pretrain
    else:
        config[section]['species'] = [int(i) for i in atomic_numbers]
        config[section]['single_atom_energies'] = [0 for _ in atomic_numbers]
        config[section]['descriptors'][0]['cutoff_matrix'] = cutoff_matrix
        config[section]['descriptors'][0]['cutoff_function'] = 'quadratic'
        config[section]['cutoff'] = max_cutoff

    section = 'otf'
    config[section]['md_kwargs']['filenames'] = [ filename ]
    config[section]['std_tolerance_factor'] = -1*threshold
    config[section]['update_threshold'] = update_threshold

    # write config to file
    with open(f'{folder}input-deal.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)