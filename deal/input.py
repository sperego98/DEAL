import yaml
import os
from ase import Atoms
from ase.io import read,write
from pathlib import Path
import numpy as np

__all__ = ["create_deal_input"]

def create_deal_input(trajectory,
                      folder='./deal/',
                      config_path='./configs/deal.yaml',
                      threshold=0.1,
                      update_threshold=None,
                      cutoff=5.,
                      cutoff_function='cosine',
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
    update_threshold : float, optional (default=None)
        When a new configuration is found, all local environments with threshold >= update_threshold will be added to the GP. By default it is 0.8*threshold
    cutoff : float or dict, optional (default=5.)
        Cutoff for the descriptors. Can be a single value or a dictionary where the keys are tuples of atomic numbers
    cutoff_function : str, optional (default='quadratic')
        Cutoff function for the descriptors (quadratic, cosine, etc.)
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
        filename = 'All_Data.xyz'
        traj = trajectory
        write(folder+filename,traj)
    # elif is a path, check that exists and copy eventually it inside folder
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
            # create symbolic link
            try:
                orig_filename = os.path.relpath(trajectory, folder)
                os.symlink(orig_filename,folder+'All_Data.xyz')
            except FileExistsError as e:
                print(e)
            filename = 'All_Data.xyz' 

    # get species
    if atomic_numbers is None:
        # get atomic numbers from the full trajectory
        atomic_numbers = [set(atoms.get_atomic_numbers()) for atoms in traj]
        atomic_numbers = sorted(list(set().union(*atomic_numbers)))

    # build cutoff matrix 
    if isinstance(cutoff,dict):
        cutoff_matrix = [[cutoff[i][j] for j in atomic_numbers] for i in atomic_numbers]
        max_cutoff = float(np.asarray(cutoff_matrix).max())
    else:
        cutoff_matrix = None
        max_cutoff = cutoff

    # check that update_threshold is not higher than threshold
    if update_threshold is None:
        update_threshold = 0.8*threshold
    elif update_threshold > threshold:
        raise ValueError(f'update_threshold ({update_threshold}) must be lower than threshold ({threshold})')

    # load default configs
    with open(config_path) as file:
        config = (yaml.load(file, Loader=yaml.FullLoader))

    # update config file
    section = 'supercell'
    if section in config:
        config[section]['file'] = filename
    else:
        section = 'input_data'
        config[section]['filenames'] = [ filename ]    

    section = 'flare_calc'
    if pretrain is not None:
        config[section]['file'] = pretrain
    else:
        config[section]['species'] = [int(i) for i in atomic_numbers]
        if cutoff_matrix is not None:
            config[section]['descriptors'][0]['cutoff_matrix'] = cutoff_matrix
        config[section]['descriptors'][0]['cutoff_function'] = cutoff_function
        config[section]['cutoff'] = max_cutoff

    section = 'otf'
    config[section]['std_tolerance_factor'] = -1*threshold
    config[section]['update_threshold'] = update_threshold

    # write config to file
    with open(f'{folder}input.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)


