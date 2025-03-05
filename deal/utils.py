import numpy as np
from ase.data import covalent_radii as CR
from ase.data import chemical_symbols as CS
from ase.io import read
from ase import Atoms

import os

__all__ = ["average_along_cv","compute_histogram","plumed_to_pandas","paletteFessa","create_chemiscope_input"]

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
    
##########################################################################
## FESSA COLOR PALETTE
##########################################################################
#  https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py

from matplotlib.colors import LinearSegmentedColormap, ColorConverter
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fessa colormap
paletteFessa = [
    "#1F3B73",  # dark-blue
    "#2F9294",  # green-blue
    "#50B28D",  # green
    "#A7D655",  # pisello
    "#FFE03E",  # yellow
    "#FFA955",  # orange
    "#D6573B",  # red
]

cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa)
mpl.colormaps.register(cmap=cm_fessa)
mpl.colormaps.register(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f"fessa{i}"] = paletteFessa[i]

### To set it as default
# import fessa
#plt.set_cmap('fessa')
### or the reversed one
#plt.set_cmap('fessa_r')
### For contour plots
# plt.contourf(X, Y, Z, cmap='fessa')
### For standard plots
# plt.plot(x, y, color='fessa0')

def create_chemiscope_input(trajectory, filename = None, colvar = None, cvs=['*']):
    """
    Create a chemiscope input file from a trajectory and optional collective variables (colvar) file.

    Parameters
    ----------
    trajectory : list of ase.Atoms or str
        Trajectory of atoms objects or path to an xyz file
    filename : str, optional
        Output filename. If None, it will be saved with the same name of the trajectory with _chemiscope.json.gz appended.
    colvar : str or pandas.DataFrame, optional
        Path of the COLVAR file or a pandas dataframe
    cvs : list of str, optional
        List of collective variable names to be saved into the chemiscope file. If a string contains '*', it will be used as a filter for the property names in the colvar file (e.g. 'cv.*' will extract all properties with 'cv.' prefix). Default is ['*']

    Returns
    -------
    filename : str
        Path of the chemiscope input file
    """

    try: 
        import chemiscope
    except ImportError:
        raise ImportError("Chemiscope is not installed. Please install it with pip install chemiscope")
        
    # check if trajectory is a list of atoms or a filename
    if isinstance(trajectory,list) & isinstance(trajectory[0],Atoms):
        traj = trajectory
    elif isinstance(trajectory,str):
        print('[INFO] Reading file:',trajectory)
        traj = read(trajectory,index=':')
    atoms = traj[0]

    # load colvar file into traj if requested
    if colvar is not None:
        if isinstance(colvar,str):
            colvar = plumed_to_pandas(colvar)
        elif isinstance(colvar,pd.DataFrame):
            pass
        else:
            raise TypeError("colvar must be a string filename or a pandas dataframe")
        assert len(colvar) == len(traj), "colvar and trajectory must have the same number of frames"

        for i,atoms in enumerate(traj):
            for col in colvar.columns:
                atoms.info['colvar.'+col] = colvar[col].iloc[i]

        print('[INFO] COLVAR info saved in trajectory.')

    # Get CV names
    prop_names, prop_names_float = [],[]
    for c in cvs:
        if '*' in c:
            prop_names.extend([p for p in atoms.info.keys() if c.replace('*','') in p ])
        else:
            prop_names.append(c)

    # Check if CV names can be converted to float
    for p in prop_names:
        try:
            float(atoms.info[p])
            prop_names_float.append(p)
        except TypeError:
            print(f'skipping "{p}" as it cannot be converted to float.')

    print('[INFO] CV names:',prop_names_float)

    # Extract properties
    properties = chemiscope.extract_properties(traj, only=prop_names_float)

    # Define shape and colors
    shapes_selection = []
    for atoms in traj:
        target_atoms = atoms.info.get('target_atoms', None)
        for i,atom in enumerate(atoms):
            if target_atoms is not None:
                if not isinstance(target_atoms,np.ndarray):
                    target_atoms = np.asarray([target_atoms])
                shapes_selection.append({"radius": CR[CS.index(atom.symbol)], "color": None if i in target_atoms else '#d4d4d4' })
            else:
                shapes_selection.append({"radius": CR[CS.index(atom.symbol)], "color": None })
    if target_atoms is not None:
        print('[INFO] "target_atoms" found in atoms.')

    # Write input
    if filename is None:
        if isinstance(trajectory,str):
            filename = os.path.splitext(trajectory)[0]+'_chemiscope.json.gz'
        else: 
            filename = 'chemiscope.json.gz'

    chemiscope.write_input(
        filename,
        frames=traj,
        properties=properties,
        meta=dict(name="DEAL selection"),
        shapes = { "selection": {
            "kind": "sphere",
            "parameters": {"atom": shapes_selection}
            },
        },
        settings={ 'structure': [{               
                    "atoms": False,
                    "bonds": False,
                    "shape": "selection",
                    "axes": "off",
                    "keepOrientation": False,
                    "playbackDelay": 700,
                    }]
                }
    )

    print('[INFO] Chemiscope input saved in:',filename)

    return filename