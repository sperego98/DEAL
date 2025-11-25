from ase.spacegroup import crystal
from ase.build import surface, molecule, add_adsorbate 
from ase.constraints import FixAtoms
import numpy as np

__all__ = ["build_surface_FeCo","add_molecules","set_magmom","fix_layers"]

def set_magmom(atoms):
    """Set magnetic moments for QE calculations""" 

    valence_magmom = {
        'Fe' : 16,
        'Co' : 17,
        'N'  :  0,
        'H'  :  0,
    }
    
    initial_magmom = 0.6
    atoms.set_initial_magnetic_moments([valence_magmom[sp]*0.6 for sp in atoms.symbols])

def fix_layers(atoms,layers, slab_elements = ['Fe','Co'],verbose=True):
    """
    Fix some layers of the slab. 

    If the argument `layer` is a list, then it will be interpreted as `from_layer` and `to_layer`.
    Otherwise, `from_layer` will be set to 0 and `to_layer` will be set to `layers` (fix bottom of the slab).

    This function do the following operations:
    - add a FixAtoms constraint (for ASE) and 
    - set the array 'fixed_atoms' (for LAMMPS) 
    """

    if isinstance(layers,list):
        from_layer,to_layer = layers[0],layers[1]+0.1 # add 0.1 to avoid rounding errors
    else:
        from_layer=0
        to_layer=layers+0.1

    # retrieve information about layers
    z_layer_max = np.max([atom.position[2] for atom in atoms if atom.symbol in slab_elements])
    z_layer_min = np.min([atom.position[2] for atom in atoms])
    atom_per_layer = len([1 for atom in atoms if atom.position[2]<(z_layer_min+0.05)])
    layer=int(len(atoms)/atom_per_layer)
    
    # create mask
    lower_threshold = z_layer_min + (z_layer_max-z_layer_min)/layer*(from_layer-0.5)
    upper_threshold = z_layer_min + (z_layer_max-z_layer_min)/layer*(to_layer-0.5)
    mask = [ (atom.position[2] > lower_threshold) and (atom.position[2] < upper_threshold) for atom in atoms]

    # add ASE constraint
    c = FixAtoms(mask = mask)
    atoms.set_constraint(c)

    # set array and info for lammps input
    atoms.set_array('fixed_atoms', np.asarray(mask,dtype=int))
    atoms.info['free'] = ' '.join([str(i+1) for i in np.argwhere( atoms.get_array('fixed_atoms') == 0 ) [:,0]])
    atoms.info['fixed'] = ' '.join([str(i+1) for i in np.argwhere( atoms.get_array('fixed_atoms') == 1 ) [:,0]])

    # print info
    if verbose:
        fixed_atoms = np.argwhere( mask )[:,0] +1
        free_atoms = np.argwhere( [not i for i in mask] )[:,0] +1
        print(f'ATOMS: {len(atoms)} ({len(fixed_atoms)} fixed - {len(free_atoms)} free)')
        print('Fixed atoms: ',fixed_atoms)
        print('Free atoms: ',free_atoms)

def build_surface_FeCo(miller_index=(1,1,0), layers=2, size=(1,1,1), vacuum = 0, a = 2.843, fixed_layers=[0,2]):
    """Create surface with ASE"""

    bulk_seed = crystal(('Fe', 'Co'),
                        basis=[(0., 0., 0.), (0.5, 0.5, 0.5)],
                        spacegroup=221,
                        cellpar=[a, a, a, 90, 90, 90],
                        size=(1,1,1)
                        )
    atoms = surface(bulk_seed, miller_index, layers=int(layers/2),vacuum=vacuum/2)
    # repeat
    atoms = atoms.repeat(size)
    # wrap
    atoms.wrap(eps=0.) 

    # shift cell to the bottom
    zmin = np.min(atoms.get_positions()[:,2])
    atoms.set_positions(atoms.get_positions()-[0,0,zmin])

    # add constraints
    if fixed_layers is not None:
        fix_layers(atoms, fixed_layers)
    
    # set pbc also along z direction 
    atoms.set_pbc([1,1,1])
    
    return atoms

def add_molecules(atoms, molecules, positions = None, height=1.5, seed=42, grid_size = None):
    """
    Add a list of adsorbate molecules to the surface, at a given height (either constant or adsorbate specific)
    """

    n_mol = len(molecules)
    if not isinstance(height, list):
        height = [height]*len(molecules)

    # create array of positions on a grid if not given
    if positions is None:
        if grid_size is None:
            grid_size = int(np.ceil(n_mol**0.5))
        grid_x = np.linspace(0,atoms.get_cell()[0,0],grid_size,endpoint=False)
        grid_y = np.linspace(0,atoms.get_cell()[1,1],grid_size,endpoint=False)
        positions = np.array(np.meshgrid(grid_x,grid_y)).T.reshape(-1,2)
        # shuffle
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(positions[1:])

    for i,molname in enumerate(molecules):
        m = molecule(molname)
        m.rotate(180,'x')
        add_adsorbate(atoms, m, height=height[i], position = positions[i] )
        
    # rattle free atoms 
    atoms.rattle()
    atoms.wrap()

    # set magnetic moments
    set_magmom(atoms)

    return atoms