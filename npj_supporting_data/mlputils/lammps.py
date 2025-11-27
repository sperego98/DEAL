from ase.data import chemical_symbols, atomic_masses
from ase.io.lammpsdata import write_lammps_data
from ase.constraints import FixAtoms
import numpy as np

__all__ = ["LAMMPS_input"]

class LAMMPS_input:
    def __init__(self, atoms, project_name = None ):

        # save atoms
        """
        Initialize LAMMPS input generator.

        Then, each time a method of the class is called, the corresponding input will be appended to `input`.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to be written to LAMMPS input file.
        project_name : str (optional)
            Project name to be used in LAMMPS input file. If not specified,
            the chemical formula of the Atoms object will be used.

        Attributes
        ----------
        name : str
            Project name.
        input : list
            List of LAMMPS commands to be written to input file.
        """
        self.atoms = atoms.copy()
        if project_name is not None:
            self.name = project_name
        else:
            self.name = str(atoms.symbols)

        # init dictionary
        self.input = []

    def get_masses(self, specorder = None):

        symbols = self.atoms.get_chemical_symbols()
        # If unspecified default to atom types in alphabetic order
        self.specorder = specorder if specorder is not None else sorted(set(symbols))

        self.masses = []
        for type_id, specie in enumerate(self.specorder):
            mass = atomic_masses[chemical_symbols.index(specie)]
            self.masses += [ f"{type_id + 1:d} {mass:f} #{specie:s}"
            ]

    def setup(self, max_time = 86400, gpu = False, atom_style = 'atomic', units = 'metal', boundary = 'p p p', logfile = 'log.lammps', append_log = True, seed = 1, newton = None):
        
        self.input.append('# == SETUP ==')
        if append_log:
            self.input.append(f'log\t\t{logfile} append')
            
        if max_time is not None:
            self.input.append(f'timer\t\ttimeout {max_time}')
        if gpu:
            self.input.append(f'gpu\t\t1 neigh no split -1')

        self.input.append(f'atom_style\t{atom_style}')
        self.input.append(f'units\t\t{units}')
        self.input.append(f'boundary\t{boundary}')
        if newton is not None:
            self.input.append(f'newton\t\t{newton}')

        self.seed = seed
        self.atom_style = atom_style
        self.units = units

    def data(self, read_data = 'data.lammps', read_restart = None, specorder = None):

        self.input.append('\n# == DATA ==')
        self.restart = True if read_restart is not None else False

        if self.restart: 
            self.input.append(f'read_restart\t{read_restart}')
        else:
            self.input.append(f'read_data\t{read_data}')

        # masses
        self.get_masses(specorder)
        for mass in self.fmasses:
            self.input.append(f'mass\t\t{mass}')
        
        # check constraints
        constraints = self.atoms.constraints
        if len(constraints) > 0:
            constr_idxs = []
            for constr in constraints:
                if type(constr) == FixAtoms:
                    constr_idxs.extend( constr.get_indices() )

            all_idxs = [i for i in range(len(self.atoms))]
            # lammps: serial start from 1 not 0
            free_atoms = [str(x+1) for x in all_idxs if x not in constr_idxs]
            sum = np.sum([int(x) for x in free_atoms])
            if sum == len(free_atoms)*(int(free_atoms[0])+int(free_atoms[-1]))/2 :
                free_atoms = free_atoms[0]+":"+free_atoms[-1]
            else:
                free_atoms = " ".join(free_atoms)

            self.atom_group = 'free_atoms'
            self.atom_group_idxs = free_atoms 

            self.input.append(f'group {self.atom_group} id {self.atom_group_idxs}')
        else:
            self.atom_group = 'all'

    def interactions(self, pair_style='lj/cut 2.5',pair_coeff='* * 1 1', timestep=0.001, neigh_modify = None):
        self.input.append('\n# == INTERACTIONS ==')
        self.input.append(f'pair_style\t{pair_style}')
        self.input.append(f'pair_coeff\t{pair_coeff}')
        self.input.append(f'timestep\t{timestep}')
        if neigh_modify is not None:
            self.input.append(f'neigh_modify\t{neigh_modify}') #neigh_modify    delay 1 every 1

    def output(self, out_freq = 500, restart_freq = None, dump_style = 'atom', file_dump = 'traj.lammps'):
        self.input.append('\n# == OUTPUT ==')
        # thermo
        self.input.append(f'thermo\t\t{out_freq}')
        self.input.append('thermo_style\tcustom step temp press ke pe etotal vol pxx pyy pzz pxy pxz pyz')
        self.input.append('thermo_modify\tflush yes format float %23.16g')
        
        # dump
        self.dump(out_freq,dump_style=dump_style,file_dump=file_dump)
        # restart
        if restart_freq == None:
            restart_freq = out_freq
        self.input.append(f'restart\t\t{restart_freq} lmp.restart lmp2.restart')

        ##################### self.input.append(f'\t')
    def dump(self, out_freq = 500, dump_name='dump_all', dump_style = 'atom', file_dump = 'traj.tmp'):
        # dump
        if dump_style == 'atom':
            self.input.append(f'dump\t\t{dump_name} all custom {out_freq} {file_dump} id type element x y z')
            self.input.append(f'dump_modify\t{dump_name} format float %12.8f append yes element {" ".join(self.specorder)}')
        elif dump_style == 'full':
            self.input.append(f'dump\t\t{dump_name} all custom {out_freq} {file_dump} id type element x y z vx vy vz fx fy fz')
            self.input.append(f'dump_modify\t{dump_name} format float %12.8f append yes element {" ".join(self.specorder)}')
        elif dump_style == 'xyz':
            self.input.append(f'dump\t\t{dump_name} all xyz {out_freq} {file_dump}')
            self.input.append(f'dump_modify\t{dump_name} append yes element {" ".join(self.specorder)}')

    def minimize(self, etol=1e-4, ftol=1e-6, maxiter=100, maxeval=1000):
        self.input.append('\n# == MINIMIZATION ==')
        self.input.append(f'minimize\t{etol} {ftol} {maxiter} {maxeval}')

    def nvt(self, temperature = 300, temp_damp = 0.1, steps = 1000000, steps_equil_nvt = 10000, steps_equil_npt = 0, pressure = 1, pressure_damp = 1):
        if not self.restart:
            if steps_equil_nvt > 0:
                self._equil_nvt(temperature,temp_damp, steps_equil_nvt)
            if steps_equil_npt > 0:
                self._equil_npt(temperature,temp_damp,pressure,pressure_damp,steps_equil_npt)        
        self._prod_nvt(temperature,temp_damp,steps)

    def equil_nvt(self, temperature = 300, temp_damp = 0.1, steps = 10000):
        self.input.append('\n# == NVT EQUILIBRATION ==')
        self.input.append(f'fix\t\tfix_nvt {self.atom_group} nvt temp {temperature} {temperature} {temp_damp}')
        #self.input.append(f'fix\t\tfix_nve {self.atom_group} nve')
        #self.input.append(f'fix\t\tfix_temp {self.atom_group} temp/csvr {temperature} {temperature} {temp_damp} {self.seed}')
        self.input.append(f'velocity\tall create {temperature} {self.seed} dist gaussian')
        self.input.append(f'run\t\t{steps}')
        self.input.append(f'unfix\t\tfix_nvt')
        #self.input.append(f'unfix\t\tfix_nve')
        #self.input.append(f'unfix\t\tfix_temp')

    def _equil_npt(self, temperature = 300, temp_damp = 0.1, pressure = 1, press_damp = 1, steps = 10000):
        self.input.append('\n# == NPT EQUILIBRATION ==')
        self.input.append(f'fix\t\tfix_nph {self.atom_group} nph iso ${pressure} ${pressure} ${press_damp}')
        self.input.append(f'fix\t\tfix_temp {self.atom_group} temp/csvr {temperature} {temperature} {temp_damp} {self.seed}')
        #self.input.append(f'velocity\tall create {temperature} {self.seed} dist gaussian')
        self.input.append(f'run\t\t{steps}')
        self.input.append(f'unfix\t\tfix_nph')
        self.input.append(f'unfix\t\tfix_temp')
        
    def prod_nvt(self, thermostat='bussi', temperature = 300, temp_damp = 0.1, steps = 1000000, create_velocity = 'false', plumed = None, wall_zlo=None,wall_zhi=None,
                multiple_dump= False, dump_interval = 10000, dump_tmp = 'traj.tmp', dump_stored = 'traj_hf.lammps'):
        self.input.append('\n# == NVT PRODUCTION ==')
        if not self.restart:
            self.input.append('reset_timestep\t0')

        if create_velocity:
            self.input.append(f'velocity\tall create {temperature} {self.seed} dist gaussian')

        if plumed is not None:
            self.input.append(f'fix\t\tfix_plumed all plumed plumedfile {plumed} outfile log.plumed')

        if thermostat == 'bussi':
            self.input.append(f'fix\t\tfix_nve {self.atom_group} nve')
            self.input.append(f'fix\t\tfix_temp {self.atom_group} temp/csvr {temperature} {temperature} {temp_damp} {self.seed}')
        else:
            self.input.append(f'fix\t\tfix_nvt {self.atom_group} nvt temp {temperature} {temperature} {temp_damp}')

        if (wall_zlo is not None) or (wall_zhi is not None):
            self.set_wall(wall_zlo,wall_zhi)

        if multiple_dump:
            self.input.append(f'run\t\t{steps}\tevery\t{dump_interval} &')
            self.input.append(f'\t\t"shell cp -b {dump_tmp} {dump_stored}" &')
            self.input.append(f'\t\t"shell cat /dev/null > {dump_tmp}"')
            self.input.append(f'shell rm {dump_tmp}')
        else:
            self.input.append(f'run\t\t{steps}')

        if plumed is not None:
            self.input.append(f'unfix\t\tfix_plumed')

        if thermostat == 'bussi':
            self.input.append(f'unfix\t\tfix_nve')
            self.input.append(f'unfix\t\tfix_temp')
        else:
            self.input.append(f'unfix\t\tfix_nvt')
        
        if (wall_zlo is not None) or (wall_zhi is not None):
            self.input.append(f'unfix\t\tfix_wall')

    def set_wall(self, zlo=None,zhi=None):
        boundary = [i for i in self.input if 'boundary' in i][0].split('\t')[-1].split(' ')[-1]
        assert boundary == 'f', f'you need to disable pbc along z in order to use walls (z boundary = {boundary} ==> f)'

        wall = 'fix\t\tfix_wall all wall/reflect '
        if zlo is not None:
            wall += f'zlo {zlo} '
        if zhi is not None:
            wall += f'zhi {zhi} '
        self.input.append(wall)

    def finalize(self):
        self.input.append(f'write_data\tfinal.lammpsdata')
        self.input.append(f'write_restart\tfinal.restart')

    def write_input(self, folder = './', fname = 'in.lammps'):
        f = open(folder+fname, "w")
        for inp in self.input:
            f.write(inp+'\n')
        f.close()

    def write_data(self, folder = './', fname = 'data.lammps'):
        write_lammps_data(folder+fname, self.atoms, specorder=self.specorder, units = self.units, atom_style=self.atom_style)

    def print(self):
        for inp in self.input:
            print(inp)