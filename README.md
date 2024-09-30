# Data-efficient modeling of catalytic reactions via enhanced sampling and on-the-fly learning of machine learning potentials

This repository contains the code needed to reproduce the workflow presented in:

S. Perego and L. Bonati, _Data-efficient modeling of catalytic reactions via enhanced sampling and on-the-fly learning of machine learning potentials_, ChemRxiv, doi:[10.26434/chemrxiv-2024-nsp7n](https://doi.org/10.26434/chemrxiv-2024-nsp7n)

## Contents:
```
├── notebooks           # notebooks describing the different stages of the MLP construction 
│   ├── 0_preliminary
│   ├── 1_exploration
│   └── 2_convergence
├── configs             # configuration files for the different tasks
│   ├── deal
│   ├── flare
│   ├── lammps
│   ├── mace
│   ├── plumed
│   └── qe
└── mlputils            # python module containing the necessary functions
```

## Requirements

The following software and versions have been used:

* FLARE (v1.3.3-fix from [luigibonati/flare](https://github.com/luigibonati/flare/tree/1.3.3-fix)) $\rightarrow$ modified FLARE version with small fixes to ensure it work correctly with LAMMPS & PLUMED simulations as well as for the DEAL active learning selection scheme
* MACE (v0.35 from [ACEsuit/mace](https://github.com/ACEsuit/mace))
* LAMMPS with support of:
    * MACE (patched version from [ACEsuit/lammps](https://github.com/ACEsuit/lammps))
    * FLARE (pair style from [mir-group](https://github.com/mir-group/flare))
    * PLUMED (v2.9 from [plumed/plumed2](https://github.com/plumed/plumed2))
