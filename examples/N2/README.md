## $N_2$ decomposition on FeCo

- Ref.: Perego and Bonati, *npj Computational Materials* 10, 291 (2024), doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)
- Data: Nitrogen dissociation on FeCo(110) surface, collected via OPES enhanced sampling simulations.
- Summary: 
    1. pre-select high uncertainty configurations based on ensemble-based standard deviation
    2. filter them using the GP predictive variance to identify structurally different configurations
    3. analyze distribution of selected CVs
- See also the notebooks in: `DEAL/npj_supporting_data/notebooks/2_convergence/`

___

### Instructions

#### Input
Copy original trajectory (xyz) from OPES simulation and corresponding COLVAR file (used for monitoring the distribution of CVs)
```bash
cd input 
bash get_data.sh 
```
#### Selection
Pre-process data and run DEAL: [example-N2.ipynb](example-N2.ipynb)

#### Run DEAL
Enable the relevant environment and....
```bash
deal -c input.yaml
``` 
Consider running it on a HPC cluster or on a high-memory machine: below an example SLURM script to run DEAL on CSCS.
Make sure to adjust the number of nodes, tasks and cpus per task according to your system and the size of your dataset.
```bash
#!/bin/bash -l
#========================================
#SBATCH --account=xxxxxx          # account to be charged
#========================================
# #SBATCH --partition=debug         # partition to be used
#SBATCH --time 24:00:00             # format: HH:MM:SS
#========================================
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=16        # MPI
#SBATCH --cpus-per-task=16          # openMP
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#========================================
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules
#========================================
## to submit the job, use:
# sbatch this_script.sh

#========================================
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
ulimit -s unlimited
#========================================
# load modules (load the module used to install DEAL and its dependencies)
module load gcc/14.2.0 openblas/0.3.29
conda activate deal
#========================================
# Run DEAL 
deal -c input.yaml > deal.log 2>&1         # run deal

# END OF SCRIPT
```
