#!/bin/bash
#SBATCH --account=IscrB_AmmoFeCo
#SBATCH --partition=boost_usr_prod  # partition to be used
##SBATCH --time 24:00:00             # format: HH:MM:SS
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=1         # tasks out of 32
#SBATCH --cpus-per-task=32
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00             # format: HH:MM:SS
############################

#source env
module load intel-oneapi-tbb
module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load openblas/0.3.21--gcc--11.3.0
module load intel-oneapi-mkl

source activate /leonardo/pub/userexternal/sperego0/envs/flare_v1.3.3-mkl
conda activate /leonardo/pub/userexternal/sperego0/envs/flare_v1.3.3-mkl

# Program folder
DEAL=/leonardo_work/IscrB_AmmoFeCo/FeCo/DEAL/deal/deal.py

#running
python $DEAL input.yaml    
