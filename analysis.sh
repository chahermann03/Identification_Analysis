#!/bin/bash -l

#SBATCH -o ./comp_big.out
#SBATCH -D ./
#SBATCH -J comp_big

# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
# Wall clock limit (max. is 24 hours):
#SBATCH --time=03:00:00


module load texlive
conda activate raven_voids

python3 comparison_nbody+2lpt.py
