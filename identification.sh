#!/bin/bash -l

#SBATCH -o ./spark_emu.%A_%a
#SBATCH -D ./
#SBATCH -J spark_emu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=05:00:00
#SBATCH --array=0-24

#-- Load the modules

module purge
module load gcc openmpi gsl
LD_LIBRARY_PATH=${GSL_HOME}/lib:${FFTW_HOME}/lib:${HDF5_HOME}/lib
export LD_LIBRARY_PATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#-- Build the parameter file

padded_id=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

cp sparkling_box.param /u/chahermann/Identification_Analysis/paramfiles/sparkling_box{$padded_id}.param
param_file=/u/chahermann/Identification_Analysis/paramfiles/sparkling_box{$padded_id}.param
sed -i "s/ID/$padded_id/g" $param_file

#-- Execute the code
exepath=/u/chahermann/Sparkling/build
srun ${exepath}/sparkling_box $param_file
