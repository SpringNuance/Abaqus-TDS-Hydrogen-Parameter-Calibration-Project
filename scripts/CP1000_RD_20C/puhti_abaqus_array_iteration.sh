#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus_array_small_iteration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --account=project_2008630
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset SLURM_GTIDS
module purge
module load abaqus

### Change to the work directory
fullpath=$(sed -n ${SLURM_ARRAY_TASK_ID}p scripts/CP1000_RD_20C/iteration_simulation_array_paths.txt) 
cd ${fullpath}

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abaqus job=geometry input=geometry.inp cpus=$CPUS_TOTAL -verbose 2 interactive

# run postprocess.py after the simulation completes
abaqus cae noGUI=postprocess.py
