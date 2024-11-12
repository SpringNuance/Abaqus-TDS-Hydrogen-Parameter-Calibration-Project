#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus_OneNodeSmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=small
#SBATCH --account=project_2008630
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

module purge
module load abaqus

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abaqus job=geometry input=geometry.inp cpus=$CPUS_TOTAL -verbose 2 interactive

# run postprocess.py after the simulation completes
abaqus cae noGUI=postprocess.py