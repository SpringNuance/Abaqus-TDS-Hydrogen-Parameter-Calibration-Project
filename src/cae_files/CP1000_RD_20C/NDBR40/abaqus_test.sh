#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:15:00
#SBATCH --partition=test
#SBATCH --account=project_2008630
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

# This script runs in parallel Abaqus example e1 on Puhti server using 10 cores.

unset SLURM_GTIDS

module load abaqus/2024

# Old Intel compilers
module load intel-oneapi-compilers-classic
# module load gcc

cd $PWD

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abaqus job=Job-1 input=Job-1.inp cpus=$CPUS_TOTAL -verbose 2 interactive