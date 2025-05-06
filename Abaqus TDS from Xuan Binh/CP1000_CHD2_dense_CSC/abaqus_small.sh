#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --partition=small
#SBATCH --account=project_2008630
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

unset SLURM_GTIDS

module load abaqus/2023
module load python-data

# Old Intel compilers (ifort)
# module load intel-oneapi-compilers-classic
# New Intel compilers (ifx)
module load intel-oneapi-compilers
# module load gcc

cd $PWD

# Find the .inp file in the current directory (there should be only one)
INP_FILE=$(ls *.inp)

# Remove the extension for the job name
JOB_NAME=${INP_FILE%.inp}

CPUS_TOTAL=$(( $SLURM_NTASKS * $SLURM_CPUS_PER_TASK ))

abaqus job=$JOB_NAME input=$INP_FILE user=deformation_diffusion cpus=$CPUS_TOTAL -verbose 2 interactive

# Postprocess results
# abaqus cae noGUI=postprocess.py
