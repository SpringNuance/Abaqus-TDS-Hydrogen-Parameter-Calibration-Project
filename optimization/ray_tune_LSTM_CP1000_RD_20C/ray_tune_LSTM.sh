#!/bin/bash
#SBATCH --job-name=ray-tune-LSTM
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1

# Loading the modules
module load intel-oneapi-compilers
module load gcc/11.3.0 cuda/11.7.0
module load openmpi
module load python-data

# Start the Ray head node
ray start --head --port=6379

# Get the IP address of the head node
HEAD_IP=$(hostname -I | awk '{print $1}')

# Environment variables for Ray workers to join the cluster
export RAY_HEAD_IP=$HEAD_IP
export RAY_HEAD_PORT=6379

# Start Ray worker nodes from other allocated nodes
srun --exclusive --nodes=${SLURM_NNODES-1} --ntasks=${SLURM_NNODES-1} \
     ray start --address=$RAY_HEAD_IP:$RAY_HEAD_PORT

# Run your Python script using Ray Tune
python your_tuning_script.py

# Stop the Ray cluster after the job is done
ray stop
