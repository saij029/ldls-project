#!/bin/bash
#SBATCH --job-name=tinyllama-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=training_%j.log

# Load modules
module load cuda/12.1
source activate megatron_env

# Go to working directory
cd ~/tinyllama-pretraining/Megatron-LM

# Run training
bash train_tinyllama.sh

