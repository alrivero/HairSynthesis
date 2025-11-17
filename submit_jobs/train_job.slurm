#!/bin/bash

#SBATCH --job-name=smirkhair
#SBATCH --output=logs/results_%j.out
#SBATCH --error=logs/results_%j.err

# Resource
#SBATCH --partition a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=00:20:00

# Load required modules
module load cuda120/toolkit/12.0

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Display job info
echo "========================================="
echo "Job ID: $SLURM_JOBID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "========================================="

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run the application
echo "Starting computation..."
srun python train.py configs/config_train_hs.yaml

# Post-processing (if needed)
echo "Job completed at: $(date)"
