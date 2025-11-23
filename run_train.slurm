#!/bin/bash
#SBATCH -J nips_9d_train           # Job name
#SBATCH -o nips_9d_train.o.%j    # Output file name (%j expands to jobID)
#SBATCH -e nips_9d_train.e.%j    # Error file name (%j expands to jobID)
#SBATCH -p gpu-a100-small          # Queue name
#SBATCH -N 1                       # Total number of nodes
#SBATCH -n 1                       # Total number of tasks (1 GPU)
#SBATCH -t 48:00:00                # Run time (hh:mm:ss) - Max 48h usually
#SBATCH --mail-type=begin,end,fail # Email events
#SBATCH --mail-user=rrl897@my.utexas.edu # <--- CHANGE THIS

# ---------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------
echo "Job started at: $(date)"
echo "Node: $(hostname)"

# Load necessary TACC modules
module load python3

# Activate Virtual Environment
# Adjust path if your venv is named differently or located elsewhere
source $WORK/venv_siren/bin/activate

# ---------------------------------------------------
# 2. WandB Setup (Offline Mode)
# ---------------------------------------------------
# Compute nodes usually don't have internet, so we log to disk.
# You must run 'wandb sync wandb/offline-run...' from the login node later.
export WANDB_MODE=offline

# ---------------------------------------------------
# 3. Execution
# ---------------------------------------------------
# Ensure we are in the directory containing the scripts
cd $WORK/irn/  # <--- CHANGE THIS to your actual script folder path

echo "Starting Training..."
python3 train_9d.py

echo "Job finished at: $(date)"
