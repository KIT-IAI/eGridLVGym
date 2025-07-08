#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train CAPQL in a chosen environment for a given number of episodes.
# Usage: sbatch run_mo_training.sh <ENV_NAME> <EPISODES> [USE_WANDB=true|false]
# -----------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_train/slurm-mo-train-%j.out
#SBATCH --error=slurm_logs_train/slurm-mo-train-%j.err
#SBATCH --job-name=rl-mo-train-%j

set -euo pipefail

# Read arguments passed to the script
ENV_NAME=${1:?Missing environment name}
EPISODES=${2:?Missing number of episodes}
# Optional third argument
USE_WANDB=${3:-false}

PYTHON_ENV="../MORL_ENV"
source "$PYTHON_ENV/bin/activate"

CMD="python3.9 train_mo.py --env_name \"$ENV_NAME\" --episodes \"$EPISODES\""

if [ "$USE_WANDB" = true ]; then
 # Replace with your wandb API key
  export WANDB_API_KEY="xxx"
  CMD+=" --wandb_log"
fi

echo "Running CAPQL training: $CMD"
eval $CMD

echo "All trainings finished."
