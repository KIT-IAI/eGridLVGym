#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train single-objective RL algorithms with GPU resources.
# Usage: sbatch slurm_scripts/run_training.sh <ALGORITHM> <EPISODES> <NUM_ENVS> <USE_CASE> <SEED>
#        chmod +x slurm_scripts/run_training.sh 
#        slurm_scripts/run_training.sh ppo 8000 4 household-economic 42
# logs:  slurm_logs_train
# -----------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_train/slurm-train-%j.out
#SBATCH --error=slurm_logs_train/slurm-train-%j.err
#SBATCH --job-name=rl-train-%j

set -euo pipefail

PYTHON_ENV="../SORL_ENV"
source "$PYTHON_ENV/bin/activate"

ALGORITHM=${1:?Missing algorithm}
EPISODES=${2:?Missing episodes}
NUM_ENVS=${3:?Missing number of environments}
USE_CASE=${4:?Missing use case}
SEED=${5:?Missing seed}

echo "Training $ALGORITHM for $EPISODES episodes on $USE_CASE (envs=$NUM_ENVS)."

python3.9 train.py --algorithm "$ALGORITHM" --episodes "$EPISODES" --num-envs "$NUM_ENVS" --usecase "$USE_CASE" --seed "$SEED"

echo "Training finished successfully."
