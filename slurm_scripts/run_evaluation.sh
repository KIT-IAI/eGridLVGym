#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Evaluate single-objective RL algorithms.
# Usage: sbatch run_evaluation.sh <ALGORITHM> <NUM_ENVS> <USE_CASE>
# -----------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=cpuonly
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_eval/slurm-eval-%j.out
#SBATCH --error=slurm_logs_eval/slurm-eval-%j.err
#SBATCH --job-name=rl-eval-%j

set -euo pipefail

PYTHON_ENV="../SORL_ENV"
source "$PYTHON_ENV/bin/activate"

ALGORITHM=${1:?Missing algorithm}
NUM_ENVS=${2:?Missing num_envs}
USE_CASE=${3:?Missing use_case}

echo "Using Python environment: $PYTHON_ENV"
echo "Algorithm: $ALGORITHM, Envs: $NUM_ENVS, Usecase: $USE_CASE"
echo "****************************************"
python3.9 evaluation.py --algorithm "$ALGORITHM" --envs "$NUM_ENVS" --usecase "$USE_CASE"
echo "****************************************"
