#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Evaluate rule-based controllers.
# Usage: sbatch run_rbc_evaluation.sh <RBC_VARIANT> <USE_CASE>
# -----------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_eval/slurm-rbc-eval-%j.out
#SBATCH --error=slurm_logs_eval/slurm-rbc-eval-%j.err
#SBATCH --job-name=rbc-eval-%j

set -euo pipefail

PYTHON_ENV="../SORL_ENV"
source "$PYTHON_ENV/bin/activate"

RBC_VARIANT=${1:?Missing RBC variant}
USE_CASE=${2:?Missing use case}

python3.9 evaluation.py --algorithm "$RBC_VARIANT" --envs 1 --no-normalization --usecase "$USE_CASE" --obs-space dict
