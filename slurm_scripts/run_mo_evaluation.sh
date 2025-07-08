#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Evaluate CAPQL models in various multi-objective modes.
# -----------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_eval/slurm-mo-eval-%j.out
#SBATCH --error=slurm_logs_eval/slurm-mo-eval-%j.err
#SBATCH --job-name=rl-mo-eval-%j

set -euo pipefail
#module load devel/cuda/12.2

PYTHON_ENV="../MORL_ENV"
source "$PYTHON_ENV/bin/activate"
# Replace with your wandb API key
export WANDB_API_KEY="xxx"

echo "Multi-objective evaluation started."

python3.9 evaluation_mo.py --mode weight_single --model_folder best_capql --weights "4,-2"
python3.9 evaluation_mo.py --mode weight_multi  --model_folder best_capql --grid_type linear
python3.9 evaluation_mo.py --mode weight_multi  --model_folder best_capql --grid_type combined
python3.9 evaluation_mo.py --mode weight_single --model_folder capql --usecase feeder-voltage --weights "1,1,1"

echo "Multi-objective evaluation completed."
