#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launch evaluation jobs for a suite of RL algorithms on the selected use case.
# Usage:    sbatch run_all_algorithms_evaluation.sh <USE_CASE>
# Example:  sbatch run_all_algorithms_evaluation.sh feeder-economic
# Supported use cases: household-economic | feeder-economic | feeder-voltage | feeder-combined
# -----------------------------------------------------------------------------
USE_CASE=${1:?Missing use case argument}

echo "Starting evaluation jobs for use case: $USE_CASE"

for ALG in ppo sac td3 ddpg a2c tqc trpo recurrent_ppo; do
  sbatch slurm_scripts/run_evaluation.sh "$ALG" 4 "$USE_CASE"
done
