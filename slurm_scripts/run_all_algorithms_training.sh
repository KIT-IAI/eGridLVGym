#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launch training jobs for a suite of RL algorithms on the selected use case.
# Usage:    sbatch slurm_scripts/run_all_algorithms_training.sh <USE_CASE>
#           chmod +x slurm_scripts/run_all_algorithms_training.sh 
# Example:  slurm_scripts/run_all_algorithms_training.sh household-economic
# logs:  slurm_logs_train
# Supported use cases: household-economic | feeder-economic | feeder-voltage | feeder-combined
# -----------------------------------------------------------------------------
USE_CASE=${1:?Missing use case argument}
EPISODES=8000
NUM_ENVS=4
# SEED=42
# SEED=43
# SEED=44
# SEED=45
# SEED=46
SEED=42

echo "Starting training jobs for use case: $USE_CASE"
echo "SEED: $SEED"

# Dynamic mapping
ALG1=ppo
ALG2=sac
ALG3=td3
ALG4=ddpg
ALG5=a2c
ALG6=tqc
ALG7=trpo
ALG8=recurrent_ppo

# Checking output
echo "Algorithms to be launched via sbatch:"
for i in {1..8}; do
    eval "alg=\$ALG$i"
    echo "  - ALG$i = $alg"
done

# Fix block with variables
sbatch ./slurm_scripts/run_training.sh "$ALG1" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #1
sbatch ./slurm_scripts/run_training.sh "$ALG2" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #2
sbatch ./slurm_scripts/run_training.sh "$ALG3" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #3
sbatch ./slurm_scripts/run_training.sh "$ALG4" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #4
sbatch ./slurm_scripts/run_training.sh "$ALG5" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #5
sbatch ./slurm_scripts/run_training.sh "$ALG6" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #6
sbatch ./slurm_scripts/run_training.sh "$ALG7" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #7
sbatch ./slurm_scripts/run_training.sh "$ALG8" "$EPISODES" "$NUM_ENVS" "$USE_CASE" "$SEED" #8
