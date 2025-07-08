#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Convenience launcher to evaluate all rule-based controller variants.
# -----------------------------------------------------------------------------
sbatch slurm_scripts/run_rbc_evaluation.sh baseline household-economic
sbatch slurm_scripts/run_rbc_evaluation.sh baseline_price_window household-economic
sbatch slurm_scripts/run_rbc_evaluation.sh baseline feeder-economic
sbatch slurm_scripts/run_rbc_evaluation.sh baseline_price_window feeder-economic
sbatch slurm_scripts/run_rbc_evaluation.sh baseline_voltage feeder-voltage
