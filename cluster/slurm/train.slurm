#!/usr/bin/env bash

#SBATCH --job-name=mcc-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch train.slurm --experiment FILE [--machine FILE] [--scale FILE]" >&2
    exit 2
fi

: "${NEKRS_HOME:?Set NEKRS_HOME in the site submission environment}"

export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "Training arguments: $*"
echo "Commit:     $(git rev-parse HEAD)"

# The controller owns the allocation. SmartSim creates the database and model
# srun steps, so wrapping the controller in another srun would create nested
# and potentially resource-blocked Slurm steps.
mcc-train "$@"
