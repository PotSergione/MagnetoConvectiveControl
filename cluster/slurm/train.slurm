#!/usr/bin/env bash

#SBATCH --job-name=mcc-train
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch train.slurm EXPERIMENT_YAML [MACHINE_YAML]" >&2
    exit 2
fi

experiment="$1"
machine="${2:-configs/machines/slurm_a100.example.yaml}"

: "${NEKRS_HOME:?Set NEKRS_HOME in the site submission environment}"
: "${MCC_ARTIFACT_ROOT:?Set MCC_ARTIFACT_ROOT to durable run storage}"
: "${RL_SNAPSHOTS_DIR:?Set RL_SNAPSHOTS_DIR to the restart dataset}"

export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "Experiment: ${experiment}"
echo "Machine:    ${machine}"
echo "Artifacts:  ${MCC_ARTIFACT_ROOT}"
echo "Snapshots:  ${RL_SNAPSHOTS_DIR}"
echo "Commit:     $(git rev-parse HEAD)"

srun --ntasks=1 mcc-train --experiment "${experiment}" --machine "${machine}"
