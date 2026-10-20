#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-4time
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=mcc-power9-four-v100-timing-%j.out
#SBATCH --error=mcc-power9-four-v100-timing-%j.err

set -euo pipefail

export MCC_TIMING_ENVS=4
project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
exec "${project_root}/cluster/slurm/bede_power9_v100_timing.slurm"
