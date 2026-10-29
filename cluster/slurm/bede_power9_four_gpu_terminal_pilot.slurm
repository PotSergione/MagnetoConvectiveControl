#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-term
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=mcc-power9-v100-terminal-%j.out
#SBATCH --error=mcc-power9-v100-terminal-%j.err

set -euo pipefail

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
export MCC_EXPERIMENT="${project_root}/configs/experiments/bede_v100_ppo_200_terminal_pilot.yaml"
exec bash "${project_root}/cluster/slurm/bede_power9_four_gpu_train.slurm"
