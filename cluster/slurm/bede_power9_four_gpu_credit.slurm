#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-credit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-19:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

variant="${1:-reference}"
case "${variant}" in
    reference|replica|one_epoch|low_noise) ;;
    *) echo "Usage: sbatch $0 [reference|replica|one_epoch|low_noise]" >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { echo "Expected at most one variant argument" >&2; exit 2; }

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
storage_root="${MCC_STORAGE_ROOT:-/nobackup/projects/bdman39/sergione}"
export MCC_EXPERIMENT="${project_root}/configs/experiments/bede_v100_ppo_200_credit_${variant}.yaml"
# SmartSim and TensorBoard otherwise use shared paths in the checkout.
export MCC_WORK_DIR="${storage_root}/training/work/credit-${variant}-${SLURM_JOB_ID:?Submit with sbatch}"

# Distinct variants have distinct artifacts; prevent simultaneous writers for one variant.
mkdir -p "${storage_root}/training"
exec 9>"${storage_root}/training/.credit-${variant}.lock"
flock -n 9 || { echo "Another ${variant} job is already running" >&2; exit 1; }
exec bash "${project_root}/cluster/slurm/bede_power9_four_gpu_train.slurm"
