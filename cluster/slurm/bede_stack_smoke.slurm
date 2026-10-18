#!/usr/bin/env bash

#SBATCH --job-name=mcc-stack-smoke
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=mcc-stack-smoke-%j.out
#SBATCH --error=mcc-stack-smoke-%j.err

set -euo pipefail

: "${MCC_STORAGE_ROOT:?Set MCC_STORAGE_ROOT to your user directory under /nobackup}"

require_directory() {
    [[ -d "$1" ]] || {
        echo "Required directory not found: $1" >&2
        exit 1
    }
}

require_file() {
    [[ -f "$1" ]] || {
        echo "Required file not found: $1" >&2
        exit 1
    }
}

require_executable() {
    [[ -x "$1" ]] || {
        echo "Required executable not found: $1" >&2
        exit 1
    }
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required command not found: $1" >&2
        exit 1
    }
}

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
conda_base="${MCC_CONDA_BASE:-${MCC_STORAGE_ROOT}/aarch64/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/magnetoconvective-control}"
export NEKRS_HOME="${NEKRS_HOME:-${MCC_STORAGE_ROOT}/aarch64/nekrs-mhd}"

require_directory "${project_root}/.git"
require_file "${project_root}/data/snapshots/blanket0.f00260"
require_file "${conda_base}/etc/profile.d/conda.sh"
require_executable "${conda_env}/bin/python"
require_executable "${NEKRS_HOME}/bin/nrsmpi"

module purge
module load gcc/12.2
module load cuda/12.4.1
module load openmpi/4.1.6

source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

export PATH="${NEKRS_HOME}/bin:${PATH}"
require_command mpicc
require_command mpic++
require_command mpifort
require_command cmake
export CC="$(command -v mpicc)"
export CXX="$(command -v mpic++)"
export FC="$(command -v mpifort)"
export BLANKET_CMAKE="$(command -v cmake)"
export NEKRS_UDF_INCLUDES="${conda_env}/include;${NEKRS_HOME}/udf"
export NEKRS_UDF_LIBS="${conda_env}/lib/libsmartredis.so"

cache_root="${MCC_CACHE_ROOT:-${MCC_STORAGE_ROOT}/smoke/cache}"
runtime_lib="${MCC_STORAGE_ROOT}/smoke/smartredis-lib"
work_dir="${MCC_STORAGE_ROOT}/smoke/mcc-stack-${SLURM_JOB_ID}"
mkdir -p \
    "${cache_root}/occa" \
    "${cache_root}/nekrs" \
    "${runtime_lib}" \
    "${work_dir}"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"

export LD_LIBRARY_PATH="${runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export MCC_ARTIFACT_ROOT="${work_dir}/artifacts"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))

echo "Host:       $(hostname)"
echo "Arch:       $(uname -m)"
echo "Commit:     $(git -C "${project_root}" rev-parse HEAD)"
echo "Work:       ${work_dir}"
echo "Artifacts:  ${MCC_ARTIFACT_ROOT}"
nvidia-smi -L

python - <<'PY'
import torch
from importlib.metadata import version

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GH200")
print("GPU:", torch.cuda.get_device_name())
print("SmartSim:", version("smartsim"))
print("SmartRedis:", version("smartredis"))
PY

cd "${work_dir}"

mcc-train \
    --experiment "${project_root}/configs/experiments/stack_smoke.yaml" \
    --machine "${project_root}/configs/machines/bede_stack_smoke.yaml" \
    --scale "${project_root}/configs/scales/one_gpu.yaml"

python - <<'PY'
import os
from pathlib import Path

from sb3_contrib import RecurrentPPO

run = Path(os.environ["MCC_ARTIFACT_ROOT"]) / "bede_stack_smoke"
required = (
    run / "metadata.json",
    run / "config.resolved.yaml",
    run / "run_contract.json",
    run / "action_mapping.json",
    run / "reward_metrics.csv",
    run / "latest.zip",
    run / "checkpoints" / "step_000000000020.zip",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError("Missing smoke-test artifacts: " + ", ".join(missing))

model = RecurrentPPO.load(run / "latest.zip", device="cpu")
if model.num_timesteps != 20:
    raise RuntimeError(
        f"Expected a 20-sample PPO update, found {model.num_timesteps} timesteps"
    )

print("Reloaded PPO checkpoint at", model.num_timesteps, "timesteps")
print("Persistent artifacts:", run)
PY

# These directories contain generated case copies and TensorBoard event data,
# not scientific results. Preserve them automatically when any earlier command
# fails, but remove them after a fully validated smoke test.
rm -rf -- "${work_dir}/rl_training" "${work_dir}/tensorboard"

echo "MCC BEDE ONE-GPU STACK SMOKE: SUCCESS"
