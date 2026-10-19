#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-smoke
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=mcc-power9-stack-smoke-%j.out
#SBATCH --error=mcc-power9-stack-smoke-%j.err

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
platform_root="${MCC_STORAGE_ROOT}/ppc64le"
conda_base="${MCC_CONDA_BASE:-${platform_root}/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/mcc-stack-probe}"
export NEKRS_HOME="${NEKRS_HOME:-${platform_root}/nekrs-mhd}"

require_directory "${project_root}/.git"
require_file "${project_root}/data/snapshots/blanket0.f00260"
require_file "${conda_base}/etc/profile.d/conda.sh"
require_executable "${conda_env}/bin/python"
require_executable "${NEKRS_HOME}/bin/nrsmpi"

module purge
module load gcc/10.2.0
module load cuda/12.0.1
module load openmpi/4.0.5

source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

if [[ "$(uname -m)" != "ppc64le" ]]; then
    echo "Expected a POWER9 ppc64le node, found $(uname -m)" >&2
    exit 1
fi

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
require_file "${conda_env}/include/client.h"
require_file "${conda_env}/lib/libsmartredis.so"

# OCCA and NekRS caches contain architecture-specific generated code. Never
# share them with the aarch64/GH installation under MCC_STORAGE_ROOT.
cache_root="${MCC_CACHE_ROOT:-${platform_root}/cache}"
runtime_lib="${platform_root}/runtime-lib"
work_dir="${MCC_STORAGE_ROOT}/smoke/mcc-power9-stack-${SLURM_JOB_ID}"
mkdir -p \
    "${cache_root}/occa" \
    "${cache_root}/nekrs" \
    "${runtime_lib}" \
    "${work_dir}"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"

# The native SmartRedis library and OpenCE PyTorch need the ppc64le Conda
# runtime. Module-provided MPI/CUDA paths remain available later in the list.
export LD_LIBRARY_PATH="${runtime_lib}:${conda_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export MCC_ARTIFACT_ROOT="${work_dir}/artifacts"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))

echo "Host:       $(hostname)"
echo "Arch:       $(uname -m)"
echo "Commit:     $(git -C "${project_root}" rev-parse HEAD)"
echo "Work:       ${work_dir}"
echo "Artifacts:  ${MCC_ARTIFACT_ROOT}"
echo "NekMHD:     ${NEKRS_HOME}"
echo "Conda env:  ${conda_env}"
module list
nvidia-smi -L

python - <<'PY'
import platform
from importlib.metadata import version

import torch

print("Python architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if platform.machine() != "ppc64le":
    raise RuntimeError("Python is not running on ppc64le")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated V100")
gpu = torch.cuda.get_device_name()
print("GPU:", gpu)
if "V100" not in gpu.upper():
    raise RuntimeError(f"Expected a V100, found {gpu}")
print("SmartSim:", version("smartsim"))
print("SmartRedis:", version("smartredis"))
PY

cd "${work_dir}"

mcc-train \
    --experiment "${project_root}/configs/experiments/stack_smoke.yaml" \
    --machine "${project_root}/configs/machines/bede_power9_stack_smoke.yaml" \
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

# Preserve generated cases and logs on failure. After full validation, remove
# only transient SmartSim workers and TensorBoard events to keep storage small.
rm -rf -- "${work_dir}/rl_training" "${work_dir}/tensorboard"

echo "MCC BEDE POWER9/V100 ONE-GPU STACK SMOKE: SUCCESS"
