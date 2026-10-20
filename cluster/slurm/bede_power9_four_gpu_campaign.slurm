#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-design
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=mcc-power9-four-gpu-campaign-%j.out
#SBATCH --error=mcc-power9-four-gpu-campaign-%j.err

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: sbatch bede_power9_four_gpu_campaign.slurm [CAMPAIGN_YAML] [MACHINE_YAML]" >&2
    exit 2
fi

: "${MCC_STORAGE_ROOT:?Set MCC_STORAGE_ROOT to your user directory under /nobackup}"
: "${MCC_DESIGN_SNAPSHOTS_DIR:?Set MCC_DESIGN_SNAPSHOTS_DIR to exactly eight design snapshots}"
: "${MCC_ARTIFACT_ROOT:?Set MCC_ARTIFACT_ROOT to durable campaign storage}"

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
campaign="${1:-${project_root}/configs/campaigns/four_gpu_design.yaml}"
machine="${2:-${project_root}/configs/machines/bede_power9_four_gpu.yaml}"
platform_root="${MCC_STORAGE_ROOT}/ppc64le"
conda_base="${MCC_CONDA_BASE:-${platform_root}/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/mcc-stack-probe}"
export NEKRS_HOME="${NEKRS_HOME:-${platform_root}/nekrs-mhd}"

for path in \
    "${project_root}/.git" \
    "${campaign}" \
    "${machine}" \
    "${MCC_DESIGN_SNAPSHOTS_DIR}" \
    "${conda_base}/etc/profile.d/conda.sh" \
    "${conda_env}/include/client.h" \
    "${conda_env}/lib/libsmartredis.so" \
    "${NEKRS_HOME}/bin/nrsmpi"
do
    [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

snapshot_count="$(find "${MCC_DESIGN_SNAPSHOTS_DIR}" -maxdepth 1 -type f -name 'blanket0.f*' | wc -l)"
[[ "${snapshot_count}" == "8" ]] || {
    echo "Expected exactly eight design snapshots; found ${snapshot_count}" >&2
    exit 1
}

module purge
module load gcc/10.2.0
module load cuda/12.0.1
module load openmpi/4.0.5

source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

[[ "$(uname -m)" == "ppc64le" ]] || {
    echo "Expected ppc64le, found $(uname -m)" >&2
    exit 1
}

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PATH="${NEKRS_HOME}/bin:${PATH}"
export CC="$(command -v mpicc)"
export CXX="$(command -v mpic++)"
export FC="$(command -v mpifort)"
export BLANKET_CMAKE="$(command -v cmake)"
export NEKRS_UDF_INCLUDES="${conda_env}/include;${NEKRS_HOME}/udf"
export NEKRS_UDF_LIBS="${conda_env}/lib/libsmartredis.so"

cache_root="${MCC_CACHE_ROOT:-${platform_root}/cache}"
runtime_lib="${platform_root}/runtime-lib"
mkdir -p \
    "${cache_root}/occa" \
    "${cache_root}/nekrs" \
    "${runtime_lib}" \
    "${MCC_ARTIFACT_ROOT}" \
    "${MCC_ARTIFACT_ROOT}/.matplotlib"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"

export LD_LIBRARY_PATH="${runtime_lib}:${conda_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export MPLCONFIGDIR="${MCC_ARTIFACT_ROOT}/.matplotlib"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))

python - <<'PY'
import platform

import torch

if platform.machine() != "ppc64le":
    raise RuntimeError(f"Expected ppc64le, found {platform.machine()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError(
        f"Expected four visible V100s, found {torch.cuda.device_count()} CUDA devices"
    )
names = [torch.cuda.get_device_name(index) for index in range(4)]
if any("V100" not in name.upper() for name in names):
    raise RuntimeError(f"Expected four V100s, found {names}")
print("Allocated GPUs:", names)
PY

echo "Campaign:  ${campaign}"
echo "Machine:   ${machine}"
echo "Snapshots: ${MCC_DESIGN_SNAPSHOTS_DIR}"
echo "Artifacts: ${MCC_ARTIFACT_ROOT}"
echo "Commit:    $(git -C "${project_root}" rev-parse HEAD)"
echo "Job:       ${SLURM_JOB_ID} on $(hostname)"

cd "${project_root}"
python -u -m magnetoconvective_control.campaign \
    --campaign "${campaign}" \
    --machine "${machine}"
