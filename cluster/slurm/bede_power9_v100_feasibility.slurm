#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-feas
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-19:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=mcc-power9-v100-feasibility-%j.out
#SBATCH --error=mcc-power9-v100-feasibility-%j.err

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: sbatch bede_power9_v100_feasibility.slurm [CAMPAIGN_YAML] [MACHINE_YAML]" >&2
    exit 2
fi

: "${MCC_STORAGE_ROOT:?Set MCC_STORAGE_ROOT to your user directory under /nobackup}"

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
campaign="${1:-${project_root}/configs/campaigns/bede_v100_feasibility.yaml}"
machine="${2:-${project_root}/configs/machines/bede_power9_four_gpu.yaml}"
snapshots="${MCC_DESIGN_SNAPSHOTS_DIR:-${MCC_STORAGE_ROOT}/data/snapshots}"
artifacts="${MCC_ARTIFACT_ROOT:-${MCC_STORAGE_ROOT}/campaigns}"
platform_root="${MCC_STORAGE_ROOT}/ppc64le"
conda_base="${MCC_CONDA_BASE:-${platform_root}/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/mcc-stack-probe}"
export NEKRS_HOME="${NEKRS_HOME:-${platform_root}/nekrs-mhd}"

for path in \
    "${project_root}/.git" \
    "${campaign}" \
    "${machine}" \
    "${snapshots}" \
    "${conda_base}/etc/profile.d/conda.sh" \
    "${conda_env}/include/client.h" \
    "${conda_env}/lib/libsmartredis.so" \
    "${NEKRS_HOME}/bin/nrsmpi"
do
    [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

expected=(
    blanket0.f00250 blanket0.f00260 blanket0.f00270
    blanket0.f00280 blanket0.f00290
)
for name in "${expected[@]}"; do
    [[ -f "${snapshots}/${name}" ]] || {
        echo "Required feasibility snapshot not found: ${snapshots}/${name}" >&2
        exit 1
    }
done

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
    "${cache_root}/occa" "${cache_root}/nekrs" "${runtime_lib}" \
    "${artifacts}" "${artifacts}/.matplotlib"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"

export LD_LIBRARY_PATH="${runtime_lib}:${conda_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export MPLCONFIGDIR="${artifacts}/.matplotlib"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MCC_DESIGN_SNAPSHOTS_DIR="${snapshots}"
export MCC_ARTIFACT_ROOT="${artifacts}"

python - <<'PY'
import platform
import torch

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if platform.machine() != "ppc64le" or len(names) != 4:
    raise RuntimeError(f"Expected ppc64le with four GPUs, found {platform.machine()} and {names}")
if any("V100" not in name.upper() for name in names):
    raise RuntimeError(f"Expected four V100s, found {names}")
print("Allocated GPUs:", names)
PY

echo "Campaign:  ${campaign}"
echo "Machine:   ${machine}"
echo "Snapshots: ${snapshots}"
echo "Artifacts: ${artifacts}"
echo "Commit:    $(git -C "${project_root}" rev-parse HEAD)"
echo "Job:       ${SLURM_JOB_ID} on $(hostname)"

cd "${project_root}"
python -u -m magnetoconvective_control.feasibility \
    --campaign "${campaign}" \
    --machine "${machine}"
