#!/bin/bash --login

#SBATCH --job-name=mcc-4gpu-design
#SBATCH --partition=gpuA
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_80g:4
#SBATCH --ntasks=8
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=four-gpu-campaign-%j.out
#SBATCH --error=four-gpu-campaign-%j.err

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: sbatch four_gpu_campaign.slurm [CAMPAIGN_YAML] [MACHINE_YAML]" >&2
    exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign="${1:-${root}/configs/campaigns/four_gpu_design.yaml}"
machine="${2:-${root}/configs/machines/csf3_four_gpu.yaml}"

: "${MCC_DESIGN_SNAPSHOTS_DIR:?Set MCC_DESIGN_SNAPSHOTS_DIR to exactly eight non-confirmatory restart fields}"
: "${MCC_ARTIFACT_ROOT:?Set MCC_ARTIFACT_ROOT to durable campaign storage}"

# Use the MPI/CUDA toolchain of the CSF3 NekRS build. This is the same
# activation sequence as the working SmartSim smoke and MARL jobs in scratch.
nekrs_activate="${MCC_NEKRS_ACTIVATE:-${HOME}/.local/nekMHD/activate.sh}"
[[ -f "${nekrs_activate}" ]] || {
    echo "ERROR: NekRS activation script not found: ${nekrs_activate}" >&2
    exit 1
}
# shellcheck disable=SC1090
source "${nekrs_activate}"
: "${NEKRS_HOME:?The NekRS activation script did not set NEKRS_HOME}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${MCC_CONDA_ENV:-nekrs-smartsim}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MCC_ARTIFACT_ROOT}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
export NEKRS_HOME
export PATH="${NEKRS_HOME}/bin:${PATH}"
export BLANKET_CMAKE="${BLANKET_CMAKE:-/usr/bin/cmake}"
export OMP_NUM_THREADS=1
export SMARTSIM_INTERFACE=lo
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))

# The Python package lives in nekrs-smartsim, while the headers and shared
# library used to build the UDF are in the separately installed C++ client.
smartredis_root="${SMARTREDIS_ROOT:-${HOME}/.local/smartredis-0.6.1}"
smartredis_include="${smartredis_root}/include"
if [[ -f "${smartredis_root}/lib/libsmartredis.so" ]]; then
    smartredis_library_dir="${smartredis_root}/lib"
elif [[ -f "${smartredis_root}/lib64/libsmartredis.so" ]]; then
    smartredis_library_dir="${smartredis_root}/lib64"
else
    echo "ERROR: libsmartredis.so not found under ${smartredis_root}" >&2
    exit 1
fi
smartredis_library="${smartredis_library_dir}/libsmartredis.so"
[[ -f "${smartredis_include}/client.h" ]] || {
    echo "ERROR: SmartRedis header not found: ${smartredis_include}/client.h" >&2
    exit 1
}
[[ -f "${smartredis_library}" ]] || {
    echo "ERROR: SmartRedis library not found: ${smartredis_library}" >&2
    exit 1
}
export NEKRS_UDF_INCLUDES="${smartredis_include}"
export NEKRS_UDF_LIBS="${smartredis_library}"
export CC="${CC:-$(command -v mpicc)}"
export CXX="${CXX:-$(command -v mpic++ || command -v mpicxx)}"
export FC="${FC:-$(command -v mpifort)}"

export LD_LIBRARY_PATH="${smartredis_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${root}"

command -v nrsmpi >/dev/null || {
    echo "ERROR: nrsmpi not found under NEKRS_HOME=${NEKRS_HOME}" >&2
    exit 1
}
[[ -d "${MCC_DESIGN_SNAPSHOTS_DIR}" ]] || {
    echo "ERROR: design snapshot directory not found: ${MCC_DESIGN_SNAPSHOTS_DIR}" >&2
    exit 1
}
snapshot_count="$(find "${MCC_DESIGN_SNAPSHOTS_DIR}" -maxdepth 1 -type f -name 'blanket0.f*' | wc -l)"
[[ "${snapshot_count}" == "8" ]] || {
    echo "ERROR: expected exactly eight design snapshots; found ${snapshot_count}" >&2
    exit 1
}

[[ "$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES:-}" | sed '/^$/d' | wc -l)" == "4" ]] || {
    echo "ERROR: CUDA_VISIBLE_DEVICES does not expose exactly four devices: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
}

python - <<'PY'
from importlib import import_module

required = ("yaml", "gymnasium", "sb3_contrib", "smartredis", "smartsim", "torch")
missing = []
for name in required:
    try:
        import_module(name)
    except Exception as error:
        missing.append(f"{name}: {error}")
if missing:
    raise SystemExit(
        "ERROR: campaign Python dependencies are unavailable in the active "
        "environment:\n  " + "\n  ".join(missing)
    )
PY

echo "Campaign:  ${campaign}"
echo "Machine:   ${machine}"
echo "Snapshots: ${MCC_DESIGN_SNAPSHOTS_DIR}"
echo "Artifacts: ${MCC_ARTIFACT_ROOT}"
echo "Python:    $(command -v python)"
echo "NekRS:     $(command -v nrsmpi)"
echo "SmartRedis:${smartredis_library}"
echo "Job:       ${SLURM_JOB_ID} on $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "Commit:    $(git rev-parse HEAD)"

python -u -m magnetoconvective_control.campaign \
    --campaign "${campaign}" \
    --machine "${machine}"
