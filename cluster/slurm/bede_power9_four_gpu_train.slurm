#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-ppo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-19:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=mcc-power9-v100-ppo-%j.out
#SBATCH --error=mcc-power9-v100-ppo-%j.err

set -euo pipefail

storage_root="${MCC_STORAGE_ROOT:-/nobackup/projects/bdman39/sergione}"
project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
platform_root="${storage_root}/ppc64le"
conda_base="${MCC_CONDA_BASE:-${platform_root}/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/mcc-stack-probe}"
nekrs_home="${NEKRS_HOME:-${platform_root}/nekrs-mhd}"
snapshots="${storage_root}/data/snapshots"
experiment="${project_root}/configs/experiments/bede_v100_ppo_200_episode.yaml"
machine="${project_root}/configs/machines/bede_power9_four_gpu.yaml"
scale="${project_root}/configs/scales/bede_power9_four_gpu.yaml"

for path in \
    "${project_root}/.git" \
    "${experiment}" \
    "${machine}" \
    "${scale}" \
    "${snapshots}" \
    "${conda_base}/etc/profile.d/conda.sh" \
    "${conda_env}/include/client.h" \
    "${conda_env}/lib/libsmartredis.so" \
    "${nekrs_home}/bin/nrsmpi"
do
    [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

expected_snapshots=(
    blanket0.f00250 blanket0.f00260 blanket0.f00270
    blanket0.f00280 blanket0.f00290
)
for name in "${expected_snapshots[@]}"; do
    [[ -f "${snapshots}/${name}" ]] || {
        echo "Required snapshot not found: ${snapshots}/${name}" >&2
        exit 1
    }
done
snapshot_count="$(find "${snapshots}" -maxdepth 1 -type f -name 'blanket0.f*' | wc -l)"
[[ "${snapshot_count}" == "5" ]] || {
    echo "Expected exactly five snapshots; found ${snapshot_count}" >&2
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
export PATH="${nekrs_home}/bin:${PATH}"
export NEKRS_HOME="${nekrs_home}"
export CC="$(command -v mpicc)"
export CXX="$(command -v mpic++)"
export FC="$(command -v mpifort)"
export BLANKET_CMAKE="$(command -v cmake)"
export NEKRS_UDF_INCLUDES="${conda_env}/include;${nekrs_home}/udf"
export NEKRS_UDF_LIBS="${conda_env}/lib/libsmartredis.so"
export MPLCONFIGDIR="${storage_root}/training/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

cache_root="${MCC_CACHE_ROOT:-${platform_root}/cache}"
runtime_lib="${platform_root}/runtime-lib"
mkdir -p "${cache_root}/occa" "${cache_root}/nekrs" "${runtime_lib}"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"
export LD_LIBRARY_PATH="${runtime_lib}:${conda_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))

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

echo "Experiment: ${experiment}"
echo "Snapshots:  ${snapshots}"
echo "Artifacts:  /nobackup/projects/bdman39/sergione/training"
echo "Commit:     $(git -C "${project_root}" rev-parse HEAD)"
echo "Job:        ${SLURM_JOB_ID} on $(hostname)"

cd "${project_root}"
python -u -m magnetoconvective_control.cli.train \
    --experiment "${experiment}" \
    --machine "${machine}" \
    --scale "${scale}"
