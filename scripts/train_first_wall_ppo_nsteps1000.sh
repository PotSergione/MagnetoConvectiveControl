#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${MCC_CONDA_ENV:-smartsim}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}"

export NEKRS_HOME="${NEKRS_HOME:-${HOME}/.local/nekrs-mhd}"
export PATH="${NEKRS_HOME}/bin:${PATH}"
export BLANKET_CMAKE="${BLANKET_CMAKE:-/usr/bin/cmake}"

command -v nrsmpi >/dev/null || {
    echo "ERROR: nrsmpi not found under NEKRS_HOME=${NEKRS_HOME}" >&2
    exit 1
}
command -v mpirun >/dev/null || {
    echo "ERROR: mpirun is required by nrsmpi" >&2
    exit 1
}
nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: no working NVIDIA GPU/driver is visible to NekRS" >&2
    exit 1
}

smartredis_include="${CONDA_PREFIX}/include"
smartredis_library="${CONDA_PREFIX}/lib/libsmartredis.so"
[[ -f "${smartredis_include}/client.h" ]] || {
    echo "ERROR: SmartRedis header not found: ${smartredis_include}/client.h" >&2
    exit 1
}
[[ -f "${smartredis_library}" ]] || {
    echo "ERROR: SmartRedis library not found: ${smartredis_library}" >&2
    exit 1
}

export CC="${CC:-$(command -v mpicc)}"
export CXX="${CXX:-$(command -v mpic++ || command -v mpicxx)}"
export FC="${FC:-$(command -v mpifort)}"
export OMP_NUM_THREADS=1
export NEKRS_UDF_INCLUDES="${smartredis_include}"
export NEKRS_UDF_LIBS="${smartredis_library}"
runtime_library_dir="${root}/.smartredis-lib"
mkdir -p "${runtime_library_dir}"
ln -sfn "${smartredis_library}" "${runtime_library_dir}/libsmartredis.so"
export LD_LIBRARY_PATH="${runtime_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export SMARTSIM_INTERFACE="${SMARTSIM_INTERFACE:-lo}"
export SMARTSIM_PORT="${SMARTSIM_PORT:-16780}"
export RL_LAUNCHER=local
export RL_N_ENVS=1
export RL_SIMULATIONS_PER_GPU=1
export RL_CPUS_PER_SIMULATION=1
export RL_SNAPSHOTS_DIR="${RL_SNAPSHOTS_DIR:-${HOME}/nekMHD/research_tests/fusionB/rl/snapshots}"
export MCC_ARTIFACT_ROOT="${MCC_ARTIFACT_ROOT:-${root}/runs}"
export RL_SIM_EXECUTABLE="${root}/cases/toroidal_gr1e9_ha1750/run_case.sh"
export RL_MODEL_NAME=blanket_first_wall_ppo_nsteps1000_scratch
export RL_ACTION_BASELINE=0.0
export RL_ACTION_SCALE=1.0
export RL_REWARD_SCALE=1.0
export RL_FIRST_WALL_HEAT_FLUX_BASELINE=0.0166913
export RL_EWMA_TIMESCALE=10.0
export RL_ACTION_MAG_REG=0.0
export RL_ACTION_SLEW_REG=0.0
export RL_N_STEPS=1000
export RL_BATCH_SIZE=1000
export RL_N_EPOCHS=5
export RL_LEARNING_RATE=0.0001
export RL_TOTAL_TIMESTEPS=300000
export RL_CHECKPOINT_INTERVAL=25000
export RL_METRICS_INTERVAL=2500
export RL_MAX_EPISODE_STEPS=1000

run_name="${RL_MODEL_NAME}"
run_directory="${MCC_ARTIFACT_ROOT}/${run_name}"
experiment="${root}/configs/experiments/first_wall_ppo_nsteps1000.yaml"

if [[ "${1:-}" == "--dry-run" ]]; then
    [[ -x "${RL_SIM_EXECUTABLE}" ]] || {
        echo "ERROR: case launcher is not executable: ${RL_SIM_EXECUTABLE}" >&2
        exit 1
    }
    snapshot_count="$(find "${RL_SNAPSHOTS_DIR}" -maxdepth 1 -type f -name 'blanket0.f*' | wc -l)"
    [[ "${snapshot_count}" -gt 0 ]] || {
        echo "ERROR: no snapshots found in ${RL_SNAPSHOTS_DIR}" >&2
        exit 1
    }
    python -c 'import sb3_contrib, smartsim, smartredis; import magnetoconvective_control.training.train'
    echo "Model:             ${RL_MODEL_NAME}"
    echo "Snapshots:         ${RL_SNAPSHOTS_DIR} (${snapshot_count})"
    echo "Artifacts:         ${run_directory}"
    echo "Rollout/episode:   ${RL_N_STEPS}/${RL_MAX_EPISODE_STEPS} physical steps"
    echo "Samples/update:    $((RL_N_STEPS * 10))"
    echo "Batch/epochs/LR:   ${RL_BATCH_SIZE}/${RL_N_EPOCHS}/${RL_LEARNING_RATE}"
    echo "Total SB3 samples: ${RL_TOTAL_TIMESTEPS}"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
if [[ -e "${run_directory}/latest.zip" ]]; then
    echo "ERROR: ${run_directory}/latest.zip exists; this run must start from scratch" >&2
    exit 1
fi

mkdir -p "${run_directory}"
cp "${experiment}" "${run_directory}/experiment.yaml"
git rev-parse HEAD > "${run_directory}/git_revision.txt"
echo "Launching from-scratch 1000-step-rollout training"
echo "Artifacts: ${run_directory}"
echo "Log:       ${run_directory}/training.log"
exec python -u -m magnetoconvective_control.training.train \
    > "${run_directory}/training.log" 2>&1
