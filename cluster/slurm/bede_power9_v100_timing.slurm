#!/usr/bin/env bash

#SBATCH --job-name=mcc-p9-time
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=mcc-power9-v100-timing-%j.out
#SBATCH --error=mcc-power9-v100-timing-%j.err

set -euo pipefail

: "${MCC_STORAGE_ROOT:?Set MCC_STORAGE_ROOT to your user directory under /nobackup}"

project_root="${MCC_PROJECT_ROOT:-${HOME}/MagnetoConvectiveControl}"
timing_envs="${MCC_TIMING_ENVS:-1}"
[[ "${timing_envs}" == "1" || "${timing_envs}" == "4" ]] || {
    echo "MCC_TIMING_ENVS must be 1 or 4" >&2
    exit 1
}
platform_root="${MCC_STORAGE_ROOT}/ppc64le"
conda_base="${MCC_CONDA_BASE:-${platform_root}/miniconda}"
conda_env="${MCC_CONDA_ENV:-${conda_base}/envs/mcc-stack-probe}"
export NEKRS_HOME="${NEKRS_HOME:-${platform_root}/nekrs-mhd}"

for path in \
    "${project_root}/data/snapshots/blanket0.f00260" \
    "${conda_base}/etc/profile.d/conda.sh" \
    "${conda_env}/include/client.h" \
    "${conda_env}/lib/libsmartredis.so" \
    "${NEKRS_HOME}/bin/nrsmpi"
do
    [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
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
if [[ "${timing_envs}" == "1" ]]; then
    work_name="mcc-power9-v100-timing-${SLURM_JOB_ID}"
else
    work_name="mcc-power9-${timing_envs}gpu-timing-${SLURM_JOB_ID}"
fi
work_dir="${MCC_STORAGE_ROOT}/smoke/${work_name}"
mkdir -p "${cache_root}/occa" "${cache_root}/nekrs" "${runtime_lib}" "${work_dir}"
ln -sfn "${conda_env}/lib/libsmartredis.so" "${runtime_lib}/libsmartredis.so"

export LD_LIBRARY_PATH="${runtime_lib}:${conda_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RL_OCCA_CACHE_DIR="${cache_root}/occa"
export RL_NEKRS_CACHE_DIR="${cache_root}/nekrs"
export SMARTSIM_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MCC_TIMING_REPORT="${work_dir}/timing.json"
export MCC_TIMING_PROJECT_ROOT="${project_root}"
export MCC_TIMING_ENVS="${timing_envs}"

echo "Host:      $(hostname)"
echo "Commit:    $(git -C "${project_root}" rev-parse HEAD)"
echo "Work:      ${work_dir}"
echo "Workers:   ${timing_envs}"
echo "Snapshot:  ${project_root}/data/snapshots/blanket0.f00260"
nvidia-smi -L

cd "${work_dir}"

python - <<'PY'
import json
import os
import time
from pathlib import Path

import numpy as np

from magnetoconvective_control.envs.smartsim_env import SmartSimVecEnv

transitions = 100
warmup = 10
n_envs = int(os.environ["MCC_TIMING_ENVS"])
root = Path(os.environ["MCC_TIMING_PROJECT_ROOT"])
start = time.perf_counter()
core = None

try:
    core = SmartSimVecEnv(
        n_envs=n_envs,
        simulations_per_gpu=1,
        sim_executable=str(root / "cases/toroidal_gr1e9_ha1750/run_case.sh"),
        wall_shape=(128, 160),
        launcher="local",
        gpu_gres="gpu:1",
        cpus_per_simulation=4,
        max_episode_steps=transitions + 1,
        snapshot_path=str(root / "data/snapshots/blanket0.f00260"),
        startup_timeout=1200,
        database_port=int(os.environ["SMARTSIM_PORT"]),
        database_interface="lo",
        occa_cache_dir=os.environ["RL_OCCA_CACHE_DIR"],
        nekrs_cache_dir=os.environ["RL_NEKRS_CACHE_DIR"],
    )
    core.reset()
    initialization = time.perf_counter() - start
    initial_physical_times = core._last_times.copy()

    durations = []
    actions = np.zeros((n_envs, 10), dtype=np.float64)
    for _ in range(transitions):
        step_start = time.perf_counter()
        _, _, dones, _ = core.step(actions)
        durations.append(time.perf_counter() - step_start)
        if bool(np.any(dones)):
            raise RuntimeError("Timing environment ended before 100 transitions")

    simulated_times = core._last_times - initial_physical_times
finally:
    shutdown_start = time.perf_counter()
    if core is not None:
        core.close()
    shutdown = time.perf_counter() - shutdown_start

steady = np.asarray(durations[warmup:], dtype=float)
steady_mean = float(steady.mean())
projected_block = initialization + sum(durations[:warmup]) + (1000 - warmup) * steady_mean + shutdown
report = {
    "physical_environments": n_envs,
    "transitions": transitions,
    "simulated_time": float(simulated_times.mean()),
    "simulated_time_min": float(simulated_times.min()),
    "simulated_time_max": float(simulated_times.max()),
    "probe_seconds": initialization + sum(durations) + shutdown,
    "initialization_seconds": initialization,
    "first_10_transition_seconds": sum(durations[:warmup]),
    "steady_transition_seconds_mean": steady_mean,
    "steady_transition_seconds_median": float(np.median(steady)),
    "steady_transition_seconds_p95": float(np.quantile(steady, 0.95)),
    "shutdown_seconds": shutdown,
    "projected_1000_transition_block_hours": projected_block / 3600.0,
    "projected_26_block_hours": 26.0 * projected_block / 3600.0,
}
Path(os.environ["MCC_TIMING_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

if np.max(np.abs(simulated_times - 10.0)) > 0.11:
    raise RuntimeError(
        f"Expected 10 nondimensional time units, got {simulated_times.tolist()}"
    )
PY

# Keep only the compact timing report after a successful run. On failure the
# SmartSim directory remains available for diagnosis.
rm -rf -- "${work_dir}/rl_training"

echo "Timing report: ${MCC_TIMING_REPORT}"
echo "MCC BEDE POWER9/V100 TIMING PROBE: SUCCESS"
