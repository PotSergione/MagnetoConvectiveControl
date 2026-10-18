#!/usr/bin/env bash

#SBATCH --job-name=mcc-2gpu-smoke
#SBATCH --partition=ghtest
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=mcc-2gpu-smoke-%j.out
#SBATCH --error=mcc-2gpu-smoke-%j.err

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

require_command bede-mpirun
export MCC_BEDE_MPIRUN="$(command -v bede-mpirun)"

# Slurm's aarch64 client commands are unavailable on the GH compute nodes.
# Ask Bede's supported MPI wrapper to run one tagged hostname probe per node.
set +e
host_probe="$("${MCC_BEDE_MPIRUN}" --bede-par 1ppn:none bash -c \
    'printf "__MCC_HOST__%s\n" "$(hostname -s)"' 2>&1)"
host_probe_status=$?
set -e
if [[ ${host_probe_status} -ne 0 ]]; then
    echo "bede-mpirun allocation probe failed with exit ${host_probe_status}" >&2
    echo "${host_probe}" >&2
    exit "${host_probe_status}"
fi
mapfile -t allocation_nodes < <(
    sed -n 's/^.*__MCC_HOST__//p' <<<"${host_probe}" |
        awk '!seen[$0]++'
)
if [[ ${#allocation_nodes[@]} -ne 2 ]]; then
    echo "Expected exactly two allocated nodes, found ${#allocation_nodes[@]}" >&2
    echo "bede-mpirun host probe output:" >&2
    echo "${host_probe}" >&2
    exit 1
fi

controller_node="$(hostname -s)"
peer_node=""
for candidate in "${allocation_nodes[@]}"; do
    if [[ "${candidate}" != "${controller_node}" ]]; then
        peer_node="${candidate}"
        break
    fi
done
if [[ -z "${peer_node}" ]]; then
    echo "Could not identify the peer node from: ${allocation_nodes[*]}" >&2
    exit 1
fi
printf -v MCC_BEDE_HOSTS '%s,' "${allocation_nodes[@]}"
export MCC_BEDE_HOSTS="${MCC_BEDE_HOSTS%,}"

peer_ip="$(getent ahostsv4 "${peer_node}" | awk 'NR == 1 {print $1}')"
if [[ -z "${peer_ip}" ]]; then
    echo "Could not resolve an IPv4 address for ${peer_node}" >&2
    exit 1
fi

route="$(ip -o -4 route get "${peer_ip}")"
export SMARTSIM_INTERFACE="$(
    awk '{
        for (i = 1; i <= NF; i++) {
            if ($i == "dev") {
                print $(i + 1)
                exit
            }
        }
    }' <<<"${route}"
)"
if [[ -z "${SMARTSIM_INTERFACE}" || "${SMARTSIM_INTERFACE}" == "lo" ]]; then
    echo "Could not identify a routable interface from: ${route}" >&2
    exit 1
fi

controller_ip="$(
    ip -o -4 addr show dev "${SMARTSIM_INTERFACE}" scope global |
        awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}'
)"
if [[ -z "${controller_ip}" ]]; then
    echo "Interface ${SMARTSIM_INTERFACE} has no global IPv4 address" >&2
    exit 1
fi

cache_root="${MCC_CACHE_ROOT:-${MCC_STORAGE_ROOT}/smoke/cache}"
runtime_lib="${MCC_STORAGE_ROOT}/smoke/smartredis-lib"
work_dir="${MCC_STORAGE_ROOT}/smoke/mcc-2gpu-stack-${SLURM_JOB_ID}"
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

echo "Controller:         ${controller_node}"
echo "Peer:               ${peer_node}"
echo "Peer address:       ${peer_ip}"
echo "Route:              ${route}"
echo "SmartSim interface: ${SMARTSIM_INTERFACE}"
echo "Controller address: ${controller_ip}"
echo "Worker hosts:       ${MCC_BEDE_HOSTS}"
echo "MPI launcher:       ${MCC_BEDE_MPIRUN}"
echo "Arch:               $(uname -m)"
echo "Commit:             $(git -C "${project_root}" rev-parse HEAD)"
echo "Work:               ${work_dir}"
echo "Artifacts:          ${MCC_ARTIFACT_ROOT}"

# Verify through the same launch mechanism used by the workers that the
# selected interface exists and one allocated GH200 is visible on both nodes.
"${MCC_BEDE_MPIRUN}" \
    --bede-par 1ppn:none \
    -x SMARTSIM_INTERFACE \
    bash -c '
        set -euo pipefail
        test -d "/sys/class/net/${SMARTSIM_INTERFACE}"
        echo "host=$(hostname) interface=${SMARTSIM_INTERFACE}"
        ip -o -4 addr show dev "${SMARTSIM_INTERFACE}" scope global
        nvidia-smi -L
    '

python - <<'PY'
import torch
from importlib.metadata import version

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available on controller:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the controller node's GH200")
print("Controller GPU:", torch.cuda.get_device_name())
print("SmartSim:", version("smartsim"))
print("SmartRedis:", version("smartredis"))
PY

cd "${work_dir}"

mcc-train \
    --experiment "${project_root}/configs/experiments/stack_smoke_two_gpu.yaml" \
    --machine "${project_root}/configs/machines/bede_two_gpu_stack_smoke.yaml" \
    --scale "${project_root}/configs/scales/two_gpu.yaml"

for env_id in 0 1; do
    model_dir="${work_dir}/rl_training/nekrs_env_${env_id}_ep_0"
    test -s "${model_dir}/blanket.log"
done

python - <<'PY'
import json
import os
from pathlib import Path

from sb3_contrib import RecurrentPPO

run = Path(os.environ["MCC_ARTIFACT_ROOT"]) / "bede_two_gpu_stack_smoke"
required = (
    run / "metadata.json",
    run / "config.resolved.yaml",
    run / "run_contract.json",
    run / "action_mapping.json",
    run / "reward_metrics.csv",
    run / "latest.zip",
    run / "checkpoints" / "step_000000000040.zip",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError("Missing smoke-test artifacts: " + ", ".join(missing))

metadata = json.loads((run / "metadata.json").read_text())
parallelism = metadata["parallelism"]
if parallelism["physical_environments"] != 2:
    raise RuntimeError(f"Unexpected parallel plan: {parallelism}")
if parallelism["required_gpus"] != 2:
    raise RuntimeError(f"Unexpected GPU plan: {parallelism}")

model = RecurrentPPO.load(run / "latest.zip", device="cpu")
if model.num_timesteps != 40:
    raise RuntimeError(
        f"Expected a 40-sample PPO update, found {model.num_timesteps} timesteps"
    )

print("Reloaded PPO checkpoint at", model.num_timesteps, "timesteps")
print("Persistent artifacts:", run)
PY

# Preserve generated worker directories on failure. Once both worker logs and
# the PPO checkpoint have been validated, remove only job-local transient data.
rm -rf -- "${work_dir}/rl_training" "${work_dir}/tensorboard"

echo "MCC BEDE TWO-GPU STACK SMOKE: SUCCESS"
