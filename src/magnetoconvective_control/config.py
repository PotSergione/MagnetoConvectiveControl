"""Configuration loading and environment-variable compatibility mapping."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import yaml


ENVIRONMENT_KEYS = {
    ("run", "name"): "RL_MODEL_NAME",
    ("run", "artifact_root"): "MCC_ARTIFACT_ROOT",
    ("data", "snapshots"): "RL_SNAPSHOTS_DIR",
    ("case", "executable"): "RL_SIM_EXECUTABLE",
    ("reward", "reference_flux"): "RL_FIRST_WALL_HEAT_FLUX_BASELINE",
    ("reward", "ewma_timescale"): "RL_EWMA_TIMESCALE",
    ("reward", "scale"): "RL_REWARD_SCALE",
    ("reward", "action_magnitude_regularization"): "RL_ACTION_MAG_REG",
    ("reward", "action_slew_regularization"): "RL_ACTION_SLEW_REG",
    ("ppo", "rollout_steps"): "RL_N_STEPS",
    ("ppo", "batch_size"): "RL_BATCH_SIZE",
    ("ppo", "epochs"): "RL_N_EPOCHS",
    ("ppo", "learning_rate"): "RL_LEARNING_RATE",
    ("training", "episode_steps"): "RL_MAX_EPISODE_STEPS",
    ("training", "total_timesteps"): "RL_TOTAL_TIMESTEPS",
    ("training", "checkpoint_interval"): "RL_CHECKPOINT_INTERVAL",
    ("training", "metrics_interval"): "RL_METRICS_INTERVAL",
    ("launcher", "type"): "RL_LAUNCHER",
    ("launcher", "cpus_per_simulation"): "RL_CPUS_PER_SIMULATION",
    ("launcher", "simulations_per_gpu"): "RL_SIMULATIONS_PER_GPU",
    ("launcher", "gpu_resource"): "RL_GPU_GRES",
    ("launcher", "mpi_command"): "MCC_BEDE_MPIRUN",
    # Scale values are exported after machine defaults so the selected scale
    # has the same precedence in the compatibility environment as in YAML.
    ("parallelism", "physical_environments"): "RL_N_ENVS",
    ("parallelism", "simulations_per_gpu"): "RL_SIMULATIONS_PER_GPU",
    ("resources", "gpus_per_node"): "RL_GPUS_PER_NODE",
    ("smartsim", "interface"): "SMARTSIM_INTERFACE",
    ("smartsim", "port"): "SMARTSIM_PORT",
    ("runtime", "startup_timeout"): "RL_STARTUP_TIMEOUT",
    ("runtime", "occa_cache"): "RL_OCCA_CACHE_DIR",
    ("runtime", "nekrs_cache"): "RL_NEKRS_CACHE_DIR",
}


@dataclass(frozen=True)
class ParallelPlan:
    """Resolved simulation and allocation shape."""

    physical_environments: int
    simulations_per_gpu: int
    required_gpus: int
    nodes: int | None
    gpus_per_node: int


def _nested(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return default
        value = value[component]
    return value


def parallel_plan(config: dict[str, Any]) -> ParallelPlan:
    """Validate and return the physical parallelism requested by a run."""
    physical_environments = int(
        _nested(
            config,
            "parallelism",
            "physical_environments",
            default=os.environ.get("RL_N_ENVS", "1"),
        )
    )
    simulations_per_gpu = int(
        _nested(
            config,
            "parallelism",
            "simulations_per_gpu",
            default=_nested(
                config,
                "launcher",
                "simulations_per_gpu",
                default=os.environ.get("RL_SIMULATIONS_PER_GPU", "1"),
            ),
        )
    )
    nodes_value = _nested(config, "resources", "nodes")
    nodes = int(nodes_value) if nodes_value is not None else None
    gpus_per_node = int(
        _nested(config, "resources", "gpus_per_node", default=1)
    )

    if physical_environments < 1:
        raise ValueError("parallelism.physical_environments must be positive")
    if simulations_per_gpu < 1:
        raise ValueError("parallelism.simulations_per_gpu must be positive")
    if nodes is not None and nodes < 1:
        raise ValueError("resources.nodes must be positive")
    if gpus_per_node < 1:
        raise ValueError("resources.gpus_per_node must be positive")

    required_gpus = ceil(physical_environments / simulations_per_gpu)
    if nodes is not None and required_gpus > nodes * gpus_per_node:
        raise ValueError(
            f"The run needs {required_gpus} GPUs for {physical_environments} "
            f"physical environments, but the scale profile provides only "
            f"{nodes * gpus_per_node}"
        )

    launcher = str(
        _nested(
            config,
            "launcher",
            "type",
            default=os.environ.get("RL_LAUNCHER", "slurm"),
        )
    ).lower()
    if launcher == "bede_mpirun" and gpus_per_node != 1:
        raise ValueError(
            "The bede_mpirun backend maps one physical environment to one "
            "GPU node and therefore requires resources.gpus_per_node=1"
        )
    interface_value = _nested(
        config,
        "smartsim",
        "interface",
        default=os.environ.get("SMARTSIM_INTERFACE", "lo"),
    )
    interface = "" if interface_value is None else str(interface_value).strip()
    effective_nodes = nodes
    if effective_nodes is None:
        effective_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if (
        launcher in {"slurm", "bede_mpirun"}
        and effective_nodes > 1
        and interface.lower()
        in {"", "lo", "none", "change_me", "replace_me"}
    ):
        raise ValueError(
            "Multi-node SmartSim cannot advertise its database on the loopback "
            "interface; set smartsim.interface to a routable compute-node "
            "interface"
        )
    allocated_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    if launcher in {"slurm", "bede_mpirun"} and allocated_nodes is not None:
        allocated_gpus = int(allocated_nodes) * gpus_per_node
        if required_gpus > allocated_gpus:
            raise ValueError(
                f"The active Slurm allocation exposes at most {allocated_gpus} "
                f"GPUs according to the scale profile, but {required_gpus} "
                "are required"
            )

    return ParallelPlan(
        physical_environments=physical_environments,
        simulations_per_gpu=simulations_per_gpu,
        required_gpus=required_gpus,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and reject empty or non-mapping documents."""
    with path.open() as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return content


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def compose(
    experiment: Path,
    root: Path,
    machine: Path | None = None,
    scale: Path | None = None,
) -> dict[str, Any]:
    """Merge experiment, machine, and scale files in increasing precedence."""
    config = load_yaml(experiment.resolve())
    if machine is not None:
        config = merge(config, load_yaml(machine.resolve()))
    if scale is not None:
        config = merge(config, load_yaml(scale.resolve()))
    return resolve_paths(config, root)


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve repository-relative case and data paths."""
    resolved = merge({}, config)
    for section, key in (
        ("case", "executable"),
        ("data", "snapshots"),
        ("run", "artifact_root"),
        ("runtime", "occa_cache"),
        ("runtime", "nekrs_cache"),
    ):
        value = resolved.get(section, {}).get(key)
        if value and not Path(str(value)).expanduser().is_absolute():
            resolved[section][key] = str((root / str(value)).resolve())
    return resolved


def export_compatibility_environment(config: dict[str, Any]) -> None:
    """Expose a resolved configuration to the proven training implementation."""
    for path, environment_key in ENVIRONMENT_KEYS.items():
        value: Any = config
        for component in path:
            if not isinstance(value, dict) or component not in value:
                break
            value = value[component]
        else:
            if value is not None:
                os.environ[environment_key] = str(value)
