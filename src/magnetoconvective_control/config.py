"""Configuration loading and environment-variable compatibility mapping."""

from __future__ import annotations

import os
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
    ("smartsim", "interface"): "SMARTSIM_INTERFACE",
    ("smartsim", "port"): "SMARTSIM_PORT",
}


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


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve repository-relative case and data paths."""
    resolved = merge({}, config)
    for section, key in (("case", "executable"), ("data", "snapshots"), ("run", "artifact_root")):
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
