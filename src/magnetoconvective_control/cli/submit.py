"""Submit a configured MagnetoConvectiveControl run to Slurm."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..config import compose, parallel_plan


def _nested(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return default
        value = value[component]
    return value


def _submission_executable(config: dict[str, Any]) -> list[str]:
    configured = _nested(config, "scheduler", "submit_command", default="sbatch")
    if isinstance(configured, str):
        command = shlex.split(configured)
    elif isinstance(configured, list) and all(
        isinstance(component, str) for component in configured
    ):
        command = configured.copy()
    else:
        raise ValueError("scheduler.submit_command must be a string or string list")
    if not command:
        raise ValueError("scheduler.submit_command cannot be empty")
    return command


def build_submission_command(
    config: dict[str, Any],
    root: Path,
    experiment: Path,
    machine: Path | None,
    scale: Path | None,
) -> list[str]:
    """Construct an argv-safe Slurm submission command."""
    plan = parallel_plan(config)
    nodes = plan.nodes
    if nodes is None:
        nodes = (plan.required_gpus + plan.gpus_per_node - 1) // plan.gpus_per_node

    gpu_resource = _nested(config, "resources", "gpu_resource_per_node")
    if gpu_resource is None:
        if plan.gpus_per_node != 1:
            raise ValueError(
                "resources.gpu_resource_per_node is required when a node has "
                "more than one allocated GPU"
            )
        gpu_resource = _nested(
            config, "launcher", "gpu_resource", default="gpu:1"
        )

    command = _submission_executable(config)
    command.extend([f"--nodes={nodes}", f"--gres={gpu_resource}"])
    for key, option in (
        ("account", "--account"),
        ("partition", "--partition"),
        ("time", "--time"),
        ("job_name", "--job-name"),
        ("output", "--output"),
        ("error", "--error"),
    ):
        value = _nested(config, "scheduler", key)
        if value is not None:
            command.append(f"{option}={value}")

    extra_args = _nested(config, "scheduler", "extra_args", default=[])
    if not isinstance(extra_args, list) or not all(
        isinstance(argument, str) for argument in extra_args
    ):
        raise ValueError("scheduler.extra_args must be a list of strings")
    command.extend(extra_args)

    script = Path(
        _nested(
            config,
            "scheduler",
            "script",
            default=root / "cluster" / "slurm" / "train.slurm",
        )
    ).expanduser()
    if not script.is_absolute():
        script = (root / script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Slurm script not found: {script}")

    command.extend([str(script), "--experiment", str(experiment.resolve())])
    if machine is not None:
        command.extend(["--machine", str(machine.resolve())])
    if scale is not None:
        command.extend(["--scale", str(scale.resolve())])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    config = compose(args.experiment, root, args.machine, args.scale)
    command = build_submission_command(
        config, root, args.experiment, args.machine, args.scale
    )
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
