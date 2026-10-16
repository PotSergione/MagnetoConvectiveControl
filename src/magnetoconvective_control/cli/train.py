"""Portable training command."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config import compose, export_compatibility_environment, parallel_plan


def _git_revision(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--scale", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    config = compose(args.experiment, root, args.machine, args.scale)
    plan = parallel_plan(config)
    export_compatibility_environment(config)

    run_name = config.get("run", {}).get("name", "unnamed")
    artifact_root = Path(os.environ.get("MCC_ARTIFACT_ROOT", root / "runs"))
    run_directory = artifact_root / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": _git_revision(root),
        "experiment": str(args.experiment.resolve()),
        "machine": str(args.machine.resolve()) if args.machine else None,
        "scale": str(args.scale.resolve()) if args.scale else None,
        "parallelism": {
            "physical_environments": plan.physical_environments,
            "simulations_per_gpu": plan.simulations_per_gpu,
            "required_gpus": plan.required_gpus,
        },
    }
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(
            "Parallel plan: "
            f"{plan.physical_environments} environments on "
            f"{plan.required_gpus} GPUs"
        )
        print(f"Run directory: {run_directory}")
        return

    from ..training.train import main as train

    train()


if __name__ == "__main__":
    main()
