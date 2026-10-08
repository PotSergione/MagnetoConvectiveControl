"""Small, restartable paired evaluation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path


SNAPSHOT_PATTERN = "blanket0.f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_time(path: Path) -> float:
    header = path.read_bytes()[:132].decode("ascii", errors="strict")
    fields = header.split()
    if not fields or fields[0] != "#std":
        raise ValueError(f"Not a NekRS checkpoint: {path}")
    return float(fields[7])


def freeze_snapshots(source: Path, destination: Path) -> Path:
    """Copy and hash the prespecified t=310,...,400 evaluation states."""
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, expected_time in zip(range(30, 40), range(310, 401, 10)):
        name = f"{SNAPSHOT_PATTERN}{index:05d}"
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        actual_time = snapshot_time(source_file)
        if abs(actual_time - expected_time) > 1.0e-6:
            raise ValueError(
                f"{source_file} has time {actual_time}, expected {expected_time}"
            )
        target = destination / name
        if not target.exists():
            shutil.copy2(source_file, target)
        source_hash = sha256(source_file)
        if sha256(target) != source_hash:
            raise ValueError(f"Hash mismatch after copying {source_file}")
        records.append(
            {"name": name, "time": actual_time, "sha256": source_hash,
             "bytes": target.stat().st_size}
        )
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"snapshots": records}, indent=2) + "\n")
    return manifest


def build_plan(snapshot_directory: Path, checkpoint: Path, seed: int) -> dict:
    manifest_path = snapshot_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    snapshots = manifest.get("snapshots", [])
    if len(snapshots) != 10:
        raise ValueError("The confirmatory campaign requires exactly ten snapshots")
    for record in snapshots:
        path = snapshot_directory / record["name"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Snapshot does not match its manifest: {path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    rng = random.Random(seed)
    feedback_first = set(rng.sample(range(10), 5))
    runs = []
    for index, record in enumerate(snapshots):
        controllers = (
            ["feedback", "constant_001"]
            if index in feedback_first
            else ["constant_001", "feedback"]
        )
        for controller in controllers:
            runs.append({"snapshot": record["name"], "controller": controller})
    return {
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "snapshot_directory": str(snapshot_directory.resolve()),
        "snapshot_manifest_sha256": sha256(manifest_path),
        "runs": runs,
    }


def _spaces(wall_shape: tuple[int, int]):
    import gymnasium as gym
    import numpy as np

    observation_space = gym.spaces.Dict(
        {
            "heat_flux": gym.spaces.Box(
                -np.inf, np.inf, shape=(3, *wall_shape), dtype=np.float32
            ),
            "previous_actions": gym.spaces.Box(
                -1.0, 1.0, shape=(10,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.Box(-0.25, 0.25, shape=(1,), dtype=np.float32)
    return observation_space, action_space


def run_once(spec: dict, attempt: Path) -> None:
    """Execute one physical trajectory; any termination before the horizon fails."""
    import numpy as np
    from sb3_contrib import RecurrentPPO

    from .envs.smartsim_env import SB3MARLSmartSimVecEnv, SmartSimVecEnv
    from .policies.cnn_lstm import CNNFeatureExtractor

    attempt.mkdir(parents=True, exist_ok=False)
    work = attempt / "work"
    work.mkdir()
    (attempt / "run.json").write_text(json.dumps(spec, indent=2) + "\n")

    old_directory = Path.cwd()
    core = env = None
    rows = []
    try:
        os.chdir(work)
        core = SmartSimVecEnv(
            n_envs=1,
            sim_executable=spec["case_executable"],
            wall_shape=tuple(spec["wall_shape"]),
            launcher=spec["launcher"],
            gpu_gres=spec["gpu_resource"],
            cpus_per_simulation=spec["cpus_per_simulation"],
            database_port=spec["port"],
            database_interface=spec["interface"],
            snapshot_path=spec["snapshot_path"],
            max_episode_steps=None,
            startup_timeout=spec["startup_timeout"],
            action_baseline=0.0,
            action_scale=1.0,
            first_wall_heat_flux_baseline=spec["reference_flux"],
            reward_scale=spec["reward_scale"],
            ewma_timescale=spec["ewma_timescale"],
        )
        observation_space, action_space = _spaces(tuple(spec["wall_shape"]))
        env = SB3MARLSmartSimVecEnv(core, observation_space, action_space)
        observation = env.reset()
        state = None
        episode_start = np.ones(env.num_envs, dtype=bool)
        model = None
        if spec["controller"] == "feedback":
            # Historical checkpoints serialized this class as ``model.*``.
            # Supplying the equivalent package-native kwargs avoids importing
            # the old training directory solely to unpickle that alias.
            policy_kwargs = {
                "features_extractor_class": CNNFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
                "normalize_images": False,
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
                "log_std_init": -2.3,
            }
            model = RecurrentPPO.load(
                spec["checkpoint"], env=env, device="cpu",
                custom_objects={"policy_kwargs": policy_kwargs},
            )

        for step in range(1, spec["horizon"] + 1):
            if model is None:
                action = np.full((env.num_envs, 1), 0.01, dtype=np.float32)
            else:
                action, state = model.predict(
                    observation, state=state, episode_start=episode_start,
                    deterministic=True,
                )
            observation, rewards, dones, infos = env.step(action)
            if np.any(dones):
                raise RuntimeError(f"Environment ended unexpectedly at step {step}")
            info = infos[0]
            row = {
                "step": step,
                "reward": float(rewards[0]),
                "first_wall_heat_flux": info["first_wall_heat_flux"],
                "first_wall_heat_flux_ewma": info["first_wall_heat_flux_ewma"],
                "total_heating_power": info["total_heating_power"],
                "Nu": info["Nu"], "Tbar": info["Tbar"], "Kbar": info["Kbar"],
            }
            row.update(
                {f"action_{i}": float(value)
                 for i, value in enumerate(info["physical_action"])}
            )
            row.update(
                {f"policy_action_{i}": float(value)
                 for i, value in enumerate(info["policy_action"])}
            )
            rows.append(row)
            episode_start[:] = False
    finally:
        if env is not None:
            env.close()
        elif core is not None:
            core.close()
        os.chdir(old_directory)

    trajectory = attempt / "trajectory.csv.partial"
    with trajectory.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(trajectory, attempt / "trajectory.csv")
    summary = {
        "steps": len(rows),
        "return": sum(row["reward"] for row in rows),
        "mean_first_wall_heat_flux": sum(
            row["first_wall_heat_flux"] for row in rows
        ) / len(rows),
    }
    summary_path = attempt / "summary.json.partial"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(summary_path, attempt / "summary.json")
    (attempt / "COMPLETE").touch()


def _next_attempt(run_directory: Path) -> Path:
    existing = sorted(run_directory.glob("attempt_*"))
    return run_directory / f"attempt_{len(existing) + 1:03d}"


def run_campaign(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    proposed = build_plan(args.snapshots.resolve(), args.checkpoint.resolve(), args.seed)
    proposed["protocol"] = {
        "case_executable": str(args.case.resolve()),
        "horizon": args.horizon,
        "wall_shape": [128, 160],
        "launcher": args.launcher,
        "gpu_resource": args.gpu_resource,
        "cpus_per_simulation": args.cpus,
        "port": args.port,
        "interface": args.interface,
        "startup_timeout": args.startup_timeout,
        "reference_flux": 0.0166913,
        "reward_scale": 1.0,
        "ewma_timescale": 10.0,
    }
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan != proposed:
            raise ValueError("Existing campaign plan differs from the requested plan")
    else:
        plan_path.write_text(json.dumps(proposed, indent=2) + "\n")
        plan = proposed
    if args.plan_only:
        print(plan_path)
        return

    for item in plan["runs"]:
        run_directory = output / item["snapshot"] / item["controller"]
        if any(path.name == "COMPLETE" for path in run_directory.glob("attempt_*/COMPLETE")):
            continue
        spec = {
            **item,
            **plan["protocol"],
            "snapshot_path": str(Path(plan["snapshot_directory"]) / item["snapshot"]),
            "checkpoint": plan["checkpoint"],
        }
        attempt = _next_attempt(run_directory)
        command = [sys.executable, "-m", "magnetoconvective_control.evaluation",
                   "single", json.dumps(spec), str(attempt)]
        run_directory.mkdir(parents=True, exist_ok=True)
        with (run_directory / "stdout.log").open("a") as stdout, \
             (run_directory / "stderr.log").open("a") as stderr:
            subprocess.run(command, check=True, stdout=stdout, stderr=stderr)


def analyse(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    differences = []
    snapshots = list(dict.fromkeys(item["snapshot"] for item in plan["runs"]))
    for snapshot in snapshots:
        returns = {}
        for controller in ("feedback", "constant_001"):
            attempts = sorted((output / snapshot / controller).glob("attempt_*"))
            complete = [path for path in attempts if (path / "COMPLETE").exists()]
            if not complete:
                raise ValueError(f"Incomplete result: {snapshot}/{controller}")
            returns[controller] = json.loads(
                (complete[-1] / "summary.json").read_text()
            )["return"]
        differences.append(returns["feedback"] - returns["constant_001"])
    observed = abs(sum(differences) / len(differences))
    null = [abs(sum(sign * value for sign, value in zip(signs, differences)) /
                len(differences)) for signs in product((-1, 1), repeat=len(differences))]
    result = {
        "pairs": len(differences),
        "mean_return_difference": sum(differences) / len(differences),
        "feedback_wins": sum(value > 0 for value in differences),
        "exact_two_sided_p": sum(value >= observed for value in null) / len(null),
        "differences": differences,
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("destination", type=Path)
    campaign = commands.add_parser("run")
    campaign.add_argument("--snapshots", type=Path, required=True)
    campaign.add_argument("--checkpoint", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--case", type=Path, required=True)
    campaign.add_argument("--launcher", choices=("local", "slurm"), default="local")
    campaign.add_argument("--gpu-resource", default="gpu:1")
    campaign.add_argument("--cpus", type=int, default=1)
    campaign.add_argument("--port", type=int, default=6780)
    campaign.add_argument("--interface", default="lo")
    campaign.add_argument("--startup-timeout", type=float, default=500.0)
    campaign.add_argument("--horizon", type=int, default=1000)
    campaign.add_argument("--seed", type=int, default=5489)
    campaign.add_argument("--plan-only", action="store_true")
    analysis = commands.add_parser("analyse")
    analysis.add_argument("output", type=Path)
    single = commands.add_parser("single", help=argparse.SUPPRESS)
    single.add_argument("spec")
    single.add_argument("attempt", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze_snapshots(args.source.resolve(), args.destination.resolve()))
    elif args.command == "run":
        run_campaign(args)
    elif args.command == "analyse":
        analyse(args.output.resolve())
    else:
        run_once(json.loads(args.spec), args.attempt.resolve())


if __name__ == "__main__":
    main()
