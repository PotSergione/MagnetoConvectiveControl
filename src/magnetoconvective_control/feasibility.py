"""Five-snapshot, four-V100 control-feasibility campaign.

This deliberately small campaign asks whether a recurrent modal controller can
learn a sustained improvement over the inherited uniform +0.01 reference.  It
is a feasibility gate, not a confirmatory generalisation experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .campaign import (
    N_ELECTRODES,
    AllocationPause,
    ModalCampaignVecEnv,
    Runtime,
    TimeBudget,
    TrajectoryWriter,
    _atomic_copy,
    _atomic_json,
    _attempt_seconds,
    _completed_attempt,
    _core_kwargs,
    _git_revision,
    _json_hash,
    _multisine_sequences,
    _next_attempt,
    _safe_name,
    _sha256,
    _stage_c_rollout,
    _transition_row,
    _utc,
    _visible_gpu_count,
    electrode_to_modal,
    modal_to_electrode,
    real_cyclic_fourier_basis,
)
from .config import load_yaml, merge, resolve_paths
from .evaluation import snapshot_time


def _training_snapshot_groups(snapshots: Sequence[Path]) -> list[list[Path]]:
    """Return five four-field rollouts, omitting each field exactly once."""
    if len(snapshots) != 5:
        raise ValueError("The feasibility design requires exactly five snapshots")
    groups = []
    for omitted in range(5):
        group = [path for index, path in enumerate(snapshots) if index != omitted]
        groups.append(group[omitted % 4 :] + group[: omitted % 4])
    return groups


def _identification_layout(snapshots: Sequence[Path]) -> list[list[Path]]:
    if len(snapshots) != 5:
        raise ValueError("The feasibility design requires exactly five snapshots")
    return [list(snapshots[:4]), [snapshots[4], snapshots[0], snapshots[2], snapshots[3]]]


def _evaluation_layout(snapshots: Sequence[Path]) -> list[list[tuple[Path, str]]]:
    """Pair every field once; use the final two slots only as diagnostics."""
    if len(snapshots) != 5:
        raise ValueError("The feasibility design requires exactly five snapshots")
    return [
        [(snapshots[0], "learned"), (snapshots[0], "reference"),
         (snapshots[1], "learned"), (snapshots[1], "reference")],
        [(snapshots[2], "learned"), (snapshots[2], "reference"),
         (snapshots[3], "learned"), (snapshots[3], "reference")],
        [(snapshots[4], "learned"), (snapshots[4], "reference"),
         (snapshots[4], "learned_repeat_diagnostic"),
         (snapshots[4], "reference_repeat_diagnostic")],
    ]


def _validate_config(config: dict[str, Any]) -> None:
    required = (
        "run", "case", "data", "campaign", "reward", "reference",
        "identification", "stage_c", "evaluation", "launcher", "smartsim",
    )
    missing = [name for name in required if name not in config]
    if int(config.get("schema_version", 0)) != 1 or missing:
        raise ValueError(f"Invalid feasibility configuration; missing={missing}")
    campaign = config["campaign"]
    if int(campaign["design_snapshots"]) != 5:
        raise ValueError("The feasibility campaign requires exactly five snapshots")
    if int(campaign["transitions"]) != 1000:
        raise ValueError("Every physical trajectory must contain 1000 transitions")
    if float(campaign["allocation_hours"]) != 43.0:
        raise ValueError("The frozen Bede protocol uses a 43-hour allocation")
    if float(campaign["reserve_hours"]) != 5.0:
        raise ValueError("The frozen Bede protocol keeps a five-hour reserve")
    identification = config["identification"]
    if int(identification["blocks"]) != 2:
        raise ValueError("The frozen feasibility design uses two identification blocks")
    if (
        int(identification["warmup_steps"])
        + int(identification["excitation_steps"])
        + int(identification["tail_steps"])
        != 1000
    ):
        raise ValueError("Identification warmup, excitation, and tail must sum to 1000")
    if int(config["stage_c"]["rollouts"]) != 5:
        raise ValueError("The frozen feasibility design uses five PPO rollouts")
    if (
        int(config["evaluation"]["paired_blocks"]) != 3
        or int(config["evaluation"]["primary_pairs"]) != 5
    ):
        raise ValueError("The frozen feasibility design uses three blocks and five pairs")
    reference = float(config["reference"]["action"])
    low, high = map(float, campaign["safe_action_range"])
    if not low < reference < high:
        raise ValueError("The inherited reference must have modal headroom")
    executable = Path(config["case"]["executable"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Case executable is missing or not executable: {executable}")


def _snapshot_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(config["data"]["snapshots"]).expanduser().resolve()
    requested = config["data"].get("snapshot_names") or []
    files = [directory / name for name in requested] if requested else sorted(
        path for path in directory.iterdir()
        if path.is_file() and re.fullmatch(r"blanket0\.f\d+", path.name)
    )
    if len(files) != 5:
        raise ValueError(f"Expected exactly five snapshots, found {len(files)} in {directory}")
    records = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "name": path.name, "path": str(path.resolve()),
            "time": snapshot_time(path), "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return records


def _proposed_plan(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    identity = merge({}, config)
    identity.get("smartsim", {}).pop("port", None)
    case_dir = Path(config["case"]["executable"]).parent
    source_paths = [
        root / "src/magnetoconvective_control/feasibility.py",
        root / "src/magnetoconvective_control/campaign.py",
        root / "src/magnetoconvective_control/envs/smartsim_env.py",
        root / "src/magnetoconvective_control/policies/cnn_lstm.py",
    ]
    return {
        "schema_version": 1,
        "name": _safe_name(config["run"]["name"]),
        "purpose": "feasibility_not_confirmatory_inference",
        "configuration_sha256": _json_hash(identity),
        "git": {"commit": _git_revision(root)["commit"]},
        "snapshots": _snapshot_records(config),
        "case_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(case_dir.iterdir())
            if path.is_file() and path.suffix in {".par", ".re2", ".udf"}
        ],
        "code_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "fourier_basis": real_cyclic_fourier_basis().tolist(),
        "blocks": {"identification": 2, "training": 5, "evaluation": 3},
        "primary_evaluation_pairs": 5,
        "minimum_two_sided_exact_sign_test_p": 0.0625,
    }


def _plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    value.pop("git", None)
    return value


def _verify_plan(plan: dict[str, Any]) -> None:
    for record in plan["snapshots"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Frozen snapshot changed: {path}")


def _preflight(runtime: Runtime) -> None:
    path = runtime.output / "preflight.json"
    if path.is_file() and json.loads(path.read_text()).get("passed"):
        return
    imports = {}
    for name in ("gymnasium", "sb3_contrib", "smartredis", "smartsim", "torch"):
        try:
            module = __import__(name)
            imports[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            imports[name] = f"ERROR: {error}"
    problems = [name for name, value in imports.items() if str(value).startswith("ERROR:")]
    gpu_count = _visible_gpu_count()
    if gpu_count != 4:
        problems.append(f"expected four visible GPUs, found {gpu_count}")
    for command in ("nrsmpi", "mpirun"):
        if shutil.which(command) is None:
            problems.append(f"missing command: {command}")
    report = {
        "created_utc": _utc(), "passed": not problems, "problems": problems,
        "hostname": platform.node(), "visible_gpus": gpu_count,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "imports": imports,
    }
    _atomic_json(path, report)
    if problems:
        raise RuntimeError("Feasibility preflight failed: " + "; ".join(problems))


def _wall_modal_projection(raw_maps: np.ndarray) -> np.ndarray:
    """Compress three wall maps to 30 cyclic coefficients without raw storage."""
    maps = np.asarray(raw_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] != 3:
        raise ValueError(f"Expected three wall maps, got {maps.shape}")
    z_profile = maps.mean(axis=1)
    sectors = np.asarray([
        chunk.mean(axis=1) for chunk in np.array_split(z_profile, N_ELECTRODES, axis=1)
    ]).T
    return np.concatenate([electrode_to_modal(wall) for wall in sectors])


def _actions_for_identification(
    config: dict[str, Any], block: int, reversed_flags: Sequence[bool],
) -> np.ndarray:
    identification = config["identification"]
    warmup = int(identification["warmup_steps"])
    excitation = int(identification["excitation_steps"])
    transitions = int(config["campaign"]["transitions"])
    reference = float(config["reference"]["action"])
    target = float(identification["target_perturbation_rms"])
    low, high = map(float, config["campaign"]["safe_action_range"])
    result = np.full((4, transitions, N_ELECTRODES), reference, dtype=np.float64)
    for env_id in range(4):
        sequence = _multisine_sequences(
            int(config["campaign"]["seed"]) + 1000 * block + 37 * env_id,
            N_ELECTRODES, excitation,
        )
        modal = sequence * target / math.sqrt(N_ELECTRODES)
        if reversed_flags[env_id]:
            modal *= -1.0
        physical = modal_to_electrode(modal)
        peak = float(np.max(np.abs(physical)))
        headroom = min(high - reference, reference - low)
        if peak > headroom:
            physical *= headroom / peak
        result[env_id, warmup : warmup + excitation] += physical
    return result


def _run_open_loop(
    runtime: Runtime, stage: str, block: int, snapshots: Sequence[Path],
    arms: Sequence[str], actions: np.ndarray, *, projections: bool = False,
) -> Path:
    from .envs.smartsim_env import SmartSimVecEnv

    directory = runtime.output / stage / f"block_{block:03d}"
    complete = _completed_attempt(directory)
    if complete is not None:
        return complete
    if (
        len(snapshots) != 4 or len(arms) != 4 or actions.ndim != 3
        or actions.shape[0] != 4 or actions.shape[2] != N_ELECTRODES
    ):
        raise ValueError("Open-loop blocks require four environments")
    attempt = _next_attempt(directory)
    started = time.monotonic()
    attempt.mkdir(parents=True)
    _atomic_json(attempt / "spec.json", {
        "stage": stage, "block": block, "snapshots": [path.name for path in snapshots],
        "arms": list(arms), "actions_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
    })
    work = attempt / "work"
    work.mkdir()
    writers = [
        TrajectoryWriter(
            attempt / arms[index], runtime.config["campaign"]["raw_chunk_steps"],
            store_raw_maps=False,
        )
        for index in range(4)
    ]
    projection_streams = []
    projection_writers = []
    if projections:
        fields = ["transition", *(f"wall_{wall}_modal_{mode}" for wall in range(3) for mode in range(10))]
        for index in range(4):
            stream = (attempt / arms[index] / "wall_modal_projections.csv.partial").open("w", newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            projection_streams.append(stream)
            projection_writers.append(writer)
    old_directory = Path.cwd()
    core = None
    try:
        os.chdir(work)
        kwargs = _core_kwargs(runtime, snapshots[0], max_episode_steps=actions.shape[1])
        kwargs["snapshot_schedules"] = [[str(path)] for path in snapshots]
        kwargs["episode_horizon_truncated"] = False
        core = SmartSimVecEnv(**kwargs)
        core.reset()
        for transition in range(actions.shape[1]):
            _, rewards, dones, infos = core.step(actions[:, transition])
            if transition + 1 < actions.shape[1] and np.any(dones):
                raise RuntimeError(f"Environment ended early at transition {transition + 1}")
            for env_id in range(4):
                applied = np.asarray(infos[env_id]["applied_physical_action"])
                if not np.allclose(applied, actions[env_id, transition], atol=1e-12, rtol=0):
                    raise RuntimeError(f"Environment {env_id} applied the wrong action")
                row = _transition_row(
                    runtime.plan["name"], stage, block, arms[env_id],
                    snapshots[env_id].name, infos[env_id], rewards[env_id],
                    electrode_to_modal(applied), applied,
                    done=bool(dones[env_id]),
                    truncated=bool(infos[env_id].get("TimeLimit.truncated", False)),
                )
                raw = (
                    infos[env_id]["terminal_raw_wall_maps"]
                    if dones[env_id] and "terminal_raw_wall_maps" in infos[env_id]
                    else core._last_raw_wall_maps[env_id]
                )
                writers[env_id].append(row, raw)
                if projections:
                    values = _wall_modal_projection(raw)
                    projection_writers[env_id].writerow({
                        "transition": transition + 1,
                        **{
                            f"wall_{wall}_modal_{mode}": float(values[wall * 10 + mode])
                            for wall in range(3) for mode in range(10)
                        },
                    })
        for index, writer in enumerate(writers):
            writer.complete()
            if projections:
                stream = projection_streams[index]
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                partial = attempt / arms[index] / "wall_modal_projections.csv.partial"
                os.replace(partial, partial.with_suffix(""))
        _atomic_json(attempt / "attempt_summary.json", {
            "completed_utc": _utc(), "elapsed_seconds": time.monotonic() - started,
            "physical_trajectories": 4, "physical_transitions": 4 * actions.shape[1],
            "gpu_hours": 4.0 * (time.monotonic() - started) / 3600.0,
        })
        (attempt / "COMPLETE").touch()
        return attempt
    except BaseException as error:
        for writer in writers:
            writer.abort()
        for stream in projection_streams:
            if not stream.closed:
                stream.close()
        _atomic_json(attempt / "failure.json", {
            "time_utc": _utc(), "type": type(error).__name__, "message": str(error),
        })
        raise
    finally:
        if core is not None:
            core.close()
        os.chdir(old_directory)
        if (attempt / "COMPLETE").is_file():
            shutil.rmtree(work, ignore_errors=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _run_identification(runtime: Runtime) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_a_path = runtime.output / "decisions/stage_a.json"
    decision_b_path = runtime.output / "decisions/stage_b.json"
    if decision_a_path.is_file() and decision_b_path.is_file():
        return json.loads(decision_a_path.read_text()), json.loads(decision_b_path.read_text())
    reference = float(runtime.config["reference"]["action"])
    decision_a = {
        "schema_version": 1, "created_utc": _utc(),
        "reference_action": reference,
        "basis": "inherited_uniform_reference_without_static_search",
    }
    _atomic_json(decision_a_path, decision_a)
    snapshots = [Path(record["path"]) for record in runtime.plan["snapshots"]]
    layouts = _identification_layout(snapshots)
    completed: list[Path] = []
    for block, block_snapshots in enumerate(layouts):
        if _completed_attempt(runtime.output / "identification" / f"block_{block:03d}") is None:
            runtime.budget.require(10 - block, f"identification block {block}")
        reversed_flags = [False] * 4 if block == 0 else [False, True, True, True]
        actions = _actions_for_identification(runtime.config, block, reversed_flags)
        started = time.monotonic()
        attempt = _run_open_loop(
            runtime, "identification", block, block_snapshots,
            [f"id_{block}_{index}" for index in range(4)], actions,
            projections=True,
        )
        runtime.budget.record(
            f"identification:{block}", _attempt_seconds(attempt, time.monotonic() - started)
        )
        completed.append(attempt)

    warmup = int(runtime.config["identification"]["warmup_steps"])
    excitation = int(runtime.config["identification"]["excitation_steps"])
    maximum_lag = int(runtime.config["identification"]["maximum_impulse_lag"])
    by_snapshot: dict[str, list[np.ndarray]] = {}
    scalar_values: dict[str, list[float]] = {
        "first_wall_heat_flux": [], "first_wall_heat_flux_ewma": [],
    }
    trajectory_evidence = []
    for block, (attempt, block_snapshots) in enumerate(zip(completed, layouts)):
        for env_id, snapshot in enumerate(block_snapshots):
            arm = f"id_{block}_{env_id}"
            rows = _read_csv(attempt / arm / "trajectory.csv")
            projection_rows = _read_csv(attempt / arm / "wall_modal_projections.csv")
            physical = np.asarray([
                [float(row[f"physical_action_{mode}"]) for mode in range(10)] for row in rows
            ])
            inputs = electrode_to_modal(physical) - np.array([reference] + [0.0] * 9)
            outputs = np.asarray([
                [float(row[f"wall_{wall}_modal_{mode}"]) for wall in range(3) for mode in range(10)]
                for row in projection_rows
            ])
            baseline = outputs[max(0, warmup - 50):warmup].mean(axis=0)
            outputs = outputs - baseline
            scores = np.zeros(10)
            signs = np.zeros(10)
            lags = np.zeros(10, dtype=int)
            for mode in range(10):
                x = inputs[warmup:warmup + excitation, mode]
                y = outputs[warmup:, mode::10].mean(axis=1)
                correlations = []
                for lag in range(maximum_lag + 1):
                    count = min(excitation, len(y) - lag)
                    denominator = float(np.dot(x[:count], x[:count])) + 1e-30
                    correlations.append(float(np.dot(x[:count], y[lag:lag + count]) / denominator))
                best = int(np.argmax(np.abs(correlations)))
                noise = float(np.std(outputs[:warmup, mode::10])) + 1e-12
                scores[mode] = abs(correlations[best]) / noise
                signs[mode] = np.sign(correlations[best])
                lags[mode] = best
            by_snapshot.setdefault(snapshot.name, []).append(np.column_stack((scores, signs, lags)))
            for name in scalar_values:
                scalar_values[name].extend(float(row[name]) for row in rows)
            trajectory_evidence.append({
                "block": block, "environment": env_id, "snapshot": snapshot.name,
                "mode_snr": scores.tolist(), "mode_sign": signs.tolist(),
                "peak_lag": lags.tolist(),
            })

    per_snapshot = {
        name: np.mean(np.stack(values), axis=0) for name, values in by_snapshot.items()
    }
    evidence = np.stack(list(per_snapshot.values()))
    median_snr = np.median(evidence[:, :, 0], axis=0)
    sign_consistency = np.maximum(
        np.sum(evidence[:, :, 1] > 0, axis=0),
        np.sum(evidence[:, :, 1] < 0, axis=0),
    )
    median_lag = np.median(evidence[:, :, 2], axis=0).astype(int)
    threshold = float(runtime.config["identification"]["minimum_median_snr"])
    candidates = [
        mode for mode in range(1, 10)
        if median_snr[mode] >= threshold and sign_consistency[mode] >= 3
    ]
    candidates.sort(key=lambda mode: median_snr[mode], reverse=True)
    retained = [0] + candidates[: max(0, int(runtime.config["identification"]["maximum_retained_modes"]) - 1)]
    scalings = {}
    for name, values in scalar_values.items():
        array = np.asarray(values)
        scalings[name] = {"mean": float(array.mean()), "std": max(float(array.std()), 1e-8)}
    decision_b = {
        "schema_version": 1, "created_utc": _utc(),
        "identification_passed": True,
        "interpretation": "screening_only_five_correlated_restart_fields",
        "retained_modes": retained,
        "median_mode_snr": median_snr.tolist(),
        "sign_consistency_out_of_five": sign_consistency.tolist(),
        "median_peak_lag": median_lag.tolist(),
        "trajectory_evidence": trajectory_evidence,
        "action_hold_base_transitions": 1,
        "episode_policy_steps": int(runtime.config["campaign"]["transitions"]),
        "reward_ewma_timescale_base_transitions": float(runtime.config["reward"]["ewma_timescale"]),
        "gamma": float(runtime.config["stage_c"]["gamma"]),
        "gae_lambda": float(runtime.config["stage_c"]["gae_lambda"]),
        "observation": {
            "scaling": "per_frame", "wall_mean": 0.0, "wall_std": 1.0,
            "scalars": ["first_wall_heat_flux", "first_wall_heat_flux_ewma"],
            "scalar_scaling": scalings,
        },
    }
    _atomic_json(decision_b_path, decision_b)
    return decision_a, decision_b


def _run_training(
    runtime: Runtime, decision_a: dict[str, Any], decision_b: dict[str, Any]
) -> Path:
    snapshots = [Path(record["path"]) for record in runtime.plan["snapshots"]]
    groups = _training_snapshot_groups(snapshots)
    latest = runtime.output / "stage_c/latest.zip"
    for rollout, group in enumerate(groups):
        directory = runtime.output / "stage_c" / f"rollout_{rollout:03d}"
        complete = _completed_attempt(directory)
        if complete is not None:
            checkpoint = complete / "model_after.zip"
            if not latest.is_file() or _sha256(latest) != _sha256(checkpoint):
                _atomic_copy(checkpoint, latest)
            runtime.budget.record(f"training:{rollout}", _attempt_seconds(complete, 3600.0))
            continue
        prior = [
            _completed_attempt(runtime.output / "stage_c" / f"rollout_{index:03d}")
            for index in range(rollout)
        ]
        prior = [attempt for attempt in prior if attempt is not None]
        if prior:
            checkpoint = prior[-1] / "model_after.zip"
            if not latest.is_file() or _sha256(latest) != _sha256(checkpoint):
                _atomic_copy(checkpoint, latest)
        runtime.budget.require(8 - rollout, f"training rollout {rollout}")
        started = time.monotonic()
        _stage_c_rollout(runtime, rollout, group, decision_a, decision_b, latest)
        complete = _completed_attempt(directory)
        if complete is None:
            raise RuntimeError(f"Training rollout {rollout} did not commit")
        runtime.budget.record(
            f"training:{rollout}", _attempt_seconds(complete, time.monotonic() - started)
        )
    if not latest.is_file():
        raise RuntimeError("Training did not produce a checkpoint")
    frozen = runtime.output / "evaluation/frozen_policy.zip"
    if not frozen.is_file():
        _atomic_copy(latest, frozen)
    return frozen


def _run_evaluation_block(
    runtime: Runtime, block: int, layout: Sequence[tuple[Path, str]],
    decision_a: dict[str, Any], decision_b: dict[str, Any], checkpoint: Path,
) -> Path:
    from sb3_contrib import RecurrentPPO

    directory = runtime.output / "evaluation" / f"block_{block:03d}"
    complete = _completed_attempt(directory)
    if complete is not None:
        return complete
    attempt = _next_attempt(directory)
    started = time.monotonic()
    attempt.mkdir(parents=True)
    snapshots = [item[0] for item in layout]
    controllers = [item[1] for item in layout]
    arms = [f"{controllers[index]}_{snapshots[index].name}" for index in range(4)]
    _atomic_json(attempt / "spec.json", {
        "block": block, "layout": [
            {"snapshot": snapshots[index].name, "controller": controllers[index]}
            for index in range(4)
        ],
        "checkpoint_sha256": _sha256(checkpoint),
    })
    work = attempt / "work"
    work.mkdir()
    old_directory = Path.cwd()
    env = None
    try:
        os.chdir(work)
        env = ModalCampaignVecEnv.create(
            runtime, block, snapshots, decision_a, decision_b, attempt,
            stage="evaluation", arm_names=arms,
        )
        model = RecurrentPPO.load(str(checkpoint), device="cpu")
        observations = env.reset()
        recurrent_state = None
        episode_starts = np.ones(4, dtype=bool)
        transitions = int(runtime.config["campaign"]["transitions"])
        for transition in range(transitions):
            actions, recurrent_state = model.predict(
                observations, state=recurrent_state,
                episode_start=episode_starts, deterministic=True,
            )
            actions = np.asarray(actions, dtype=np.float64)
            for env_id, controller in enumerate(controllers):
                if controller in {"reference", "reference_repeat_diagnostic"}:
                    actions[env_id] = 0.0
            observations, _, dones, _ = env.step(actions)
            episode_starts = np.asarray(dones, dtype=bool)
            if transition + 1 < transitions and np.any(dones):
                raise RuntimeError(f"Evaluation ended early at transition {transition + 1}")
        if not all(env.completed):
            raise RuntimeError("Evaluation block did not finish four trajectories")
        _atomic_json(attempt / "attempt_summary.json", {
            "completed_utc": _utc(), "elapsed_seconds": time.monotonic() - started,
            "physical_trajectories": 4, "physical_transitions": 4 * transitions,
            "gpu_hours": 4.0 * (time.monotonic() - started) / 3600.0,
        })
        (attempt / "COMPLETE").touch()
        return attempt
    except BaseException as error:
        _atomic_json(attempt / "failure.json", {
            "time_utc": _utc(), "type": type(error).__name__, "message": str(error),
        })
        raise
    finally:
        if env is not None:
            env.close()
        os.chdir(old_directory)
        if (attempt / "COMPLETE").is_file():
            shutil.rmtree(work, ignore_errors=True)


def _exact_sign_test_p(wins: int, pairs: int) -> float:
    extreme = max(wins, pairs - wins)
    return min(1.0, 2.0 * sum(math.comb(pairs, value) for value in range(extreme, pairs + 1)) / 2**pairs)


def _summarise_evaluation(runtime: Runtime) -> dict[str, Any]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    layouts = _evaluation_layout([Path(record["path"]) for record in runtime.plan["snapshots"]])
    for block, layout in enumerate(layouts):
        attempt = _completed_attempt(runtime.output / "evaluation" / f"block_{block:03d}")
        if attempt is None:
            raise RuntimeError(f"Missing evaluation block {block}")
        for snapshot, controller in layout:
            arm = f"{controller}_{snapshot.name}"
            rows = _read_csv(attempt / arm / "trajectory.csv")
            reward = np.asarray([float(row["reward"]) for row in rows])
            flux = np.asarray([float(row["first_wall_heat_flux"]) for row in rows])
            record = {
                "snapshot": snapshot.name, "controller": controller,
                "return": float(reward.sum()),
                "first_200_return": float(reward[:200].sum()),
                "post_200_return": float(reward[200:].sum()),
                "final_400_return": float(reward[-400:].sum()),
                "mean_first_wall_heat_flux": float(flux.mean()),
                "final_400_mean_first_wall_heat_flux": float(flux[-400:].mean()),
                "mean_Nu": float(np.mean([float(row["Nu"]) for row in rows])),
                "mean_Tbar": float(np.mean([float(row["Tbar"]) for row in rows])),
                "mean_Kbar": float(np.mean([float(row["Kbar"]) for row in rows])),
            }
            records[(snapshot.name, controller)] = record
    pairs = []
    for snapshot in (record["name"] for record in runtime.plan["snapshots"]):
        learned = records[(snapshot, "learned")]
        reference = records[(snapshot, "reference")]
        pairs.append({
            "snapshot": snapshot,
            "return_improvement": learned["return"] - reference["return"],
            "first_200_return_improvement": learned["first_200_return"] - reference["first_200_return"],
            "post_200_return_improvement": learned["post_200_return"] - reference["post_200_return"],
            "final_400_return_improvement": learned["final_400_return"] - reference["final_400_return"],
            "first_wall_flux_reduction": reference["mean_first_wall_heat_flux"] - learned["mean_first_wall_heat_flux"],
            "final_400_first_wall_flux_reduction": (
                reference["final_400_mean_first_wall_heat_flux"]
                - learned["final_400_mean_first_wall_heat_flux"]
            ),
        })
    full = np.asarray([pair["return_improvement"] for pair in pairs])
    post = np.asarray([pair["post_200_return_improvement"] for pair in pairs])
    final = np.asarray([pair["final_400_return_improvement"] for pair in pairs])
    final_snapshot = runtime.plan["snapshots"][-1]["name"]
    repeat_diagnostic = {
        "snapshot": final_snapshot,
        "learned_return_repeat_difference": (
            records[(final_snapshot, "learned_repeat_diagnostic")]["return"]
            - records[(final_snapshot, "learned")]["return"]
        ),
        "reference_return_repeat_difference": (
            records[(final_snapshot, "reference_repeat_diagnostic")]["return"]
            - records[(final_snapshot, "reference")]["return"]
        ),
    }
    wins = int(np.sum(full > 0))
    late_wins = int(np.sum(final > 0))
    minimum_wins = int(runtime.config["evaluation"]["minimum_win_count"])
    criteria = {
        "positive_mean_full_episode_improvement": bool(full.mean() > 0),
        "full_episode_wins_at_least_four_of_five": wins >= minimum_wins,
        "positive_mean_post_transient_improvement": bool(post.mean() > 0),
        "positive_mean_final_400_improvement": bool(final.mean() > 0),
        "final_400_wins_at_least_four_of_five": late_wins >= minimum_wins,
    }
    return {
        "schema_version": 1, "created_utc": _utc(),
        "purpose": "feasibility_gate_not_confirmatory_claim",
        "passed": all(criteria.values()), "criteria": criteria,
        "classification": "feasible" if all(criteria.values()) else "not_yet_feasible",
        "pairs": pairs, "all_trajectories": list(records.values()),
        "repeat_diagnostic": repeat_diagnostic,
        "mean_return_improvement": float(full.mean()),
        "mean_post_200_return_improvement": float(post.mean()),
        "mean_final_400_return_improvement": float(final.mean()),
        "wins": wins, "late_wins": late_wins,
        "two_sided_exact_sign_test_p": _exact_sign_test_p(wins, 5),
        "inferential_warning": (
            "The five restart fields are temporally correlated and five pairs cannot "
            "support a conventional two-sided p<0.05 sign test."
        ),
    }


def _run_evaluation(
    runtime: Runtime, decision_a: dict[str, Any], decision_b: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    result_path = runtime.output / "decisions/evaluation.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    layouts = _evaluation_layout([Path(record["path"]) for record in runtime.plan["snapshots"]])
    for block, layout in enumerate(layouts):
        if _completed_attempt(runtime.output / "evaluation" / f"block_{block:03d}") is None:
            runtime.budget.require(3 - block, f"paired evaluation block {block}")
        started = time.monotonic()
        attempt = _run_evaluation_block(
            runtime, block, layout, decision_a, decision_b, checkpoint
        )
        runtime.budget.record(
            f"evaluation:{block}", _attempt_seconds(attempt, time.monotonic() - started)
        )
    result = _summarise_evaluation(runtime)
    _atomic_json(result_path, result)
    return result


def _accounting(runtime: Runtime) -> dict[str, Any]:
    summaries = [
        json.loads(path.read_text()) for path in runtime.output.glob("**/attempt_summary.json")
        if (path.parent / "COMPLETE").is_file()
    ]
    return {
        "physical_trajectories": sum(int(item.get("physical_trajectories", 0)) for item in summaries),
        "physical_control_transitions": sum(int(item.get("physical_transitions", 0)) for item in summaries),
        "GPU_hours": sum(float(item.get("gpu_hours", 0.0)) for item in summaries),
        "distinct_restart_fields": 5,
        "durable_raw_wall_maps": 0,
    }


def _state(runtime: Runtime, state: str, **extra: Any) -> None:
    _atomic_json(runtime.output / "state.json", {
        "state": state, "updated_utc": _utc(), **extra,
    })


def run_feasibility(root: Path, config: dict[str, Any], dry_run: bool = False) -> Path:
    _validate_config(config)
    output = Path(config["run"]["artifact_root"]).expanduser().resolve() / _safe_name(
        config["run"]["name"]
    )
    output.mkdir(parents=True, exist_ok=True)
    proposed = _proposed_plan(root, config)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        if _plan_identity(plan) != _plan_identity(proposed):
            raise ValueError(f"Existing immutable plan differs: {plan_path}")
    else:
        _atomic_json(plan_path, proposed)
        plan = proposed
    _verify_plan(plan)
    resolved = output / "config.resolved.yaml"
    if not resolved.is_file():
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    if dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(f"Campaign directory: {output}")
        print(f"Frozen plan: {plan_path}")
        return output
    budget = TimeBudget(
        output / "time_budget.json", float(config["campaign"]["allocation_hours"]),
        float(config["campaign"]["reserve_hours"]),
    )
    runtime = Runtime(root=root, output=output, config=config, plan=plan, budget=budget)
    _state(runtime, "PREFLIGHT")
    _preflight(runtime)
    _state(runtime, "IDENTIFICATION")
    decision_a, decision_b = _run_identification(runtime)
    _state(runtime, "TRAINING", retained_modes=decision_b["retained_modes"])
    checkpoint = _run_training(runtime, decision_a, decision_b)
    _state(runtime, "PAIRED_EVALUATION")
    evaluation = _run_evaluation(runtime, decision_a, decision_b, checkpoint)
    result = {
        "outcome": "GO" if evaluation["passed"] else "NO_GO",
        "scope": "five_snapshot_control_feasibility",
        "completed_utc": _utc(), "evaluation": "decisions/evaluation.json",
        "frozen_policy": str(checkpoint.relative_to(output)),
        "accounting": _accounting(runtime),
    }
    _atomic_json(output / "campaign_result.json", result)
    _state(runtime, f"COMPLETE_{result['outcome']}", result="campaign_result.json")
    return output


def _termination(signum, _frame) -> None:
    raise KeyboardInterrupt(f"Received allocation signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = resolve_paths(
        merge(load_yaml(args.campaign.resolve()), load_yaml(args.machine.resolve())), root
    )
    if os.environ.get("MCC_ARTIFACT_ROOT"):
        config["run"]["artifact_root"] = str(Path(os.environ["MCC_ARTIFACT_ROOT"]).resolve())
    if os.environ.get("MCC_DESIGN_SNAPSHOTS_DIR"):
        config["data"]["snapshots"] = str(Path(os.environ["MCC_DESIGN_SNAPSHOTS_DIR"]).resolve())
    if os.environ.get("SMARTSIM_PORT"):
        config["smartsim"]["port"] = int(os.environ["SMARTSIM_PORT"])
    output_hint = Path(config["run"]["artifact_root"]) / _safe_name(config["run"]["name"])
    old_term = signal.signal(signal.SIGTERM, _termination)
    old_usr1 = signal.signal(signal.SIGUSR1, _termination)
    try:
        print(run_feasibility(root, config, dry_run=args.dry_run))
    except AllocationPause as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "PAUSED_ALLOCATION", "updated_utc": _utc(), "message": str(error),
            })
        print(f"Campaign paused cleanly: {error}", flush=True)
        raise SystemExit(0) from error
    except KeyboardInterrupt as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "INTERRUPTED", "updated_utc": _utc(), "message": str(error),
            })
        raise SystemExit(99) from error
    except BaseException as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "FAILED", "updated_utc": _utc(),
                "type": type(error).__name__, "message": str(error),
            })
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGUSR1, old_usr1)


if __name__ == "__main__":
    main()
