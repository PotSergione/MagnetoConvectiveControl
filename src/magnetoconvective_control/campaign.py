"""Autonomous four-GPU control-design and learnability campaign.

The command owns one immutable plan and advances it through static response,
dynamic identification, and a four-environment PPO gate.  Completed blocks
are immutable and are skipped on restart; failed attempts remain on disk.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .config import load_yaml, merge, resolve_paths
from .evaluation import snapshot_time


N_ELECTRODES = 10
WALL_SHAPE = (128, 160)
TRANSITION_FIELDS = (
    "campaign", "stage", "block", "arm", "snapshot", "allocation_gpu",
    "action_sequence", "episode_step", "solver_step", "physical_time",
    "policy_decision_step", "action_hold_phase",
    "first_wall_heat_flux", "first_wall_heat_flux_ewma",
    "x_plus_integrated_heat_flux", "y_minus_integrated_heat_flux",
    "y_plus_integrated_heat_flux", "total_heating_power", "Nu", "Tbar",
    "Kbar", "redirected_heat_fraction", "action_magnitude_cost",
    "action_slew_cost", "reward", "terminated", "truncated", "failure",
    *(f"modal_{index}" for index in range(N_ELECTRODES)),
    *(f"physical_action_{index}" for index in range(N_ELECTRODES)),
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def _git_revision(root: Path) -> dict[str, Any]:
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


def real_cyclic_fourier_basis() -> np.ndarray:
    """Return ten real cyclic modes, each with unit electrode RMS."""
    electrode = np.arange(N_ELECTRODES, dtype=np.float64)
    modes = [np.ones(N_ELECTRODES)]
    for wave_number in range(1, 5):
        angle = 2.0 * np.pi * wave_number * electrode / N_ELECTRODES
        modes.extend((np.cos(angle), np.sin(angle)))
    modes.append((-1.0) ** electrode)
    basis = np.asarray(modes, dtype=np.float64)
    basis /= np.sqrt(np.mean(np.square(basis), axis=1, keepdims=True))
    gram = basis @ basis.T / N_ELECTRODES
    if not np.allclose(gram, np.eye(N_ELECTRODES), atol=1e-12):
        raise RuntimeError("Cyclic Fourier basis is not RMS-orthonormal")
    return basis


def electrode_to_modal(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions, dtype=np.float64) @ real_cyclic_fourier_basis().T / 10.0


def modal_to_electrode(coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(coefficients, dtype=np.float64) @ real_cyclic_fourier_basis()


def _multisine_sequences(seed: int, columns: int, length: int) -> np.ndarray:
    """Make deterministic, orthogonal, band-limited unit-RMS sequences."""
    periods = np.asarray([5, 10, 20, 40, 80, 160, 320], dtype=np.float64)
    t = np.arange(length, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = None
    best_crest = math.inf
    for _ in range(128):
        candidate = np.empty((length, columns), dtype=np.float64)
        for column in range(columns):
            phases = rng.uniform(0.0, 2.0 * np.pi, periods.size)
            candidate[:, column] = sum(
                np.sin(2.0 * np.pi * t / period + phase)
                for period, phase in zip(periods, phases)
            )
        candidate -= candidate.mean(axis=0)
        q, _ = np.linalg.qr(candidate)
        q *= math.sqrt(length)
        crest = float(np.max(np.abs(q)))
        if crest < best_crest:
            best, best_crest = q, crest
    assert best is not None
    return best


def _safe_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise ValueError(f"Unsafe campaign name: {value!r}")
    return value


class AllocationPause(RuntimeError):
    """The next atomic block does not fit safely in this allocation."""


class TimeBudget:
    def __init__(self, path: Path, hours: float, reserve_hours: float):
        self.path = path
        self.started = time.monotonic()
        self.limit = hours * 3600.0
        self.reserve = reserve_hours * 3600.0
        self.block_seconds: dict[str, float] = {}
        if path.is_file():
            saved = json.loads(path.read_text())
            self.block_seconds = {
                str(key): float(value)
                for key, value in saved.get("completed_block_seconds", {}).items()
            }

    def record(self, block: str, seconds: float) -> None:
        self.block_seconds[block] = float(seconds)
        saved = json.loads(self.path.read_text()) if self.path.is_file() else {}
        saved.update({
            "updated_utc": _utc(),
            "completed_block_seconds": self.block_seconds,
        })
        _atomic_json(self.path, saved)

    def require(self, remaining_blocks: int, label: str) -> None:
        elapsed = time.monotonic() - self.started
        recent = list(self.block_seconds.values())[-4:]
        estimate = max(recent or [1.5 * 3600.0])
        next_block_projection = elapsed + estimate + self.reserve
        all_blocks_projection = elapsed + remaining_blocks * estimate + self.reserve
        _atomic_json(self.path, {
            "updated_utc": _utc(), "elapsed_hours": elapsed / 3600.0,
            "conservative_block_hours": estimate / 3600.0,
            "remaining_blocks": remaining_blocks,
            "next_block_projected_hours": next_block_projection / 3600.0,
            "all_remaining_projected_hours": all_blocks_projection / 3600.0,
            "limit_hours": self.limit / 3600.0, "next": label,
            "completed_block_seconds": self.block_seconds,
        })
        if next_block_projection > self.limit:
            raise AllocationPause(
                f"Pausing before {label}: the next block plus reserve projects "
                f"to {next_block_projection / 3600.0:.2f} h in this "
                f"{self.limit / 3600.0:.2f} h allocation"
            )


class TrajectoryWriter:
    """Atomic compact table plus independently compressed raw-map chunks."""

    def __init__(
        self, directory: Path, chunk_steps: int, store_raw_maps: bool = True
    ):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.table_partial = directory / "trajectory.csv.partial"
        self.stream = self.table_partial.open("w", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=TRANSITION_FIELDS)
        self.writer.writeheader()
        self.chunk_steps = chunk_steps
        self.store_raw_maps = store_raw_maps
        self.maps: list[np.ndarray] = []
        self.chunk_start = 1
        self.rows = 0

    def append(self, row: dict[str, Any], raw_maps: np.ndarray) -> None:
        self.writer.writerow({field: row.get(field, "") for field in TRANSITION_FIELDS})
        self.rows += 1
        if self.store_raw_maps:
            self.maps.append(np.asarray(raw_maps, dtype=np.float32))
            if len(self.maps) >= self.chunk_steps:
                self._flush_maps()

    def _flush_maps(self) -> None:
        if not self.maps:
            return
        end = self.chunk_start + len(self.maps) - 1
        final = self.directory / f"wall_maps_{self.chunk_start:04d}_{end:04d}.npy.gz"
        partial = final.with_name(final.name + ".partial")
        with gzip.open(partial, "wb", compresslevel=4) as stream:
            np.save(stream, np.stack(self.maps), allow_pickle=False)
        os.replace(partial, final)
        self.maps.clear()
        self.chunk_start = end + 1

    def complete(self) -> None:
        self._flush_maps()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.table_partial, self.directory / "trajectory.csv")
        chunks = []
        for path in sorted(self.directory.glob("wall_maps_*.npy.gz")):
            chunks.append({
                "name": path.name, "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        table = self.directory / "trajectory.csv"
        _atomic_json(self.directory / "manifest.json", {
            "schema_version": 1, "maps": {
                "enabled": self.store_raw_maps,
                "dtype": "float32", "per_transition_shape": [3, *WALL_SHAPE],
                "compression": "gzip", "chunks": chunks,
            },
            "trajectory": {
                "name": table.name, "bytes": table.stat().st_size,
                "sha256": _sha256(table),
            },
        })
        _atomic_json(self.directory / "summary.json", {
            "rows": self.rows,
            "raw_map_bytes": sum(item["bytes"] for item in chunks),
        })
        (self.directory / "COMPLETE").touch()

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()


def _next_attempt(block_directory: Path) -> Path:
    attempts = sorted(block_directory.glob("attempt_*"))
    return block_directory / f"attempt_{len(attempts) + 1:03d}"


def _completed_attempt(block_directory: Path) -> Path | None:
    for attempt in sorted(block_directory.glob("attempt_*"), reverse=True):
        if (attempt / "COMPLETE").is_file():
            return attempt
    return None


def _arm_mapping(block: int) -> list[int]:
    """Map allocation-local environment/GPU index to logical arm index."""
    return [((gpu + block) % 4) for gpu in range(4)]


def _transition_row(
    campaign: str, stage: str, block: int, arm: str, snapshot: str,
    info: dict[str, Any], reward: float, modal: np.ndarray, physical: np.ndarray,
    done: bool = False, truncated: bool = False,
) -> dict[str, Any]:
    row = {
        "campaign": campaign, "stage": stage, "block": block, "arm": arm,
        "snapshot": snapshot, "reward": float(reward),
        "terminated": bool(done and not truncated),
        "truncated": truncated, "failure": "",
    }
    row.update({key: info.get(key, "") for key in TRANSITION_FIELDS if key in info})
    row.update({f"modal_{i}": float(modal[i]) for i in range(10)})
    row.update({f"physical_action_{i}": float(physical[i]) for i in range(10)})
    return row


@dataclass
class Runtime:
    root: Path
    output: Path
    config: dict[str, Any]
    plan: dict[str, Any]
    budget: TimeBudget


def _core_kwargs(runtime: Runtime, snapshot: Path, max_episode_steps=None) -> dict[str, Any]:
    config = runtime.config
    launcher = config["launcher"]
    reward = config["reward"]
    runtime_config = config.get("runtime", {})
    return {
        "n_envs": 4,
        "simulations_per_gpu": 1,
        "sim_executable": config["case"]["executable"],
        "wall_shape": WALL_SHAPE,
        "launcher": launcher.get("type", "local"),
        "gpu_gres": launcher.get("gpu_resource", "gpu:1"),
        "cpus_per_simulation": int(launcher.get("cpus_per_simulation", 4)),
        "database_port": int(config["smartsim"]["port"]),
        "database_interface": config["smartsim"].get("interface", "lo"),
        "startup_timeout": float(config["campaign"].get("startup_timeout", 900)),
        "snapshot_schedules": [[str(snapshot)] for _ in range(4)],
        "max_episode_steps": max_episode_steps,
        "first_wall_heat_flux_baseline": float(reward["reference_flux"]),
        "reward_scale": float(reward.get("scale", 1.0)),
        "ewma_timescale": float(reward.get("ewma_timescale", 10.0)),
        "action_magnitude_regularization": 0.0,
        "action_slew_regularization": 0.0,
        "occa_cache_dir": runtime_config.get(
            "occa_cache", os.environ.get("RL_OCCA_CACHE_DIR")
        ),
        "nekrs_cache_dir": runtime_config.get(
            "nekrs_cache", os.environ.get("RL_NEKRS_CACHE_DIR")
        ),
    }


def _run_open_loop_block(
    runtime: Runtime, stage: str, block: int, snapshot: Path,
    arm_names: Sequence[str], action_table: np.ndarray,
) -> Path:
    """Run four protocols from one paired restart field."""
    from .envs.smartsim_env import SmartSimVecEnv

    block_directory = runtime.output / stage / f"block_{block:03d}"
    complete = _completed_attempt(block_directory)
    if complete is not None:
        return complete
    attempt = _next_attempt(block_directory)
    attempt_started = time.monotonic()
    attempt.mkdir(parents=True)
    _atomic_json(attempt / "spec.json", {
        "stage": stage, "block": block, "snapshot": str(snapshot),
        "snapshot_sha256": _sha256(snapshot), "arm_names": list(arm_names),
        "arm_by_allocation_gpu": _arm_mapping(block),
        "action_table_sha256": hashlib.sha256(
            np.ascontiguousarray(action_table).tobytes()
        ).hexdigest(),
    })
    work = attempt / "work"
    work.mkdir()
    store_raw_maps = stage in {"preflight", "stage_b"}
    writers = {
        arm: TrajectoryWriter(
            attempt / arm,
            runtime.config["campaign"]["raw_chunk_steps"],
            store_raw_maps=store_raw_maps,
        )
        for arm in arm_names
    }
    mapping = _arm_mapping(block)
    old_directory = Path.cwd()
    core = None
    try:
        os.chdir(work)
        core = SmartSimVecEnv(**_core_kwargs(runtime, snapshot))
        core.reset()
        for transition in range(action_table.shape[1]):
            physical = np.stack([action_table[arm, transition] for arm in mapping])
            _, rewards, dones, infos = core.step(physical)
            if np.any(dones):
                raise RuntimeError(
                    f"Open-loop environment ended at transition {transition + 1}"
                )
            for env_id, arm_index in enumerate(mapping):
                if not np.allclose(
                    infos[env_id]["applied_physical_action"], physical[env_id],
                    rtol=0.0, atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Environment {env_id} applied an action different from its protocol"
                    )
                arm = arm_names[arm_index]
                action = physical[env_id]
                modal = electrode_to_modal(action)
                row = _transition_row(
                    runtime.plan["name"], stage, block, arm, snapshot.name,
                    infos[env_id], rewards[env_id], modal, action,
                )
                writers[arm].append(row, core._last_raw_wall_maps[env_id])
        for writer in writers.values():
            writer.complete()
        _atomic_json(attempt / "attempt_summary.json", {
            "completed_utc": _utc(),
            "elapsed_seconds": time.monotonic() - attempt_started,
            "physical_trajectories": 4,
            "physical_transitions": int(action_table.shape[1]) * 4,
            "gpu_hours": 4.0 * (time.monotonic() - attempt_started) / 3600.0,
        })
        (attempt / "COMPLETE").touch()
        return attempt
    except BaseException as error:
        for writer in writers.values():
            writer.abort()
        _atomic_json(attempt / "failure.json", {
            "time_utc": _utc(), "type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.monotonic() - attempt_started,
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


def _attempt_seconds(attempt: Path, fallback: float) -> float:
    summary = attempt / "attempt_summary.json"
    if summary.is_file():
        return float(json.loads(summary.read_text())["elapsed_seconds"])
    return fallback


def _paired_stage_rows(runtime: Runtime, stage: str) -> dict[int, dict[str, list[dict[str, str]]]]:
    result = {}
    for block in range(len(runtime.plan["snapshots"])):
        attempt = _completed_attempt(runtime.output / stage / f"block_{block:03d}")
        if attempt is None:
            raise RuntimeError(f"Missing completed {stage} block {block}")
        arms = {}
        for directory in attempt.iterdir():
            if (directory / "trajectory.csv").is_file():
                arms[directory.name] = _read_csv(directory / "trajectory.csv")
        result[block] = arms
    return result


def run_stage_a(runtime: Runtime) -> dict[str, Any]:
    decision_path = runtime.output / "decisions" / "stage_a.json"
    if decision_path.is_file():
        return json.loads(decision_path.read_text())
    amplitudes = np.asarray(runtime.config["stage_a"]["uniform_actions"], dtype=float)
    if amplitudes.shape != (4,):
        raise ValueError("stage_a.uniform_actions must contain four values")
    transitions = int(runtime.config["campaign"]["transitions"])
    table = np.repeat(amplitudes[:, None, None], transitions, axis=1)
    table = np.repeat(table, N_ELECTRODES, axis=2)
    names = [f"A{index}" for index in range(4)]
    for block, record in enumerate(runtime.plan["snapshots"]):
        if _completed_attempt(
            runtime.output / "stage_a" / f"block_{block:03d}"
        ) is None:
            runtime.budget.require(
                (len(runtime.plan["snapshots"]) - block) + 8 + 10,
                f"Stage A block {block}",
            )
        started = time.monotonic()
        attempt = _run_open_loop_block(
            runtime, "stage_a", block, Path(record["path"]), names, table
        )
        runtime.budget.record(
            f"stage_a:{block}",
            _attempt_seconds(attempt, time.monotonic() - started),
        )

    data = _paired_stage_rows(runtime, "stage_a")
    zero_index = int(np.argmin(np.abs(amplitudes)))
    summaries = []
    full_effects: dict[int, list[float]] = {index: [] for index in range(4)}
    final_effects: dict[int, list[float]] = {index: [] for index in range(4)}
    paired_series: dict[int, list[np.ndarray]] = {index: [] for index in range(4)}
    for block, arms in data.items():
        zero = np.asarray([float(row["first_wall_heat_flux"]) for row in arms[names[zero_index]]])
        for index, name in enumerate(names):
            flux = np.asarray([float(row["first_wall_heat_flux"]) for row in arms[name]])
            full = float(np.mean(flux) - np.mean(zero))
            final = float(np.mean(flux[-400:]) - np.mean(zero[-400:]))
            full_effects[index].append(full)
            final_effects[index].append(final)
            paired_series[index].append(flux - zero)
            summaries.append({
                "block": block, "snapshot": runtime.plan["snapshots"][block]["name"],
                "arm": name, "amplitude": float(amplitudes[index]),
                "mean_flux": float(np.mean(flux)), "final_400_mean_flux": float(np.mean(flux[-400:])),
                "paired_full_effect": full, "paired_final_400_effect": final,
            })
    acceptable = [
        index for index in range(4)
        if np.mean(full_effects[index]) <= 0.0 and np.mean(final_effects[index]) <= 0.0
    ]
    selected = min(acceptable, key=lambda index: np.mean(full_effects[index])) if acceptable else zero_index
    effects = np.asarray(full_effects[selected], dtype=float)
    standard_error = float(effects.std(ddof=1) / math.sqrt(effects.size)) if effects.size > 1 else 0.0
    sustained = max(0.0, -float(effects.mean()))
    delta_min = max(standard_error, 0.25 * sustained)
    dynamic = float(runtime.config["stage_b"]["target_perturbation_rms"])
    safe_low, safe_high = map(float, runtime.config["campaign"]["safe_action_range"])
    dynamic = min(dynamic, safe_high - amplitudes[selected], amplitudes[selected] - safe_low)
    if dynamic <= 0:
        raise RuntimeError("No safe dynamic perturbation remains around selected reference action")
    windows = ((0, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1000))
    aggregate_arms = []
    for index, name in enumerate(names):
        series = np.asarray(paired_series[index])
        mean_series = series.mean(axis=0)
        se_series = series.std(axis=0, ddof=1) / math.sqrt(series.shape[0])
        detectable = np.abs(mean_series) > 2.0 * se_series
        first_detectable = None
        for start in range(len(detectable) - 4):
            if np.all(detectable[start:start + 5]):
                first_detectable = start + 1
                break
        reversal = None
        if first_detectable is not None:
            initial_sign = np.sign(mean_series[first_detectable - 1])
            opposite = np.sign(mean_series) == -initial_sign
            for start in range(first_detectable, len(opposite) - 4):
                if np.all(opposite[start:start + 5]):
                    reversal = start + 1
                    break
        aggregate_arms.append({
            "arm": name, "amplitude": float(amplitudes[index]),
            "first_detectable_transition": first_detectable,
            "first_sustained_sign_reversal_transition": reversal,
            "window_paired_flux_effects": [
                {
                    "start_transition": begin + 1, "end_transition": end,
                    "mean": float(series[:, begin:end].mean(axis=1).mean()),
                    "standard_error": float(
                        series[:, begin:end].mean(axis=1).std(ddof=1)
                        / math.sqrt(series.shape[0])
                    ),
                }
                for begin, end in windows
            ],
        })
    decision = {
        "schema_version": 1, "created_utc": _utc(),
        "reference_action": float(amplitudes[selected]),
        "reference_arm": names[selected], "zero_arm": names[zero_index],
        "delta_min_mean_flux": delta_min,
        "paired_standard_error": standard_error,
        "sustained_improvement": sustained,
        "dynamic_perturbation_rms": dynamic,
        "selection_rule": "lowest full-horizon flux among arms non-worse over full and final-400 windows; otherwise zero",
        "blocks": summaries,
        "aggregate_arms": aggregate_arms,
    }
    _atomic_json(decision_path, decision)
    return decision


def _stage_b_actions(runtime: Runtime, decision_a: dict[str, Any]) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    transitions = int(runtime.config["campaign"]["transitions"])
    warmup = int(runtime.config["stage_b"]["warmup_steps"])
    excite = int(runtime.config["stage_b"]["excitation_steps"])
    if warmup + excite > transitions:
        raise ValueError("Stage B warmup plus excitation exceeds trajectory length")
    sequences = _multisine_sequences(
        int(runtime.config["campaign"]["seed"]), N_ELECTRODES, excite
    )
    basis = real_cyclic_fourier_basis()
    reference = float(decision_a["reference_action"])
    target_rms = float(decision_a["dynamic_perturbation_rms"])
    low_modes = [0, 1, 2]
    high_modes = list(range(3, 10))

    def perturbation(modes: list[int]) -> np.ndarray:
        modal = np.zeros((excite, 10), dtype=np.float64)
        modal[:, modes] = sequences[:, modes] * target_rms / math.sqrt(len(modes))
        return modal_to_electrode(modal)

    low, high = perturbation(low_modes), perturbation(high_modes)
    safe_low, safe_high = map(float, runtime.config["campaign"]["safe_action_range"])
    peak = max(float(np.max(np.abs(low))), float(np.max(np.abs(high))))
    headroom = min(safe_high - reference, reference - safe_low)
    if peak > headroom:
        scale = headroom / peak
        low *= scale
        high *= scale
        target_rms *= scale
    table = np.full((4, transitions, 10), reference, dtype=np.float64)
    section = slice(warmup, warmup + excite)
    table[1, section] += low
    table[2, section] -= low
    table[3, section] += high
    metadata = {
        "seed": int(runtime.config["campaign"]["seed"]),
        "periods": [5, 10, 20, 40, 80, 160, 320],
        "sequence_sha256": hashlib.sha256(sequences.tobytes()).hexdigest(),
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "low_modes": low_modes, "high_modes": high_modes,
        "actual_target_modal_rms": target_rms,
        "crest_factor": float(np.max(np.abs(sequences))),
    }
    return [f"B{index}" for index in range(4)], table, metadata


def _load_raw_statistics(runtime: Runtime, stage: str) -> tuple[float, float]:
    count = 0
    total = 0.0
    total_square = 0.0
    for path in sorted((runtime.output / stage).glob("block_*/attempt_*/B*/wall_maps_*.npy.gz")):
        with gzip.open(path, "rb") as stream:
            values = np.load(stream, allow_pickle=False).astype(np.float64)
        count += values.size
        total += float(values.sum())
        total_square += float(np.square(values).sum())
    if count == 0:
        raise RuntimeError("No Stage B raw maps were found")
    mean = total / count
    variance = max(total_square / count - mean * mean, 1e-16)
    return mean, math.sqrt(variance)


def _load_map_chunks(directory: Path) -> np.ndarray:
    chunks = []
    for path in sorted(directory.glob("wall_maps_*.npy.gz")):
        with gzip.open(path, "rb") as stream:
            chunks.append(np.load(stream, allow_pickle=False))
    if not chunks:
        raise RuntimeError(f"No raw wall-map chunks in {directory}")
    return np.concatenate(chunks, axis=0)


def _ridge_rmse(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray, ridge: float = 1e-3,
) -> float:
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-8)
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = float(train_y.mean())
    gram = x_train.T @ x_train
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), x_train.T @ (train_y - y_mean)
    )
    prediction = y_mean + x_test @ coefficients
    return float(np.sqrt(np.mean(np.square(prediction - test_y))))


def _observation_prediction_analysis(
    runtime: Runtime, retained_modes: Sequence[int], fast: int, tail: int,
) -> dict[str, Any]:
    """Blocked causal-prediction comparison for the nested observation sets."""
    rng = np.random.default_rng(int(runtime.config["campaign"]["seed"]))
    projection = rng.normal(size=(3 * 16 * 16, 32)) / math.sqrt(32.0)
    trajectories = []
    for block in range(8):
        attempt = _completed_attempt(runtime.output / "stage_b" / f"block_{block:03d}")
        assert attempt is not None
        for arm in ("B0", "B1", "B2", "B3"):
            directory = attempt / arm
            rows = _read_csv(directory / "trajectory.csv")
            maps = _load_map_chunks(directory).astype(np.float64)
            if maps.shape[0] != len(rows):
                raise RuntimeError(f"Raw-map/table length mismatch in {directory}")
            per_frame = (
                maps - maps.mean(axis=(1, 2, 3), keepdims=True)
            ) / np.maximum(maps.std(axis=(1, 2, 3), keepdims=True), 1e-8)
            per_features = per_frame[:, :, ::8, ::10].reshape(len(rows), -1) @ projection
            fixed_features = maps[:, :, ::8, ::10].reshape(len(rows), -1) @ projection
            scalar = {
                name: np.asarray([float(row[name]) for row in rows])
                for name in (
                    "first_wall_heat_flux", "first_wall_heat_flux_ewma",
                    "Nu", "Tbar", "Kbar",
                )
            }
            modal = np.asarray([
                [float(row[f"modal_{mode}"]) for mode in range(10)] for row in rows
            ])
            trajectories.append({
                "block": block, "per": per_features, "fixed": fixed_features,
                "scalar": scalar, "modal": modal,
            })
    candidate_names = [
        "per_frame_maps", "fixed_maps", "fixed_plus_flux",
        "fixed_plus_flux_ewma", "fixed_plus_flux_ewma_diagnostics",
    ]
    horizons = sorted(set([1, 10, 25, 50, 100, 200, 400, fast, min(tail, 400)]))
    action_lags = (0, 1, 2, 4, 8, 16, 32)
    results: dict[str, Any] = {}
    for horizon in horizons:
        candidate_result = {}
        for candidate, name in enumerate(candidate_names):
            state_parts = [[] for _ in range(8)]
            action_parts = [[] for _ in range(8)]
            target_parts = [[] for _ in range(8)]
            for trajectory in trajectories:
                length = trajectory["modal"].shape[0]
                indices = np.arange(32, length - horizon)
                base = trajectory["per"] if candidate == 0 else trajectory["fixed"]
                fields = [base[indices]]
                if candidate >= 2:
                    fields.append(trajectory["scalar"]["first_wall_heat_flux"][indices, None])
                if candidate >= 3:
                    fields.append(trajectory["scalar"]["first_wall_heat_flux_ewma"][indices, None])
                if candidate >= 4:
                    fields.extend(
                        trajectory["scalar"][key][indices, None]
                        for key in ("Nu", "Tbar", "Kbar")
                    )
                state = np.column_stack(fields)
                action_history = np.column_stack([
                    trajectory["modal"][indices - lag][:, list(retained_modes)]
                    for lag in action_lags
                ])
                block = trajectory["block"]
                state_parts[block].append(state)
                action_parts[block].append(action_history)
                target_parts[block].append(
                    trajectory["scalar"]["first_wall_heat_flux"][indices + horizon]
                )
            state_parts = [np.concatenate(value) for value in state_parts]
            action_parts = [np.concatenate(value) for value in action_parts]
            target_parts = [np.concatenate(value) for value in target_parts]
            state_errors, action_errors = [], []
            for held_out in range(8):
                training = [index for index in range(8) if index != held_out]
                train_state = np.concatenate([state_parts[index] for index in training])
                train_action = np.concatenate([action_parts[index] for index in training])
                train_target = np.concatenate([target_parts[index] for index in training])
                state_errors.append(_ridge_rmse(
                    train_state, train_target, state_parts[held_out], target_parts[held_out]
                ))
                action_errors.append(_ridge_rmse(
                    np.column_stack((train_state, train_action)), train_target,
                    np.column_stack((state_parts[held_out], action_parts[held_out])),
                    target_parts[held_out],
                ))
            state_rmse = float(np.mean(state_errors))
            action_rmse = float(np.mean(action_errors))
            candidate_result[name] = {
                "state_only_rmse": state_rmse,
                "action_conditioned_rmse": action_rmse,
                "relative_action_improvement": (state_rmse - action_rmse) / state_rmse,
            }
        results[str(horizon)] = candidate_result
    fast_key = str(min(horizons, key=lambda value: abs(value - fast)))
    tail_key = str(min(horizons, key=lambda value: abs(value - min(tail, 400))))
    best_fast = min(
        results[fast_key][name]["action_conditioned_rmse"] for name in candidate_names
    )
    best_tail = min(
        results[tail_key][name]["action_conditioned_rmse"] for name in candidate_names
    )
    selected_index = next((
        index for index, name in enumerate(candidate_names)
        if results[fast_key][name]["action_conditioned_rmse"] <= 1.05 * best_fast
        and results[tail_key][name]["action_conditioned_rmse"] <= 1.05 * best_tail
    ), len(candidate_names) - 1)
    selected_name = candidate_names[selected_index]
    causal_improvement = results[tail_key][selected_name]["relative_action_improvement"]
    return {
        "method": "leave-one-restart-field-out ridge prediction on frozen random projections",
        "candidate_order": candidate_names, "horizons": results,
        "selected_index": selected_index + 1, "selected_name": selected_name,
        "fast_horizon": int(fast_key), "tail_horizon": int(tail_key),
        "tail_action_conditioning_improvement": causal_improvement,
        "causal_signal_resolved": causal_improvement > 0.01,
    }


def run_stage_b(runtime: Runtime, decision_a: dict[str, Any]) -> dict[str, Any]:
    decision_path = runtime.output / "decisions" / "stage_b.json"
    if decision_path.is_file():
        return json.loads(decision_path.read_text())
    names, table, sequence_metadata = _stage_b_actions(runtime, decision_a)
    _atomic_json(runtime.output / "manifests" / "stage_b_sequences.json", sequence_metadata)
    for block, record in enumerate(runtime.plan["snapshots"]):
        if _completed_attempt(
            runtime.output / "stage_b" / f"block_{block:03d}"
        ) is None:
            runtime.budget.require(
                (len(runtime.plan["snapshots"]) - block) + 10,
                f"Stage B block {block}",
            )
        started = time.monotonic()
        attempt = _run_open_loop_block(
            runtime, "stage_b", block, Path(record["path"]), names, table
        )
        runtime.budget.record(
            f"stage_b:{block}",
            _attempt_seconds(attempt, time.monotonic() - started),
        )

    data = _paired_stage_rows(runtime, "stage_b")
    impulse_by_block = []
    max_lag = int(runtime.config["stage_b"].get("maximum_impulse_lag", 400))
    warmup = int(runtime.config["stage_b"]["warmup_steps"])
    for block, arms in data.items():
        block_impulses = np.zeros((10, max_lag + 1), dtype=np.float64)
        counts = np.zeros(10, dtype=int)
        for arm_name in ("B1", "B2", "B3"):
            rows = arms[arm_name]
            actions = np.asarray([
                [float(row[f"modal_{mode}"]) for mode in range(10)] for row in rows
            ])
            actions[:, 0] -= float(decision_a["reference_action"])
            flux = np.asarray([float(row["first_wall_heat_flux"]) for row in rows])
            flux -= np.mean(flux[:warmup])
            for mode in range(10):
                u = actions[:, mode]
                energy = float(np.dot(u, u))
                if energy <= 1e-14:
                    continue
                correlation = np.correlate(flux, u, mode="full")[len(u) - 1:]
                block_impulses[mode] += correlation[:max_lag + 1] / energy
                counts[mode] += 1
        block_impulses /= np.maximum(counts[:, None], 1)
        impulse_by_block.append(block_impulses)
    impulses = np.asarray(impulse_by_block)
    mean_impulse = impulses.mean(axis=0)
    frequency_response = np.fft.rfft(mean_impulse, axis=1)
    identification_path = runtime.output / "decisions" / "stage_b_identification.npz"
    identification_partial = identification_path.with_name(
        identification_path.name + ".partial"
    )
    identification_path.parent.mkdir(parents=True, exist_ok=True)
    with identification_partial.open("wb") as stream:
        np.savez_compressed(
            stream, impulse_by_restart_field=impulses,
            mean_impulse_response=mean_impulse,
            frequency_response_real=frequency_response.real,
            frequency_response_imag=frequency_response.imag,
            normalized_frequency=np.fft.rfftfreq(mean_impulse.shape[1]),
        )
    os.replace(identification_partial, identification_path)
    integrated = np.sum(mean_impulse, axis=1)
    uncertainty = impulses.sum(axis=2).std(axis=0, ddof=1) / math.sqrt(impulses.shape[0])
    threshold = float(decision_a["delta_min_mean_flux"])
    retained = [
        mode for mode in range(10)
        if abs(integrated[mode]) > max(threshold, uncertainty[mode])
    ]
    modal_signal_passed = bool(retained)
    if not retained:
        retained = [int(np.argmax(np.abs(integrated)))]
    aggregate = np.sum(np.abs(mean_impulse[retained]), axis=0)
    cumulative = np.cumsum(aggregate)
    if cumulative[-1] <= 0:
        fast, tail, unresolved = 1, max_lag, True
    else:
        fast = int(np.searchsorted(cumulative, 0.1 * cumulative[-1])) + 1
        tail = int(np.searchsorted(cumulative, 0.95 * cumulative[-1])) + 1
        unresolved = tail >= max_lag
    hold = max(1, min(fast // 5, 2))
    transitions = int(runtime.config["campaign"]["transitions"])
    while transitions % hold:
        hold -= 1
    reward_tau = max(1.0, 0.1 * fast)
    gamma = 1.0
    trace_tau = max(float(tail), float(fast))
    gae_lambda = math.exp(-hold / trace_tau) / gamma
    wall_mean, wall_std = _load_raw_statistics(runtime, "stage_b")
    prediction = _observation_prediction_analysis(runtime, retained, fast, tail)
    selected_set = int(prediction["selected_index"])
    scalars = []
    if selected_set >= 3:
        scalars.append("first_wall_heat_flux")
    if selected_set >= 4:
        scalars.append("first_wall_heat_flux_ewma")
    if selected_set >= 5:
        scalars.extend(("Nu", "Tbar", "Kbar"))
    scalar_values = {key: [] for key in scalars}
    for arms in data.values():
        for rows in arms.values():
            for key in scalars:
                scalar_values[key].extend(float(row[key]) for row in rows)
    scalar_scaling = {
        key: {"mean": float(np.mean(values)), "std": max(float(np.std(values)), 1e-12)}
        for key, values in scalar_values.items()
    }
    decision = {
        "schema_version": 1, "created_utc": _utc(),
        "retained_modes": retained,
        "identification_passed": modal_signal_passed and prediction["causal_signal_resolved"],
        "modal_signal_passed": modal_signal_passed,
        "modal_integrated_responses": integrated.tolist(),
        "modal_standard_errors": uncertainty.tolist(),
        "identification_arrays": {
            "path": str(identification_path.relative_to(runtime.output)),
            "sha256": _sha256(identification_path),
        },
        "fast_response_transitions": fast,
        "causal_tail_transitions": tail,
        "causal_tail_unresolved": unresolved,
        "action_hold_base_transitions": hold,
        "policy_decision_interval": 0.1 * hold,
        "episode_policy_steps": transitions // hold,
        "reward_ewma_timescale_base_transitions": reward_tau,
        "gamma": gamma, "gae_lambda": gae_lambda,
        "observation": {
            "scaling": "per_frame" if selected_set == 1 else "fixed",
            "wall_mean": wall_mean, "wall_std": wall_std,
            "scalars": scalars, "scalar_scaling": scalar_scaling,
            "selection": prediction,
        },
        "sequence": sequence_metadata,
    }
    _atomic_json(decision_path, decision)
    return decision


def _atomic_model_save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.zip")
    model.save(str(temporary))
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


class ModalCampaignVecEnv:
    """Factory facade used to defer SB3 imports until Stage C starts."""

    @staticmethod
    def create(
        runtime: Runtime, rollout: int, snapshots: Sequence[Path],
        decision_a: dict[str, Any], decision_b: dict[str, Any], attempt: Path,
        *, stage: str = "stage_c", arm_names: Sequence[str] | None = None,
    ):
        import gymnasium as gym
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv

        from .envs.smartsim_env import SmartSimVecEnv

        selected_modes = tuple(int(value) for value in decision_b["retained_modes"])
        if arm_names is not None and len(arm_names) != 4:
            raise ValueError("arm_names must contain one label for each environment")
        basis = real_cyclic_fourier_basis()[list(selected_modes)]
        safe_low, safe_high = map(float, runtime.config["campaign"]["safe_action_range"])
        reference = float(decision_a["reference_action"])
        denominator = max(float(np.max(np.sum(np.abs(basis), axis=0))), 1.0)
        modal_bound = min(safe_high - reference, reference - safe_low) / denominator
        if modal_bound <= 0:
            raise RuntimeError("Selected reference action has no safe modal headroom")
        observation = decision_b["observation"]
        scalar_names = tuple(observation["scalars"])
        hold = int(decision_b["action_hold_base_transitions"])
        episode_steps = int(runtime.config["campaign"]["transitions"])
        if episode_steps % hold:
            raise ValueError("Action hold must divide the physical episode length")

        core_kwargs = _core_kwargs(runtime, snapshots[0], max_episode_steps=episode_steps)
        core_kwargs.pop("snapshot_schedules")
        core_kwargs.update({
            "snapshot_schedules": [[str(path)] for path in snapshots],
            "ewma_timescale": float(decision_b["reward_ewma_timescale_base_transitions"]),
            "observation_scaling": observation["scaling"],
            "fixed_wall_mean": float(observation["wall_mean"]),
            "fixed_wall_std": float(observation["wall_std"]),
            "episode_horizon_truncated": False,
        })
        core = SmartSimVecEnv(**core_kwargs)

        class _ModalEnv(VecEnv):
            def __init__(self):
                self.core = core
                self.selected_modes = selected_modes
                self.basis = basis
                self.reference = reference
                self.modal_bound = modal_bound
                self.hold = hold
                self.scalar_names = scalar_names
                self.previous_modal = np.zeros((4, len(selected_modes)), dtype=np.float32)
                self.pending_modal = None
                self.pending_physical = None
                self.writers = [
                    TrajectoryWriter(
                        attempt / (
                            str(arm_names[env_id]) if arm_names is not None
                            else f"env_{env_id}_{path.name}"
                        ),
                        runtime.config["campaign"]["raw_chunk_steps"],
                        store_raw_maps=False,
                    )
                    for env_id, path in enumerate(snapshots)
                ]
                self.completed = [False] * 4
                self.decision_step = 0
                observation_space = gym.spaces.Dict({
                    "heat_flux": gym.spaces.Box(
                        -np.inf, np.inf, shape=(3, *WALL_SHAPE), dtype=np.float32
                    ),
                    "previous_actions": gym.spaces.Box(
                        -1.0, 1.0, shape=(len(selected_modes),), dtype=np.float32
                    ),
                    "scalars": gym.spaces.Box(
                        -np.inf, np.inf, shape=(len(scalar_names),), dtype=np.float32
                    ),
                })
                action_space = gym.spaces.Box(
                    -modal_bound, modal_bound,
                    shape=(len(selected_modes),), dtype=np.float32,
                )
                super().__init__(4, observation_space, action_space)

            def _scalars(self) -> np.ndarray:
                values = {
                    "first_wall_heat_flux": self.core._last_first_wall_flux,
                    "first_wall_heat_flux_ewma": self.core._first_wall_flux_hat,
                    "Nu": self.core._last_nu,
                    "Tbar": self.core._last_tbar,
                    "Kbar": self.core._last_kbar,
                }
                columns = []
                for name in self.scalar_names:
                    scaling = observation["scalar_scaling"][name]
                    columns.append(
                        (np.asarray(values[name]) - float(scaling["mean"]))
                        / float(scaling["std"])
                    )
                if not columns:
                    return np.empty((4, 0), dtype=np.float32)
                return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)

            def _observations(self, heat_flux: np.ndarray) -> dict[str, np.ndarray]:
                return {
                    "heat_flux": np.ascontiguousarray(heat_flux, dtype=np.float32),
                    "previous_actions": np.ascontiguousarray(
                        self.previous_modal / self.modal_bound, dtype=np.float32
                    ),
                    "scalars": self._scalars(),
                }

            def reset(self):
                self.previous_modal[:] = 0.0
                return self._observations(self.core.reset())

            def step_async(self, actions):
                modal = np.asarray(actions, dtype=np.float64)
                expected = (4, len(self.selected_modes))
                if modal.shape != expected:
                    raise ValueError(f"Expected modal actions {expected}, got {modal.shape}")
                modal = np.clip(modal, -self.modal_bound, self.modal_bound)
                physical = self.reference + modal @ self.basis
                safe_min, safe_max = map(float, runtime.config["campaign"]["safe_action_range"])
                if np.any(physical < safe_min - 1e-12) or np.any(physical > safe_max + 1e-12):
                    raise RuntimeError("Modal action escaped the predeclared safe range")
                self.pending_modal = modal
                self.pending_physical = physical

            def step_wait(self):
                if self.pending_modal is None:
                    raise RuntimeError("step_wait called without step_async")
                reward_total = np.zeros(4, dtype=np.float32)
                final_observations = None
                final_dones = None
                final_infos = None
                for micro_step in range(self.hold):
                    self.core.step_async(self.pending_physical)
                    heat_flux, rewards, dones, infos = self.core.step_wait()
                    reward_total += rewards
                    for env_id, info in enumerate(infos):
                        full_modal = np.zeros(10, dtype=np.float64)
                        full_modal[list(self.selected_modes)] = self.pending_modal[env_id]
                        full_modal[0] += self.reference
                        raw = (
                            info["terminal_raw_wall_maps"]
                            if dones[env_id] and "terminal_raw_wall_maps" in info
                            else self.core._last_raw_wall_maps[env_id]
                        )
                        row = _transition_row(
                            runtime.plan["name"], stage, rollout,
                            str(arm_names[env_id]) if arm_names is not None else "PPO",
                            snapshots[env_id].name, info, rewards[env_id],
                            full_modal, self.pending_physical[env_id],
                            done=bool(dones[env_id]),
                            truncated=bool(info.get("TimeLimit.truncated", False)),
                        )
                        row["policy_decision_step"] = self.decision_step + 1
                        row["action_hold_phase"] = micro_step + 1
                        self.writers[env_id].append(row, raw)
                    if np.any(dones) and not np.all(dones):
                        raise RuntimeError("Stage C environments lost lockstep episode boundaries")
                    if np.any(dones) and micro_step != self.hold - 1:
                        raise RuntimeError("Physical episode ended inside an action-hold interval")
                    final_observations, final_dones, final_infos = heat_flux, dones, infos
                assert final_observations is not None
                for env_id, done in enumerate(final_dones):
                    final_infos[env_id]["modal_action"] = self.pending_modal[env_id].copy()
                    final_infos[env_id]["aggregated_physical_reward"] = float(reward_total[env_id])
                    if done:
                        terminal_scalars = []
                        for name in self.scalar_names:
                            scaling = observation["scalar_scaling"][name]
                            terminal_scalars.append(
                                (float(final_infos[env_id][name]) - float(scaling["mean"]))
                                / float(scaling["std"])
                            )
                        final_infos[env_id]["terminal_observation"] = {
                            "heat_flux": np.ascontiguousarray(
                                final_infos[env_id]["terminal_observation"], dtype=np.float32
                            ),
                            "previous_actions": np.ascontiguousarray(
                                self.pending_modal[env_id] / self.modal_bound,
                                dtype=np.float32,
                            ),
                            "scalars": np.asarray(terminal_scalars, dtype=np.float32),
                        }
                    if done and not self.completed[env_id]:
                        self.writers[env_id].complete()
                        self.completed[env_id] = True
                next_previous = np.asarray(self.pending_modal, dtype=np.float32)
                next_previous[np.asarray(final_dones, dtype=bool)] = 0.0
                self.previous_modal = next_previous
                observations = self._observations(final_observations)
                self.pending_modal = None
                self.pending_physical = None
                self.decision_step += 1
                return observations, reward_total, final_dones, final_infos

            def close(self):
                for env_id, writer in enumerate(self.writers):
                    if not self.completed[env_id]:
                        writer.abort()
                self.core.close()

            def get_attr(self, attr_name, indices=None):
                return [getattr(self, attr_name) for _ in self._get_indices(indices)]

            def set_attr(self, attr_name, value, indices=None):
                self._get_indices(indices)
                setattr(self, attr_name, value)

            def env_method(self, method_name, *args, indices=None, **kwargs):
                raise NotImplementedError

            def env_is_wrapped(self, wrapper_class, indices=None):
                return [False for _ in self._get_indices(indices)]

        return _ModalEnv()


def _stage_c_rollout(
    runtime: Runtime, rollout: int, snapshots: Sequence[Path],
    decision_a: dict[str, Any], decision_b: dict[str, Any], latest: Path,
) -> None:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback

    from .policies.cnn_lstm import ModalCNNFeatureExtractor

    directory = runtime.output / "stage_c" / f"rollout_{rollout:03d}"
    if _completed_attempt(directory) is not None:
        return
    attempt = _next_attempt(directory)
    attempt_started = time.monotonic()
    attempt.mkdir(parents=True)
    _atomic_json(attempt / "spec.json", {
        "rollout": rollout, "snapshots": [path.name for path in snapshots],
        "policy_checkpoint_before": str(latest) if latest.is_file() else None,
        "decision_a_sha256": _sha256(runtime.output / "decisions" / "stage_a.json"),
        "decision_b_sha256": _sha256(runtime.output / "decisions" / "stage_b.json"),
    })
    work = attempt / "work"
    work.mkdir()
    old_directory = Path.cwd()
    env = None
    try:
        os.chdir(work)
        env = ModalCampaignVecEnv.create(
            runtime, rollout, snapshots, decision_a, decision_b, attempt
        )
        stage_config = runtime.config["stage_c"]
        rollout_samples = 4 * int(decision_b["episode_policy_steps"])
        if rollout_samples % int(stage_config["batch_size"]):
            raise ValueError(
                f"Stage C batch_size={stage_config['batch_size']} does not divide "
                f"the {rollout_samples} physical-environment samples per rollout"
            )
        policy_kwargs = {
            "features_extractor_class": ModalCNNFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
            "lstm_hidden_size": int(stage_config.get("lstm_hidden_size", 128)),
            "n_lstm_layers": 1, "shared_lstm": False,
            "enable_critic_lstm": True, "log_std_init": -2.3,
        }
        if latest.is_file():
            model = RecurrentPPO.load(str(latest), env=env, device="cpu")
        else:
            model = RecurrentPPO(
                "MultiInputLstmPolicy", env,
                n_steps=int(decision_b["episode_policy_steps"]),
                batch_size=int(stage_config["batch_size"]),
                n_epochs=int(stage_config["epochs"]),
                learning_rate=float(stage_config["learning_rate"]),
                gamma=float(decision_b["gamma"]),
                gae_lambda=float(decision_b["gae_lambda"]),
                policy_kwargs=policy_kwargs, device="cpu", verbose=1,
                seed=int(runtime.config["campaign"]["seed"]),
            )
        model.set_random_seed(int(runtime.config["campaign"]["seed"]) + rollout)

        class _Diagnostics(BaseCallback):
            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                import torch as th
                from sb3_contrib.common.recurrent.type_aliases import RNNStates

                buffer = self.model.rollout_buffer
                deterministic_actions = []
                self.model.policy.set_training_mode(False)
                with th.no_grad():
                    for step in range(buffer.buffer_size):
                        observations = {
                            key: th.as_tensor(value[step], device=self.model.device)
                            for key, value in buffer.observations.items()
                        }
                        states = RNNStates(
                            pi=(
                                th.as_tensor(buffer.hidden_states_pi[step], device=self.model.device),
                                th.as_tensor(buffer.cell_states_pi[step], device=self.model.device),
                            ),
                            vf=(
                                th.as_tensor(buffer.hidden_states_vf[step], device=self.model.device),
                                th.as_tensor(buffer.cell_states_vf[step], device=self.model.device),
                            ),
                        )
                        actions, _, _, _ = self.model.policy.forward(
                            observations, states,
                            th.as_tensor(buffer.episode_starts[step], device=self.model.device),
                            deterministic=True,
                        )
                        deterministic_actions.append(actions.cpu().numpy())
                deterministic_actions = np.asarray(deterministic_actions)
                final = attempt / "ppo_rollout.npz"
                partial = final.with_name(final.name + ".partial")
                with partial.open("wb") as stream:
                    np.savez_compressed(
                        stream, actions=buffer.actions, rewards=buffer.rewards,
                        returns=buffer.returns, advantages=buffer.advantages,
                        values=buffer.values, log_probs=buffer.log_probs,
                        episode_starts=buffer.episode_starts,
                        deterministic_actions=deterministic_actions,
                        policy_log_std=self.model.policy.log_std.detach().cpu().numpy(),
                    )
                os.replace(partial, final)
                table = attempt / "ppo_transitions.csv.partial"
                action_dim = buffer.actions.shape[-1]
                fields = [
                    "policy_decision_step", "environment", "reward", "value",
                    "return", "advantage", "log_probability", "episode_start",
                    *(f"sampled_modal_{index}" for index in range(action_dim)),
                    *(f"deterministic_modal_{index}" for index in range(action_dim)),
                    *(f"action_std_{index}" for index in range(action_dim)),
                ]
                standard_deviation = np.exp(
                    self.model.policy.log_std.detach().cpu().numpy()
                )
                with table.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for step in range(buffer.buffer_size):
                        for env_id in range(buffer.n_envs):
                            row = {
                                "policy_decision_step": step + 1, "environment": env_id,
                                "reward": float(buffer.rewards[step, env_id]),
                                "value": float(buffer.values[step, env_id]),
                                "return": float(buffer.returns[step, env_id]),
                                "advantage": float(buffer.advantages[step, env_id]),
                                "log_probability": float(buffer.log_probs[step, env_id]),
                                "episode_start": bool(buffer.episode_starts[step, env_id]),
                            }
                            for index in range(action_dim):
                                row[f"sampled_modal_{index}"] = float(buffer.actions[step, env_id, index])
                                row[f"deterministic_modal_{index}"] = float(
                                    deterministic_actions[step, env_id, index]
                                )
                                row[f"action_std_{index}"] = float(standard_deviation[index])
                            writer.writerow(row)
                os.replace(table, attempt / "ppo_transitions.csv")

        model.learn(
            total_timesteps=4 * int(decision_b["episode_policy_steps"]),
            callback=_Diagnostics(), reset_num_timesteps=False,
        )
        if not all(env.completed):
            raise RuntimeError("PPO rollout ended without four complete physical trajectories")
        checkpoint = attempt / "model_after.zip"
        _atomic_model_save(model, checkpoint)
        logger_values = {
            key: float(value) for key, value in model.logger.name_to_value.items()
            if np.isscalar(value) and np.isfinite(value)
        }
        _atomic_json(attempt / "ppo_metrics.json", logger_values)
        _atomic_json(attempt / "attempt_summary.json", {
            "completed_utc": _utc(),
            "elapsed_seconds": time.monotonic() - attempt_started,
            "physical_trajectories": 4,
            "physical_transitions": int(runtime.config["campaign"]["transitions"]) * 4,
            "virtual_agent_rows": 0,
            "sb3_transitions": 4 * int(decision_b["episode_policy_steps"]),
            "gpu_hours": 4.0 * (time.monotonic() - attempt_started) / 3600.0,
        })
        (attempt / "COMPLETE").touch()
        _atomic_copy(checkpoint, latest)
    except BaseException as error:
        _atomic_json(attempt / "failure.json", {
            "time_utc": _utc(), "type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.monotonic() - attempt_started,
        })
        raise
    finally:
        if env is not None:
            env.close()
        os.chdir(old_directory)
        if (attempt / "COMPLETE").is_file():
            shutil.rmtree(work, ignore_errors=True)


def _analyse_stage_c(runtime: Runtime, decision_a: dict[str, Any], decision_b: dict[str, Any]) -> dict[str, Any]:
    by_snapshot: dict[str, list[dict[str, float]]] = {}
    mode_values: dict[int, list[tuple[int, float]]] = {
        mode: [] for mode in decision_b["retained_modes"]
    }
    kl_values = []
    for rollout in range(int(runtime.config["stage_c"]["rollouts"])):
        attempt = _completed_attempt(runtime.output / "stage_c" / f"rollout_{rollout:03d}")
        if attempt is None:
            raise RuntimeError(f"Missing completed Stage C rollout {rollout}")
        metrics = json.loads((attempt / "ppo_metrics.json").read_text())
        for key, value in metrics.items():
            if key.endswith("approx_kl"):
                kl_values.append(float(value))
        with np.load(attempt / "ppo_rollout.npz", allow_pickle=False) as diagnostics:
            deterministic = diagnostics["deterministic_actions"]
        retained_modes = list(decision_b["retained_modes"])
        for coordinate, mode in enumerate(retained_modes):
            for env_id in range(deterministic.shape[1]):
                mode_values[mode].append((
                    rollout, float(np.mean(deterministic[:, env_id, coordinate]))
                ))
        for trajectory in sorted(attempt.glob("env_*/trajectory.csv")):
            rows = _read_csv(trajectory)
            snapshot = rows[0]["snapshot"]
            rewards = np.asarray([float(row["reward"]) for row in rows])
            by_snapshot.setdefault(snapshot, []).append({
                "rollout": float(rollout), "return": float(rewards.sum()),
                "final_400_return": float(rewards[-400:].sum()),
            })
    x_values, returns, late_returns = [], [], []
    wins = 0
    for records in by_snapshot.values():
        records.sort(key=lambda row: row["rollout"])
        wins += records[-1]["return"] > records[0]["return"]
        mean_return = np.mean([row["return"] for row in records])
        mean_late = np.mean([row["final_400_return"] for row in records])
        for index, row in enumerate(records):
            x_values.append(index - (len(records) - 1) / 2)
            returns.append(row["return"] - mean_return)
            late_returns.append(row["final_400_return"] - mean_late)
    x = np.asarray(x_values)
    slope = float(np.dot(x, returns) / np.dot(x, x))
    late_slope = float(np.dot(x, late_returns) / np.dot(x, x))
    heating = []
    for path in (runtime.output / "stage_c").glob("rollout_*/attempt_*/env_*/trajectory.csv"):
        heating.extend(float(row["total_heating_power"]) for row in _read_csv(path)[::100])
    delta_return = (
        float(decision_a["delta_min_mean_flux"])
        / max(float(np.mean(heating)), 1e-12)
        * int(runtime.config["campaign"]["transitions"])
    )
    modal_changes = {}
    coherent_change = False
    for mode, values in mode_values.items():
        first = np.asarray([value for rollout, value in values if rollout < 2])
        last = np.asarray([value for rollout, value in values if rollout >= 8])
        change = float(last.mean() - first.mean())
        uncertainty = math.sqrt(
            float(first.var(ddof=1) / len(first) + last.var(ddof=1) / len(last))
        ) if len(first) > 1 and len(last) > 1 else math.inf
        modal_changes[str(mode)] = {"change": change, "standard_error": uncertainty}
        coherent_change |= abs(change) > uncertainty
    finite_kl = bool(kl_values) and np.all(np.isfinite(kl_values))
    controlled_kl = finite_kl and 0.0 < float(np.mean(kl_values)) < float(
        runtime.config["stage_c"].get("maximum_mean_kl", 0.05)
    )
    criteria = {
        "positive_snapshot_blocked_return_trend": slope > 0 and 4.0 * slope > delta_return,
        "late_return_not_declining": late_slope >= 0,
        "majority_snapshots_improve": wins > len(by_snapshot) / 2,
        "retained_modal_policy_changes": coherent_change,
        "finite_controlled_kl": controlled_kl,
        "identified_action_predictive_signal_retained": bool(
            decision_b["identification_passed"]
        ),
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1, "created_utc": _utc(), "passed": passed,
        "criteria": criteria, "snapshot_blocked_return_slope_per_occurrence": slope,
        "final_400_return_slope_per_occurrence": late_slope,
        "delta_min_return": delta_return, "improved_snapshots": wins,
        "snapshots": len(by_snapshot), "mean_approx_kl": (
            float(np.mean(kl_values)) if kl_values else None
        ), "modal_changes": modal_changes,
        "classification": "go" if passed else "learnability_gate_failed",
    }
    _atomic_json(runtime.output / "decisions" / "stage_c.json", result)
    return result


def run_stage_c(runtime: Runtime, decision_a: dict[str, Any], decision_b: dict[str, Any]) -> dict[str, Any]:
    result_path = runtime.output / "decisions" / "stage_c.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    snapshots = [Path(record["path"]) for record in runtime.plan["snapshots"]]
    if len(snapshots) != 8:
        raise ValueError("Stage C requires exactly eight design snapshots")
    latest = runtime.output / "stage_c" / "latest.zip"
    rollouts = int(runtime.config["stage_c"]["rollouts"])
    for rollout in range(rollouts):
        current_complete = _completed_attempt(
            runtime.output / "stage_c" / f"rollout_{rollout:03d}"
        )
        if current_complete is not None:
            expected = current_complete / "model_after.zip"
            if not latest.is_file() or _sha256(latest) != _sha256(expected):
                _atomic_copy(expected, latest)
            runtime.budget.record(
                f"stage_c:{rollout}",
                _attempt_seconds(current_complete, 1.5 * 3600.0),
            )
            continue
        completed_models = []
        for previous in range(rollout):
            completed = _completed_attempt(
                runtime.output / "stage_c" / f"rollout_{previous:03d}"
            )
            if completed is not None:
                completed_models.append(completed / "model_after.zip")
        if completed_models:
            expected = completed_models[-1]
            if not latest.is_file() or _sha256(latest) != _sha256(expected):
                _atomic_copy(expected, latest)
        elif latest.is_file():
            orphan = latest.with_name(f"orphaned_latest_{int(time.time())}.zip")
            latest.rename(orphan)
        group = snapshots[:4] if rollout % 2 == 0 else snapshots[4:]
        runtime.budget.require(rollouts - rollout, f"Stage C rollout {rollout}")
        started = time.monotonic()
        _stage_c_rollout(runtime, rollout, group, decision_a, decision_b, latest)
        completed = _completed_attempt(
            runtime.output / "stage_c" / f"rollout_{rollout:03d}"
        )
        runtime.budget.record(
            f"stage_c:{rollout}",
            _attempt_seconds(completed, time.monotonic() - started)
        )
    return _analyse_stage_c(runtime, decision_a, decision_b)


def _final_accounting(runtime: Runtime, decision_b: dict[str, Any] | None) -> dict[str, Any]:
    completed_summaries = []
    for path in runtime.output.glob("**/attempt_summary.json"):
        if (path.parent / "COMPLETE").is_file():
            completed_summaries.append(json.loads(path.read_text()))
    physical_transitions = sum(
        int(item.get("physical_transitions", 0)) for item in completed_summaries
    )
    physical_trajectories = sum(
        int(item.get("physical_trajectories", 0)) for item in completed_summaries
    )
    gpu_hours = sum(float(item.get("gpu_hours", 0.0)) for item in completed_summaries)
    virtual_rows = sum(int(item.get("virtual_agent_rows", 0)) for item in completed_summaries)
    tail = int(decision_b["causal_tail_transitions"]) if decision_b else 1000
    return {
        "CFD_solver_steps": 100 * physical_transitions,
        "physical_control_transitions": physical_transitions,
        "virtual_agent_rows_presented_to_SB3": virtual_rows,
        "completed_physical_trajectories": physical_trajectories,
        "distinct_restart_fields": len(runtime.plan["snapshots"]),
        "approximately_independent_causal_response_windows": (
            physical_transitions / max(tail, 1)
        ),
        "GPU_hours": gpu_hours,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Campaign configuration schema_version must be 1")
    required = (
        "run", "case", "data", "campaign", "reward", "stage_a", "stage_b",
        "stage_c", "launcher", "smartsim",
    )
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Missing campaign configuration sections: {missing}")
    if int(config["campaign"]["design_snapshots"]) != 8:
        raise ValueError("This frozen campaign design requires exactly eight snapshots")
    if int(config["campaign"]["transitions"]) != 1000:
        raise ValueError("Identification and learnability trajectories must have 1000 transitions")
    actions = np.asarray(config["stage_a"]["uniform_actions"], dtype=float)
    if actions.shape != (4,) or not np.any(np.isclose(actions, 0.0)):
        raise ValueError("Stage A requires four static actions including zero")
    safe_low, safe_high = map(float, config["campaign"]["safe_action_range"])
    if safe_low >= safe_high or np.any(actions < safe_low) or np.any(actions > safe_high):
        raise ValueError("Stage A actions must lie inside campaign.safe_action_range")
    hold_candidates = int(config["stage_c"]["batch_size"])
    if hold_candidates < 1:
        raise ValueError("Stage C batch_size must be positive")
    executable = Path(config["case"]["executable"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Case executable is missing or not executable: {executable}")


def _snapshot_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(config["data"]["snapshots"]).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Design snapshot directory not found: {directory}")
    requested = config["data"].get("snapshot_names") or []
    if requested:
        files = [directory / name for name in requested]
    else:
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and re.fullmatch(r"blanket0\.f\d+", path.name)
        )
    count = int(config["campaign"]["design_snapshots"])
    if len(files) != count:
        raise ValueError(
            f"Design snapshot set must resolve to exactly {count} files; found {len(files)}. "
            "Use data.snapshot_names when the directory contains additional fields."
        )
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
    case_directory = Path(config["case"]["executable"]).parent
    case_files = sorted(
        path for path in case_directory.iterdir()
        if path.is_file() and path.suffix in {".par", ".re2", ".udf"}
    )
    code_paths = [
        root / "src" / "magnetoconvective_control" / "campaign.py",
        root / "src" / "magnetoconvective_control" / "envs" / "smartsim_env.py",
        root / "src" / "magnetoconvective_control" / "policies" / "cnn_lstm.py",
    ]
    revision = _git_revision(root)
    identity_config = merge({}, config)
    # SmartRedis port is allocation-local transport state and changes with a
    # requeued Slurm job; it is deliberately excluded from scientific identity.
    identity_config.get("smartsim", {}).pop("port", None)
    return {
        "schema_version": 1,
        "name": _safe_name(config["run"]["name"]),
        "configuration_sha256": _json_hash(identity_config),
        # Dirtiness is execution metadata, not part of immutable identity: a
        # campaign created before committing must still resume after that exact
        # source tree is committed. Source/case hashes retain the audit trail.
        "git": {"commit": revision["commit"]},
        "snapshots": _snapshot_records(config),
        "case_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in case_files
        ],
        "code_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in code_paths
        ],
        "fourier_basis": real_cyclic_fourier_basis().tolist(),
        "accounting": {
            "virtual_agents_per_physical_transition_stage_c": 0,
            "stage_a_physical_trajectories": 32,
            "stage_b_physical_trajectories": 32,
            "stage_c_physical_trajectories": 40,
        },
    }


def _verify_plan(plan: dict[str, Any]) -> None:
    for record in plan["snapshots"]:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen design snapshot changed: {path}")


def _plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    identity = dict(plan)
    identity.pop("git", None)
    return identity


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in {"NoDevFiles", "-1"}:
        return len([item for item in visible.split(",") if item.strip()])
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        )
        return len([line for line in output.splitlines() if line.strip()])
    except (OSError, subprocess.CalledProcessError):
        return 0


def _preflight(runtime: Runtime) -> None:
    report_path = runtime.output / "preflight.json"
    if report_path.is_file() and json.loads(report_path.read_text()).get("passed"):
        return
    imports = {}
    for name in ("gymnasium", "sb3_contrib", "smartredis", "smartsim", "torch"):
        try:
            module = __import__(name)
            imports[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            imports[name] = f"ERROR: {error}"
    gpu_count = _visible_gpu_count()
    problems = [name for name, version in imports.items() if str(version).startswith("ERROR:")]
    if gpu_count != 4:
        problems.append(f"expected exactly four visible GPUs, found {gpu_count}")
    if int(os.environ.get("SLURM_GPUS", "4")) != 4:
        problems.append(f"SLURM_GPUS={os.environ.get('SLURM_GPUS')} rather than 4")
    for command in ("nrsmpi", "mpirun"):
        if shutil.which(command) is None:
            problems.append(f"missing command: {command}")
    report = {
        "created_utc": _utc(), "passed": not problems, "problems": problems,
        "hostname": platform.node(), "visible_gpus": gpu_count,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "imports": imports,
    }
    _atomic_json(report_path, report)
    if problems:
        raise RuntimeError("Campaign preflight failed: " + "; ".join(problems))


def _smoke_test(runtime: Runtime) -> None:
    report_path = runtime.output / "smoke_test.json"
    if report_path.is_file() and json.loads(report_path.read_text()).get("passed"):
        return
    fingerprints = np.asarray([
        np.zeros(10),
        0.001 * real_cyclic_fourier_basis()[0],
        0.001 * real_cyclic_fourier_basis()[1],
        0.001 * real_cyclic_fourier_basis()[9],
    ])
    smoke_steps = 100
    table = np.repeat(fingerprints[:, None, :], smoke_steps, axis=1)
    started = time.monotonic()
    attempt = _run_open_loop_block(
        runtime, "preflight", 0, Path(runtime.plan["snapshots"][0]["path"]),
        [f"P{index}" for index in range(4)], table,
    )
    elapsed = time.monotonic() - started
    checks = {}
    for arm in range(4):
        rows = _read_csv(attempt / f"P{arm}" / "trajectory.csv")
        actions = np.asarray([
            [float(row[f"physical_action_{index}"]) for index in range(10)]
            for row in rows
        ])
        checks[f"P{arm}"] = {
            "rows": len(rows),
            "action_identity": bool(np.allclose(actions, fingerprints[arm], atol=1e-12)),
            "gpu": sorted({int(row["allocation_gpu"]) for row in rows}),
            "raw_chunks": len(list((attempt / f"P{arm}").glob("wall_maps_*.npy.gz"))),
        }
    passed = all(
        value["rows"] == smoke_steps and value["action_identity"] and value["raw_chunks"] > 0
        for value in checks.values()
    )
    report = {
        "created_utc": _utc(), "passed": passed, "elapsed_seconds": elapsed,
        "projected_1000_transition_hours_including_startup": elapsed * 10.0 / 3600.0,
        "checks": checks,
    }
    _atomic_json(report_path, report)
    if not passed:
        raise RuntimeError("Four-environment smoke test failed; see smoke_test.json")


def _state(runtime: Runtime, name: str, **extra: Any) -> None:
    _atomic_json(runtime.output / "state.json", {
        "state": name, "updated_utc": _utc(), **extra,
    })


def run_campaign(root: Path, config: dict[str, Any], dry_run: bool = False) -> Path:
    _validate_config(config)
    artifact_root = Path(config["run"]["artifact_root"]).expanduser().resolve()
    output = artifact_root / _safe_name(config["run"]["name"])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    proposed = _proposed_plan(root, config)
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        if _plan_identity(plan) != _plan_identity(proposed):
            raise ValueError(
                f"Existing immutable campaign plan differs from this invocation: {plan_path}"
            )
    else:
        _atomic_json(plan_path, proposed)
        plan = proposed
    _verify_plan(plan)
    resolved_path = output / "config.resolved.yaml"
    if not resolved_path.exists():
        resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(f"Campaign directory: {output}")
        print(f"Frozen plan: {plan_path}")
        return output

    budget = TimeBudget(
        output / "time_budget.json",
        float(config["campaign"]["allocation_hours"]),
        float(config["campaign"]["reserve_hours"]),
    )
    runtime = Runtime(root=root, output=output, config=config, plan=plan, budget=budget)
    _state(runtime, "PREFLIGHT")
    _preflight(runtime)
    _smoke_test(runtime)
    _state(runtime, "A_RUNNING")
    decision_a = run_stage_a(runtime)
    _state(runtime, "A_DECIDED", decision="decisions/stage_a.json")
    decision_b = run_stage_b(runtime, decision_a)
    _state(runtime, "B_DECIDED", decision="decisions/stage_b.json")
    if not decision_b["identification_passed"]:
        result = {
            "outcome": "NO_GO", "reason": "unresolved_action_effect",
            "completed_utc": _utc(), "stage_a": decision_a, "stage_b": decision_b,
            "accounting": _final_accounting(runtime, decision_b),
        }
        _atomic_json(output / "campaign_result.json", result)
        _state(runtime, "COMPLETE_NO_GO", result="campaign_result.json")
        return output
    _state(runtime, "C_RUNNING")
    decision_c = run_stage_c(runtime, decision_a, decision_b)
    outcome = "GO" if decision_c["passed"] else "NO_GO"
    result = {
        "outcome": outcome, "completed_utc": _utc(),
        "stage_a_decision": "decisions/stage_a.json",
        "stage_b_decision": "decisions/stage_b.json",
        "stage_c_decision": "decisions/stage_c.json",
        "accounting": _final_accounting(runtime, decision_b),
    }
    _atomic_json(output / "campaign_result.json", result)
    _state(runtime, f"COMPLETE_{outcome}", result="campaign_result.json")
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
    config = merge(load_yaml(args.campaign.resolve()), load_yaml(args.machine.resolve()))
    config = resolve_paths(config, root)
    if os.environ.get("MCC_ARTIFACT_ROOT"):
        config["run"]["artifact_root"] = str(
            Path(os.environ["MCC_ARTIFACT_ROOT"]).expanduser().resolve()
        )
    if os.environ.get("MCC_DESIGN_SNAPSHOTS_DIR"):
        config["data"]["snapshots"] = str(
            Path(os.environ["MCC_DESIGN_SNAPSHOTS_DIR"]).expanduser().resolve()
        )
    if os.environ.get("SMARTSIM_PORT"):
        config["smartsim"]["port"] = int(os.environ["SMARTSIM_PORT"])
    previous_term = signal.signal(signal.SIGTERM, _termination)
    previous_usr1 = signal.signal(signal.SIGUSR1, _termination)
    output_hint = (
        Path(config["run"]["artifact_root"]).expanduser().resolve()
        / _safe_name(config["run"]["name"])
    )
    try:
        output = run_campaign(root, config, dry_run=args.dry_run)
        print(output)
    except AllocationPause as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "PAUSED_ALLOCATION", "updated_utc": _utc(),
                "message": str(error),
            })
        print(f"Campaign paused cleanly: {error}", flush=True)
        raise SystemExit(0) from error
    except KeyboardInterrupt as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "INTERRUPTED", "updated_utc": _utc(),
                "message": str(error),
            })
        print(f"Campaign interrupted safely: {error}", flush=True)
        raise SystemExit(99) from error
    except BaseException as error:
        if output_hint.is_dir():
            _atomic_json(output_hint / "state.json", {
                "state": "FAILED", "updated_utc": _utc(),
                "type": type(error).__name__, "message": str(error),
            })
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGUSR1, previous_usr1)


if __name__ == "__main__":
    main()
