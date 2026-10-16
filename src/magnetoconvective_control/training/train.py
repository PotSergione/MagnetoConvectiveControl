"""Resumable RecurrentPPO training for the fusion-blanket control problem."""

from __future__ import annotations

import csv
import json
import os
import re
import signal
from collections import deque
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from ..envs.smartsim_env import SB3MARLSmartSimVecEnv, SmartSimVecEnv
from ..policies.cnn_lstm import CNNFeatureExtractor



# Cluster settings may be overridden without editing this file.
N_ENVS = int(os.environ.get("RL_N_ENVS", "1"))
SIMULATIONS_PER_GPU = int(os.environ.get("RL_SIMULATIONS_PER_GPU", "1"))
GPUS_PER_NODE = int(os.environ.get("RL_GPUS_PER_NODE", "1"))
BATCH_SIZE = int(os.environ.get("RL_BATCH_SIZE", "256"))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_DIRECTORY = PROJECT_ROOT / "cases" / "toroidal_gr1e9_ha1750"
SIM_EXECUTABLE = os.environ.get(
    "RL_SIM_EXECUTABLE", str(DEFAULT_CASE_DIRECTORY / "run_case.sh")
)
WALL_NPERP = int(os.environ.get("RL_WALL_NPERP", "128"))
WALL_NZ = int(os.environ.get("RL_WALL_NZ", "160"))
ACT_DIM = 10
LAUNCHER = os.environ.get("RL_LAUNCHER", "slurm")
GPU_GRES = os.environ.get("RL_GPU_GRES", os.environ.get("SMARTSIM_GPU_GRES", "gpu:1"))
CPUS_PER_SIMULATION = int(
    os.environ.get(
        "RL_CPUS_PER_SIMULATION",
        os.environ.get("SLURM_CPUS_PER_TASK", "1"),
    )
)
N_STEPS = int(os.environ.get("RL_N_STEPS", "256"))
N_EPOCHS = int(os.environ.get("RL_N_EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("RL_LEARNING_RATE", "3e-4"))
MAX_EPISODE_STEPS = int(os.environ.get("RL_MAX_EPISODE_STEPS", "1000"))
ACTION_BASELINE = float(os.environ.get("RL_ACTION_BASELINE", "0.0"))
ACTION_SCALE = float(os.environ.get("RL_ACTION_SCALE", "1.0"))
FIRST_WALL_HEAT_FLUX_BASELINE = float(
    os.environ.get("RL_FIRST_WALL_HEAT_FLUX_BASELINE", "0.0166913")
)
REWARD_SCALE = float(os.environ.get("RL_REWARD_SCALE", "1.0"))
EWMA_TIMESCALE = float(os.environ.get("RL_EWMA_TIMESCALE", "10.0"))

# This is the number of additional environment transitions requested by each
# launch. RecurrentPPO may finish the current rollout before returning.
TOTAL_TIMESTEPS = int(os.environ.get("RL_TOTAL_TIMESTEPS", "100000"))
CHECKPOINT_INTERVAL = int(os.environ.get("RL_CHECKPOINT_INTERVAL", "5000"))
METRICS_INTERVAL = int(os.environ.get("RL_METRICS_INTERVAL", "500"))
MODEL_NAME = os.environ.get("RL_MODEL_NAME", "ppo_smartsim")

ARTIFACT_ROOT = Path(
    os.environ.get("MCC_ARTIFACT_ROOT", PROJECT_ROOT / "runs")
).expanduser().resolve()
MODELS_DIRECTORY = ARTIFACT_ROOT
SNAPSHOTS_DIRECTORY = Path(
    os.environ.get("RL_SNAPSHOTS_DIR", PROJECT_ROOT / "data" / "snapshots")
).expanduser().resolve()

DATABASE_INTERFACE = os.environ.get("SMARTSIM_INTERFACE", "lo")
STARTUP_TIMEOUT = float(os.environ.get("RL_STARTUP_TIMEOUT", "1200"))
OCCA_CACHE_DIRECTORY = os.environ.get("RL_OCCA_CACHE_DIR")
NEKRS_CACHE_DIRECTORY = os.environ.get("RL_NEKRS_CACHE_DIR")
DATABASE_PORT = int(
    os.environ.get(
        "SMARTSIM_PORT",
        str(10000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 50000),
    )
)


def _atomic_save(model: RecurrentPPO, destination: Path) -> None:
    """Save an SB3 archive without exposing a partially written checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.zip")
    model.save(str(temporary))
    os.replace(temporary, destination)


class TrainingCheckpointCallback(BaseCallback):
    """Append reward statistics and maintain numbered/latest checkpoints."""

    METRIC_FIELDS = (
        "timestamp_utc",
        "run_id",
        "timesteps",
        "samples",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_q05",
        "reward_median",
        "reward_q95",
        "reward_max",
        "interval_physical_steps",
        "interval_physical_reward_total",
        "completed_episodes",
        "episode_reward_mean",
        "episode_reward_std",
    )
    EPISODE_FIELDS = (
        "timestamp_utc",
        "run_id",
        "env_id",
        "episode_id",
        "snapshot",
        "physical_steps",
        "episode_return",
        "end_reason",
    )

    def __init__(
        self,
        model_directory: Path,
        checkpoint_interval: int,
        metrics_interval: int,
        reward_window: int = 10000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if checkpoint_interval < 1 or metrics_interval < 1:
            raise ValueError("Checkpoint and metrics intervals must be positive")
        self.model_directory = model_directory
        self.checkpoint_directory = model_directory / "checkpoints"
        self.latest_path = model_directory / "latest.zip"
        self.metrics_path = model_directory / "reward_metrics.csv"
        self.episode_metrics_path = model_directory / "episode_metrics.csv"
        self.checkpoint_interval = checkpoint_interval
        self.metrics_interval = metrics_interval
        self.rewards: deque[float] = deque(maxlen=reward_window)
        self.episode_rewards: deque[float] = deque(maxlen=1000)
        self.interval_physical_steps = 0
        self.interval_physical_reward_total = 0.0
        self.running_episode_rewards: np.ndarray | None = None
        self.completed_episodes = 0
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.next_checkpoint = 0
        self.next_metrics = 0

    def _on_training_start(self) -> None:
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.running_episode_rewards = np.zeros(self.training_env.num_envs)
        current = int(self.model.num_timesteps)
        self.next_checkpoint = (
            current // self.checkpoint_interval + 1
        ) * self.checkpoint_interval
        self.next_metrics = (
            current // self.metrics_interval + 1
        ) * self.metrics_interval
        self._upgrade_metrics_schema()

    def _upgrade_metrics_schema(self) -> None:
        """Add newly introduced metric columns to an existing run CSV."""
        if not self.metrics_path.is_file() or self.metrics_path.stat().st_size == 0:
            return
        with self.metrics_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == list(self.METRIC_FIELDS):
                return
            rows = list(reader)
        temporary = self.metrics_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, self.metrics_path)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=float).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals.get("infos", [{} for _ in range(rewards.size)])
        if rewards.size % ACT_DIM != 0:
            raise RuntimeError("MARL rows are not divisible into ten-agent groups")
        grouped_dones = dones.reshape(-1, ACT_DIM)
        if not np.all(grouped_dones == grouped_dones[:, :1]):
            raise RuntimeError("Virtual agents from one CFD environment disagree on done")
        self.rewards.extend(rewards.tolist())
        self.interval_physical_steps += rewards.size // ACT_DIM
        self.interval_physical_reward_total += float(rewards.sum() / ACT_DIM)

        if self.running_episode_rewards is None:
            self.running_episode_rewards = np.zeros_like(rewards)
        self.running_episode_rewards += rewards
        # All ten virtual rows share one physical episode and reward. Record
        # each physical completion once, using its first symmetry-related row.
        for group_index in np.flatnonzero(grouped_dones[:, 0]):
            index = group_index * ACT_DIM
            episode_return = float(self.running_episode_rewards[index])
            self.episode_rewards.append(episode_return)
            info = infos[index] if index < len(infos) else {}
            self._append_episode_metric(group_index, episode_return, info)
            self.running_episode_rewards[index:index + ACT_DIM] = 0.0
            self.completed_episodes += 1

        timesteps = int(self.model.num_timesteps)
        if timesteps >= self.next_metrics:
            self._write_metrics(timesteps)
            while self.next_metrics <= timesteps:
                self.next_metrics += self.metrics_interval

        if timesteps >= self.next_checkpoint:
            self._save_checkpoint(timesteps)
            while self.next_checkpoint <= timesteps:
                self.next_checkpoint += self.checkpoint_interval
        return True

    def _append_episode_metric(
        self, env_id: int, episode_return: float, info: dict
    ) -> None:
        """Persist one compact row per physical episode."""
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "env_id": info.get("env_id", env_id),
            "episode_id": info.get("episode_id"),
            "snapshot": info.get("snapshot"),
            "physical_steps": info.get("episode_step"),
            "episode_return": episode_return,
            "end_reason": info.get("episode_end_reason", "unknown"),
        }
        new_file = (
            not self.episode_metrics_path.exists()
            or self.episode_metrics_path.stat().st_size == 0
        )
        with self.episode_metrics_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.EPISODE_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _write_metrics(self, timesteps: int) -> None:
        reward_values = np.asarray(self.rewards, dtype=float)
        episode_values = np.asarray(self.episode_rewards, dtype=float)
        if reward_values.size == 0 or self.interval_physical_steps == 0:
            return

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "timesteps": timesteps,
            "samples": reward_values.size,
            "reward_mean": reward_values.mean(),
            "reward_std": reward_values.std(),
            "reward_min": reward_values.min(),
            "reward_q05": np.quantile(reward_values, 0.05),
            "reward_median": np.median(reward_values),
            "reward_q95": np.quantile(reward_values, 0.95),
            "reward_max": reward_values.max(),
            "interval_physical_steps": self.interval_physical_steps,
            "interval_physical_reward_total": self.interval_physical_reward_total,
            "completed_episodes": self.completed_episodes,
            "episode_reward_mean": (
                episode_values.mean() if episode_values.size else np.nan
            ),
            "episode_reward_std": (
                episode_values.std() if episode_values.size else np.nan
            ),
        }
        new_file = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        with self.metrics_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.METRIC_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        for key in (
            "reward_mean", "reward_std", "reward_min", "reward_q05",
            "reward_median", "reward_q95", "reward_max",
            "interval_physical_steps", "interval_physical_reward_total",
            "episode_reward_mean", "episode_reward_std",
        ):
            self.logger.record(f"reward_distribution/{key}", row[key])
        self.logger.record("reward_distribution/completed_episodes", self.completed_episodes)
        self.interval_physical_steps = 0
        self.interval_physical_reward_total = 0.0

    def _save_checkpoint(self, timesteps: int) -> None:
        numbered_path = self.checkpoint_directory / f"step_{timesteps:012d}.zip"
        _atomic_save(self.model, numbered_path)
        _atomic_save(self.model, self.latest_path)
        if self.verbose:
            print(f"Saved training checkpoint at {timesteps} steps: {numbered_path}")

    def save_final(self, model: RecurrentPPO | None = None) -> None:
        """Persist both the final numbered archive and the resume pointer."""
        active_model = model if model is not None else self.model
        timesteps = int(active_model.num_timesteps)
        if hasattr(self, "model"):
            self._write_metrics(timesteps)
            self._save_checkpoint(timesteps)
        else:
            numbered_path = (
                self.checkpoint_directory / f"step_{timesteps:012d}.zip"
            )
            _atomic_save(active_model, numbered_path)
            _atomic_save(active_model, self.latest_path)


def _raise_on_termination(signum, _frame) -> None:
    """Turn Slurm's SIGTERM into an exception so finally saves the model."""
    raise KeyboardInterrupt(f"Received signal {signum}")


def main() -> None:
    if N_ENVS < 1:
        raise ValueError("RL_N_ENVS must be positive")
    if SIMULATIONS_PER_GPU < 1:
        raise ValueError("RL_SIMULATIONS_PER_GPU must be positive")
    if GPUS_PER_NODE < 1:
        raise ValueError("RL_GPUS_PER_NODE must be positive")
    if STARTUP_TIMEOUT <= 0:
        raise ValueError("RL_STARTUP_TIMEOUT must be positive")
    allocated_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    required_gpus = ceil(N_ENVS / SIMULATIONS_PER_GPU)
    if (
        LAUNCHER.lower() == "slurm"
        and required_gpus > allocated_nodes * GPUS_PER_NODE
    ):
        raise ValueError(
            f"The active Slurm allocation provides at most "
            f"{allocated_nodes * GPUS_PER_NODE} GPUs, but {required_gpus} are required"
        )
    if (
        LAUNCHER.lower() == "slurm"
        and allocated_nodes > 1
        and DATABASE_INTERFACE == "lo"
    ):
        raise ValueError(
            "Multi-node SmartSim cannot use loopback for its database; set "
            "SMARTSIM_INTERFACE to a routable compute-node interface"
        )
    snapshot_files = sorted(
        path for path in SNAPSHOTS_DIRECTORY.iterdir() if path.is_file()
    )
    if not snapshot_files:
        raise FileNotFoundError(f"No restart fields found in {SNAPSHOTS_DIRECTORY}")
    invalid_snapshots = [
        path.name for path in snapshot_files
        if re.fullmatch(r"blanket0\.f\d+", path.name) is None
    ]
    if invalid_snapshots:
        raise ValueError(
            "All restart fields must use blanket0.fNNNNN names; found "
            + ", ".join(invalid_snapshots)
        )
    rollout_samples = N_STEPS * N_ENVS * ACT_DIM
    if N_STEPS < 2:
        raise ValueError("RL_N_STEPS must be at least 2")
    if BATCH_SIZE < 1 or rollout_samples % BATCH_SIZE != 0:
        raise ValueError(
            f"RL_BATCH_SIZE={BATCH_SIZE} must divide the {rollout_samples} "
            "virtual-agent samples in each rollout"
        )
    if WALL_NZ % ACT_DIM != 0:
        raise ValueError("RL_WALL_NZ must be divisible by ten")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", MODEL_NAME) is None:
        raise ValueError(
            "RL_MODEL_NAME must contain only letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )

    model_directory = MODELS_DIRECTORY / MODEL_NAME
    latest_path = model_directory / "latest.zip"
    model_directory.mkdir(parents=True, exist_ok=True)
    mapping_path = model_directory / "action_mapping.json"
    action_mapping = {
        "baseline": ACTION_BASELINE,
        "scale": ACTION_SCALE,
        "policy_low": -0.25,
        "policy_high": 0.25,
        "physical_low": -0.25,
        "physical_high": 0.25,
    }
    run_contract = {
        "n_physical_environments": N_ENVS,
        "virtual_agents": ACT_DIM,
        "wall_shape": [3, WALL_NPERP, WALL_NZ],
        "reward_objective": "minimize_plasma_facing_x_minus_heat_flux",
        "first_wall_heat_flux_baseline": FIRST_WALL_HEAT_FLUX_BASELINE,
        "reward_scale": REWARD_SCALE,
        "ewma_timescale": EWMA_TIMESCALE,
        "gamma": 0.997,
        "gae_lambda": 0.98,
        "episode_steps": MAX_EPISODE_STEPS,
        "ppo_rollout_steps": N_STEPS,
        "ppo_batch_size": BATCH_SIZE,
        "ppo_epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "control_interval": 0.1,
    }
    contract_path = model_directory / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != run_contract:
        raise ValueError(f"Model {MODEL_NAME!r} has an incompatible run contract")
    contract_path.write_text(json.dumps(run_contract, indent=2) + "\n")
    if mapping_path.is_file():
        saved_mapping = json.loads(mapping_path.read_text())
        if saved_mapping != action_mapping:
            raise ValueError(
                f"Model {MODEL_NAME!r} was created with action mapping "
                f"{saved_mapping}, not {action_mapping}; use a new model name"
            )
    elif latest_path.is_file() and (ACTION_BASELINE, ACTION_SCALE) != (0.0, 1.0):
        raise ValueError(
            f"Legacy model {MODEL_NAME!r} has direct-action semantics; use a "
            "new model name for residual control"
        )
    else:
        mapping_path.write_text(json.dumps(action_mapping, indent=2) + "\n")
    print(
        f"Action mapping: physical=clip({ACTION_BASELINE:g} + "
        f"{ACTION_SCALE:g} * policy, -0.25, 0.25)"
    )
    print(
        f"Parallel rollout: {N_ENVS} physical environments, "
        f"{N_ENVS * N_STEPS} physical transitions/update, "
        f"{rollout_samples} virtual-agent samples/update"
    )

    observation_space = gym.spaces.Dict(
        {
            "heat_flux": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, WALL_NPERP, WALL_NZ),
                dtype=np.float32,
            ),
            "previous_actions": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(ACT_DIM,),
                dtype=np.float32,
            ),
        }
    )
    action_space = gym.spaces.Box(
        low=-0.25,
        high=0.25,
        shape=(1,),
        dtype=np.float32,
    )

    core = None
    env = None
    model = None
    callback = None
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_termination)
    try:
        core = SmartSimVecEnv(
            n_envs=N_ENVS,
            simulations_per_gpu=SIMULATIONS_PER_GPU,
            sim_executable=SIM_EXECUTABLE,
            wall_shape=(WALL_NPERP, WALL_NZ),
            act_dim=ACT_DIM,
            launcher=LAUNCHER,
            gpu_gres=GPU_GRES,
            cpus_per_simulation=CPUS_PER_SIMULATION,
            max_episode_steps=MAX_EPISODE_STEPS,
            snapshots_dir=str(SNAPSHOTS_DIRECTORY),
            startup_timeout=STARTUP_TIMEOUT,
            database_port=DATABASE_PORT,
            database_interface=DATABASE_INTERFACE,
            occa_cache_dir=OCCA_CACHE_DIRECTORY,
            nekrs_cache_dir=NEKRS_CACHE_DIRECTORY,
            action_baseline=ACTION_BASELINE,
            action_scale=ACTION_SCALE,
            first_wall_heat_flux_baseline=FIRST_WALL_HEAT_FLUX_BASELINE,
            reward_scale=REWARD_SCALE,
            ewma_timescale=EWMA_TIMESCALE,
            action_magnitude_regularization=float(os.environ.get("RL_ACTION_MAG_REG", "0.0")),
            action_slew_regularization=float(os.environ.get("RL_ACTION_SLEW_REG", "0.0")),
        )
        env = SB3MARLSmartSimVecEnv(core, observation_space, action_space)

        if latest_path.is_file():
            print(f"Resuming model '{MODEL_NAME}' from {latest_path}")
            model = RecurrentPPO.load(str(latest_path), env=env, device="cpu")
        else:
            print(f"Starting new model '{MODEL_NAME}' in {model_directory}")
            policy_kwargs = {
                "features_extractor_class": CNNFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
                "normalize_images": False,
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm":True,
                "log_std_init":-2.3
            }
            model = RecurrentPPO(
                "MultiInputLstmPolicy",
                env,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                learning_rate=LEARNING_RATE,
                gamma=0.997,
                gae_lambda=0.98,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard/",
                device="cpu",
                verbose=1,
            )

        callback = TrainingCheckpointCallback(
            model_directory=model_directory,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            metrics_interval=METRICS_INTERVAL,
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False,
        )
    finally:
        if model is not None:
            if callback is not None:
                callback.save_final(model)
            else:
                _atomic_save(model, latest_path)
        if env is not None:
            env.close()
        elif core is not None:
            core.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
