"""A synchronous Stable-Baselines3 environment backed by NekRS and SmartSim.

Each environment is one NekRS blanket process. The UDF publishes three
wall-normal heat-flux fields and scalar thermal diagnostics, and synchronously
reads a sequenced ten-electrode action through SmartRedis. The environment
reconstructs fixed regular wall maps and rewards an EWMA reduction of the
integrated heat flux through the plasma-facing ``x-`` wall.
"""

from __future__ import annotations

import random
import re
import shutil
import time
import os
import sys
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from smartredis.client import Client
from smartsim import Experiment

from ..launchers.bede_mpirun import (
    discover_allocation_hosts,
    parse_allocation_hosts,
)


ACTION_KEY = "nekrs:timeIntensity"
RANK_ZERO_PREFIX = "nekrs:rank:0:"
DEFAULT_OBSERVATION_FIELDS = (
    "heat_flux",
    "Nu",
    "Tbar",
    "Kbar",
    "xMinusIntegratedHeatFlux",
    "xPlusIntegratedHeatFlux",
    "yMinusIntegratedHeatFlux",
    "yPlusIntegratedHeatFlux",
    "totalHeatingPower",
)
WALL_CHANNELS = ("yMinus", "yPlus", "xPlus")
ACTION_LOW = -0.25
ACTION_HIGH = 0.25
ACTION_SEQUENCE_KEY = "nekrs:actionSequence"
SHUTDOWN_KEY = "nekrs:shutdown"
MODEL_TENSOR_KEYS = (
    ACTION_KEY,
    ACTION_SEQUENCE_KEY,
    SHUTDOWN_KEY,
    *(
        RANK_ZERO_PREFIX + wall + suffix
        for wall in WALL_CHANNELS
        for suffix in ("HeatFlux", "Transverse", "Z")
    ),
    RANK_ZERO_PREFIX + "timeIntensity",
    RANK_ZERO_PREFIX + "actionSequence",
    RANK_ZERO_PREFIX + "Nu",
    RANK_ZERO_PREFIX + "Tbar",
    RANK_ZERO_PREFIX + "Kbar",
    RANK_ZERO_PREFIX + "xMinusIntegratedHeatFlux",
    RANK_ZERO_PREFIX + "xPlusIntegratedHeatFlux",
    RANK_ZERO_PREFIX + "yMinusIntegratedHeatFlux",
    RANK_ZERO_PREFIX + "yPlusIntegratedHeatFlux",
    RANK_ZERO_PREFIX + "totalHeatingPower",
    RANK_ZERO_PREFIX + "step",
    RANK_ZERO_PREFIX + "time",
)


@dataclass(frozen=True)
class _WallReconstructionPlan:
    raw_order: np.ndarray
    group_starts: np.ndarray
    group_counts: np.ndarray
    group_to_native_order: np.ndarray
    interpolation_indices: np.ndarray
    interpolation_weights: np.ndarray
    output_shape: tuple[int, int]
    transverse_coordinates: np.ndarray
    z_coordinates: np.ndarray

    def reconstruct(self, raw_field: np.ndarray) -> np.ndarray:
        raw_field = np.asarray(raw_field).reshape(-1)
        if raw_field.size != self.raw_order.size:
            raise ValueError(
                f"Expected {self.raw_order.size} raw wall samples, got "
                f"{raw_field.size}"
            )
        ordered = raw_field[self.raw_order]
        grouped = np.add.reduceat(ordered, self.group_starts) / self.group_counts
        native_flat = grouped[self.group_to_native_order]
        output = np.sum(
            native_flat[self.interpolation_indices] * self.interpolation_weights,
            axis=1,
        )
        return output.reshape(self.output_shape)


def _cluster_axis(values: np.ndarray, relative_tolerance: float = 1e-9):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    sorted_values = values[order]
    tolerance = relative_tolerance * max(np.ptp(sorted_values), 1.0)
    sorted_groups = np.r_[
        0, np.cumsum(np.diff(sorted_values) > tolerance)
    ].astype(np.int64)
    counts = np.bincount(sorted_groups)
    centers = np.bincount(sorted_groups, weights=sorted_values) / counts
    labels = np.empty_like(sorted_groups)
    labels[order] = sorted_groups
    return centers, labels


def _build_wall_plan(
    transverse: np.ndarray, z: np.ndarray, output_shape: tuple[int, int]
) -> _WallReconstructionPlan:
    x = np.asarray(transverse, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if x.shape != z.shape or x.size == 0:
        raise ValueError("Wall transverse and Z tensors must be nonempty and equally sized")

    x_nodes, x_index = _cluster_axis(x)
    z_nodes, z_index = _cluster_axis(z)
    raw_labels = z_index * x_nodes.size + x_index
    raw_order = np.argsort(raw_labels, kind="stable")
    sorted_labels = raw_labels[raw_order]
    group_starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_labels))]
    group_labels = sorted_labels[group_starts]
    group_counts = np.diff(np.r_[group_starts, raw_labels.size])
    native_size = z_nodes.size * x_nodes.size
    if group_labels.size != native_size or not np.array_equal(
        np.sort(group_labels), np.arange(native_size)
    ):
        raise ValueError("The native wall grid is incomplete")
    group_to_native_order = np.argsort(group_labels)

    nperp_output, nz_output = output_shape
    x_target = np.linspace(x_nodes[0], x_nodes[-1], nperp_output)
    z_target = np.linspace(z_nodes[0], z_nodes[-1], nz_output, endpoint=False)
    target_x, target_z = np.meshgrid(x_target, z_target, indexing="ij")
    ix1 = np.clip(np.searchsorted(x_nodes, target_x, side="right"), 1, x_nodes.size - 1)
    iz1 = np.clip(np.searchsorted(z_nodes, target_z, side="right"), 1, z_nodes.size - 1)
    ix0, iz0 = ix1 - 1, iz1 - 1
    fx = (target_x - x_nodes[ix0]) / (x_nodes[ix1] - x_nodes[ix0])
    fz = (target_z - z_nodes[iz0]) / (z_nodes[iz1] - z_nodes[iz0])
    interpolation_indices = np.column_stack(
        (
            (iz0 * x_nodes.size + ix0).ravel(),
            (iz0 * x_nodes.size + ix1).ravel(),
            (iz1 * x_nodes.size + ix0).ravel(),
            (iz1 * x_nodes.size + ix1).ravel(),
        )
    )
    interpolation_weights = np.column_stack(
        (
            ((1 - fz) * (1 - fx)).ravel(),
            ((1 - fz) * fx).ravel(),
            (fz * (1 - fx)).ravel(),
            (fz * fx).ravel(),
        )
    )
    return _WallReconstructionPlan(
        raw_order=raw_order,
        group_starts=group_starts,
        group_counts=group_counts,
        group_to_native_order=group_to_native_order,
        interpolation_indices=interpolation_indices,
        interpolation_weights=interpolation_weights,
        output_shape=output_shape,
        transverse_coordinates=x_target,
        z_coordinates=z_target,
    )


class _EpisodeCompleted(RuntimeError):
    """Raised when a NekRS model completes normally between control frames."""


class _SimulationFailed(RuntimeError):
    """Raised when a managed NekRS process fails, is cancelled, or times out."""

    def __init__(
        self,
        env_id: int,
        status: str,
        model_name: str,
        model_directory: Path,
        snapshot: Path | None,
    ) -> None:
        self.env_id = env_id
        self.status = status
        self.model_name = model_name
        self.model_directory = model_directory
        self.snapshot = snapshot
        super().__init__(
            f"NekRS environment {env_id} failed while waiting for a new frame "
            f"(status {status}, model {model_name})"
        )


class SmartSimVecEnv:
    """Run ``n_envs`` NekRS cases in lockstep.

    ``sim_executable`` is normally ``run_case.sh``.  Files in ``case_files``
    are copied into each SmartSim model directory so concurrent NekRS runs do
    not share generated files or logs.

    ``simulations_per_gpu`` controls GPU packing. Environment ``env_id`` is
    assigned to allocation-local GPU ``env_id // simulations_per_gpu``. Slurm
    model steps use ``--overlap`` when packing is enabled so that several
    independent steps may share the selected GPU.

    Every episode samples a restart checkpoint from ``snapshots_dir``. By
    default this is the ``snapshots`` directory beside this module. The chosen
    file is copied into the isolated model directory and its generated NekRS
    parameter file is updated to use it as ``startFrom``.

    The UDF exposes ten electrode controls. Observations are reconstructed as
    three rectangular heat-flux maps ordered ``(yMinus, yPlus, xPlus)``.
    """

    def __init__(
        self,
        n_envs: int,
        sim_executable: str,
        wall_shape: tuple[int, int],
        act_dim: int = 10,
        launcher: str = "slurm",
        observation_fields: Sequence[str] = DEFAULT_OBSERVATION_FIELDS,
        case_files: Iterable[str] | None = None,
        poll_interval: float = 0.05,
        startup_timeout: float | None = None,
        database_port: int = 6780,
        database_interface: str = "lo",
        initial_action: float | Sequence[float] = 0.0,
        action_baseline: float | Sequence[float] = 0.0,
        action_scale: float | Sequence[float] = 1.0,
        max_episode_steps: int | None = None,
        simulations_per_gpu: int = 1,
        snapshots_dir: str | None = None,
        first_wall_heat_flux_baseline: float = 0.0166913,
        reward_scale: float = 1.0,
        ewma_timescale: float = 10.0,
        action_magnitude_regularization: float = 0.0,
        action_slew_regularization: float = 0.0,
        gpu_gres: str | None = None,
        cpus_per_simulation: int | None = None,
        simulation_failure_penalty: float = 0.0,
        max_failure_restarts: int = 3,
        snapshot_path: str | None = None,
        snapshot_schedules: Sequence[Sequence[str]] | None = None,
        observation_scaling: str = "per_frame",
        fixed_wall_mean: float | None = None,
        fixed_wall_std: float | None = None,
        episode_horizon_truncated: bool = True,
        occa_cache_dir: str | None = None,
        nekrs_cache_dir: str | None = None,
        bede_mpi_command: str = "bede-mpirun",
        allocation_hosts: str | Sequence[str] | None = None,
    ):
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        if not isinstance(simulations_per_gpu, int) or isinstance(
            simulations_per_gpu, bool
        ):
            raise TypeError("simulations_per_gpu must be an integer")
        if simulations_per_gpu < 1:
            raise ValueError("simulations_per_gpu must be positive")
        if act_dim != 10:
            raise ValueError(
                "blanket.udf accepts ten electrode actions through "
                "nekrs:timeIntensity, so act_dim must be 10"
            )
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not np.isfinite(simulation_failure_penalty):
            raise ValueError("simulation_failure_penalty must be finite")
        if (
            not isinstance(max_failure_restarts, int)
            or isinstance(max_failure_restarts, bool)
            or max_failure_restarts < 1
        ):
            raise ValueError("max_failure_restarts must be a positive integer")

        executable = Path(sim_executable).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Simulation executable not found: {executable}")

        self.n_envs = n_envs
        if len(wall_shape) != 2 or any(int(size) < 2 for size in wall_shape):
            raise ValueError("wall_shape must contain two dimensions >= 2")
        self.wall_shape = tuple(int(size) for size in wall_shape)
        self.observation_shape = (len(WALL_CHANNELS), *self.wall_shape)
        self.obs_dim = int(np.prod(self.observation_shape))
        self.act_dim = act_dim
        self.launcher = launcher.lower()
        self.simulations_per_gpu = simulations_per_gpu
        self.n_gpus = (n_envs + simulations_per_gpu - 1) // simulations_per_gpu
        self.gpu_gres = (
            gpu_gres
            if gpu_gres is not None
            else os.environ.get("SMARTSIM_GPU_GRES", "gpu:1")
        )
        self.cpus_per_simulation = (
            cpus_per_simulation
            if cpus_per_simulation is not None
            else int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        )
        if self.cpus_per_simulation < 1:
            raise ValueError("cpus_per_simulation must be positive")
        self.bede_mpi_command = bede_mpi_command
        self.allocation_hosts: tuple[str, ...] = ()
        if self.launcher == "bede_mpirun":
            if simulations_per_gpu != 1:
                raise ValueError(
                    "bede_mpirun currently requires simulations_per_gpu=1"
                )
            host_value = allocation_hosts or os.environ.get("MCC_BEDE_HOSTS")
            self.allocation_hosts = (
                parse_allocation_hosts(host_value)
                if host_value
                else discover_allocation_hosts(self.bede_mpi_command)
            )
            if self.n_gpus > len(self.allocation_hosts):
                raise ValueError(
                    f"The run needs {self.n_gpus} Bede GPU hosts, but only "
                    f"{len(self.allocation_hosts)} were discovered"
                )
        # Retained for launcher compatibility; explicit tensor names are used.
        self.observation_fields = tuple(observation_fields)
        self.poll_interval = poll_interval
        self.startup_timeout = startup_timeout
        self.action_baseline = self._action_parameter(
            action_baseline, act_dim, "action_baseline"
        )
        self.action_scale = self._action_parameter(
            action_scale, act_dim, "action_scale"
        )
        self.initial_policy_action = self._action_parameter(
            initial_action, act_dim, "initial_action"
        )
        self.initial_action = self.policy_to_physical_action(
            self.initial_policy_action
        )
        self.max_episode_steps = max_episode_steps
        self.episode_horizon_truncated = bool(episode_horizon_truncated)
        if observation_scaling not in {"per_frame", "fixed"}:
            raise ValueError("observation_scaling must be 'per_frame' or 'fixed'")
        if observation_scaling == "fixed":
            if (
                fixed_wall_mean is None
                or fixed_wall_std is None
                or not np.isfinite(fixed_wall_mean)
                or not np.isfinite(fixed_wall_std)
                or fixed_wall_std <= 0
            ):
                raise ValueError("fixed observation scaling requires a finite mean and positive std")
        self.observation_scaling = observation_scaling
        self.fixed_wall_mean = fixed_wall_mean
        self.fixed_wall_std = fixed_wall_std
        self.simulation_failure_penalty = float(simulation_failure_penalty)
        self.max_failure_restarts = max_failure_restarts
        self.sim_executable = str(executable)
        case_directory = executable.parent
        self.occa_cache_dir = (
            Path(occa_cache_dir).expanduser().resolve()
            if occa_cache_dir is not None
            else case_directory / ".cache" / "occa"
        )
        if nekrs_cache_dir is not None:
            self.nekrs_cache_dir: Path | None = (
                Path(nekrs_cache_dir).expanduser().resolve()
            )
        elif self.launcher == "local":
            self.nekrs_cache_dir = case_directory / ".cache" / "nekrs"
        else:
            # A shared Slurm UDF cache must be prewarmed before concurrent
            # workers use it, so do not opt into one implicitly.
            self.nekrs_cache_dir = None
        self.occa_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.nekrs_cache_dir is not None:
            self.nekrs_cache_dir.mkdir(parents=True, exist_ok=True)

        if snapshot_schedules is not None and (
            snapshot_path is not None or snapshots_dir is not None
        ):
            raise ValueError(
                "Specify snapshot_schedules, snapshot_path, or snapshots_dir, not more than one"
            )
        self.snapshot_path = (
            Path(snapshot_path).expanduser().resolve()
            if snapshot_path is not None
            else None
        )
        if self.snapshot_path is not None:
            if not self.snapshot_path.is_file():
                raise FileNotFoundError(
                    f"Snapshot file not found: {self.snapshot_path}"
                )
            if re.fullmatch(r"blanket0\.f\d+", self.snapshot_path.name) is None:
                raise ValueError(
                    "Snapshot filename must look like blanket0.fNNNNN: "
                    f"{self.snapshot_path.name}"
                )
            self.snapshots_dir = None
        elif snapshot_schedules is None:
            snapshot_directory = (
                Path(__file__).resolve().parent / "snapshots"
                if snapshots_dir is None
                else Path(snapshots_dir).expanduser().resolve()
            )
            if not snapshot_directory.is_dir():
                raise FileNotFoundError(
                    f"Snapshot directory not found: {snapshot_directory}"
                )
            self.snapshots_dir = snapshot_directory
        else:
            if len(snapshot_schedules) != n_envs:
                raise ValueError(
                    "snapshot_schedules must contain one schedule per environment"
                )
            resolved_schedules = []
            for env_id, schedule in enumerate(snapshot_schedules):
                paths = tuple(Path(item).expanduser().resolve() for item in schedule)
                if not paths:
                    raise ValueError(f"Snapshot schedule {env_id} is empty")
                for path in paths:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if re.fullmatch(r"blanket0\.f\d+", path.name) is None:
                        raise ValueError(
                            f"Noncanonical snapshot in schedule {env_id}: {path.name}"
                        )
                resolved_schedules.append(paths)
            self.snapshots_dir = None
            self._snapshot_schedules = tuple(resolved_schedules)
        if snapshot_schedules is None:
            self._snapshot_schedules = None
        self._snapshot_schedule_indices = [0] * n_envs
        self._snapshot_rng = random.SystemRandom()
        self._snapshot_queue: list[Path] = []
        if (
            not np.isfinite(first_wall_heat_flux_baseline)
            or first_wall_heat_flux_baseline <= 0
        ):
            raise ValueError(
                "first_wall_heat_flux_baseline must be finite and positive"
            )
        if not np.isfinite(ewma_timescale) or ewma_timescale <= 0:
            raise ValueError("ewma_timescale must be finite and positive")
        if not np.isfinite(reward_scale):
            raise ValueError("reward_scale must be finite")
        if not np.isfinite(action_magnitude_regularization):
            raise ValueError("action_magnitude_regularization must be finite")
        if not np.isfinite(action_slew_regularization):
            raise ValueError("action_slew_regularization must be finite")
        self.first_wall_heat_flux_baseline = float(
            first_wall_heat_flux_baseline
        )
        self.reward_scale = float(reward_scale)
        self.ewma_alpha = 1.0 - exp(-1.0 / float(ewma_timescale))
        self.action_magnitude_regularization = float(action_magnitude_regularization)
        self.action_slew_regularization = float(action_slew_regularization)

        # Queried by SB3's VecEnv base class.
        self.render_mode = None

        self._episode_id = [0] * n_envs
        self._episode_steps = [0] * n_envs
        self._models = [None] * n_envs
        self._episode_snapshots: list[Path | None] = [None] * n_envs
        self._last_steps: list[int | None] = [None] * n_envs
        self._last_observations: list[np.ndarray | None] = [None] * n_envs
        self._last_rewards: list[float | None] = [None] * n_envs
        self._first_wall_flux_hat = np.full(
            n_envs, self.first_wall_heat_flux_baseline, dtype=np.float64
        )
        self._last_nu = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_first_wall_flux = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_total_heating_power = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_wall_fluxes = np.full((n_envs, 4), np.nan, dtype=np.float64)
        self._last_times = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_solver_steps = np.full(n_envs, -1, dtype=np.int64)
        self._last_raw_wall_maps: list[np.ndarray | None] = [None] * n_envs
        self._last_tbar = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_kbar = np.full(n_envs, np.nan, dtype=np.float64)
        self._last_reward_components: list[dict[str, float]] = [
            {} for _ in range(n_envs)
        ]
        self._action_sequences = np.zeros(n_envs, dtype=np.int64)
        self._wall_plans: list[tuple[_WallReconstructionPlan, ...] | None] = [
            None
        ] * n_envs
        self._last_actions: list[np.ndarray] = [
            self.initial_action.copy() for _ in range(n_envs)
        ]
        self._last_policy_actions: list[np.ndarray] = [
            self.initial_policy_action.copy() for _ in range(n_envs)
        ]
        self._pending_policy_actions: np.ndarray | None = None
        self._pending_actions: np.ndarray | None = None
        self._actions_pending = False
        self._closed = False

        experiment_launcher = (
            "local" if self.launcher == "bede_mpirun" else self.launcher
        )
        self.exp = Experiment(name="rl_training", launcher=experiment_launcher)
        self.db = self.exp.create_database(
            port=database_port,
            interface=database_interface,
        )

        if self.launcher == "slurm":
            # The Orchestrator only needs one CPU for this smoke test.
            self.db.set_cpus(1)

            # Crucial: prevent the Orchestrator srun step from acquiring
            # the GPUs allocated to the parent Slurm job.
            self.db.set_run_arg("gres", "none")

        self.exp.start(self.db)
        self.client = Client(address=self.db.get_address()[0], cluster=False)

        if case_files is None:
            case_dir = executable.parent
            candidates = (
                executable,
                *case_dir.glob("*.par"),
                *case_dir.glob("*.re2"),
                *case_dir.glob("*.udf"),
                *case_dir.glob("*.usr"),
                *case_dir.glob("*.nek5000"),
                *case_dir.glob("udf.cmake"),
            )
            self.case_files = tuple(
                str(path.resolve()) for path in candidates if path.is_file()
            )
        else:
            self.case_files = tuple(
                str(Path(path).expanduser().resolve()) for path in case_files
            )

        missing = [path for path in self.case_files if not Path(path).is_file()]
        if missing:
            self.close()
            raise FileNotFoundError(f"Case files not found: {missing}")

        try:
            for env_id in range(n_envs):
                self._launch_sim(env_id)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _action_parameter(
        value: float | Sequence[float], act_dim: int, name: str
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(act_dim, array.item(), dtype=np.float64)
        if array.shape != (act_dim,):
            raise ValueError(
                f"{name} must be a scalar or have shape {(act_dim,)}, "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return np.ascontiguousarray(array)

    def policy_to_physical_action(self, actions: np.ndarray) -> np.ndarray:
        """Map policy residuals to clipped controls sent to NekRS."""
        return np.ascontiguousarray(
            np.clip(
                self.action_baseline + self.action_scale * actions,
                ACTION_LOW,
                ACTION_HIGH,
            ),
            dtype=np.float64,
        )

    def _model_name(self, env_id: int) -> str:
        return f"nekrs_env_{env_id}_ep_{self._episode_id[env_id]}"

    def _gpu_id(self, env_id: int) -> int:
        """Return the allocation-local GPU assigned to an environment."""
        return env_id // self.simulations_per_gpu

    @staticmethod
    def _database_key(model_name: str, key: str) -> str:
        # SmartRedis's ensemble-prefix separator is a full stop.
        return f"{model_name}.{key}"

    def _key(self, env_id: int, key: str) -> str:
        model = self._models[env_id]
        if model is None:
            raise RuntimeError(f"Environment {env_id} has no active model")
        return self._database_key(model.name, key)

    def _launch_sim(self, env_id: int) -> None:
        """Generate and start one isolated case on its assigned GPU."""
        model_name = self._model_name(env_id)
        gpu_id = self._gpu_id(env_id)
        snapshot = self._select_snapshot(env_id)
        env_vars = {
            # SSKEYIN makes the UDF read this model's prefixed action.
            # SmartSim supplies SSDB and, via enable_key_prefixing, SSKEYOUT.
            "SSKEYIN": model_name,
            # Kernel binaries are immutable and safe to share.  Reuse the
            # case's populated cache instead of recompiling every NekRS kernel
            # in every generated model directory.
            "OCCA_CACHE_DIR": str(self.occa_cache_dir),
        }
        if self.nekrs_cache_dir is not None:
            env_vars["NEKRS_CACHE_DIR"] = str(self.nekrs_cache_dir)
        run_args = {}
        run_command = "auto"
        executable = self.sim_executable
        executable_args = None
        if self.launcher == "slurm":
            run_command = "srun"
            run_args = {
                "nodes": 1,
                "ntasks": 1,
                "cpus-per-task": self.cpus_per_simulation,
                "gres": self.gpu_gres,
                "mpi": "pmix",
                # "output": "/dev/null",
                # "error": "/dev/null",
            }
            if self.simulations_per_gpu > 1:
                # Parallel Slurm steps are exclusive by default, including
                # their GRES. Allow the SmartSim model steps to share the job
                # allocation and bind each one to a deterministic GPU. GPU IDs
                # are zero-based relative to the GPUs in the parent job.
                run_args["overlap"] = None
                run_args["gpu-bind"] = f"map_gpu:{gpu_id}"
        elif self.launcher == "local":
            # Useful for a workstation where GPU IDs are locally visible.
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        elif self.launcher == "bede_mpirun":
            # SmartSim supervises this adapter locally; the adapter execs
            # bede-mpirun to place the generated case on its assigned node.
            executable = sys.executable
            executable_args = [
                "-m",
                "magnetoconvective_control.launchers.bede_mpirun",
                Path(self.sim_executable).name,
            ]
            env_vars["MCC_BEDE_TARGET_HOST"] = self.allocation_hosts[gpu_id]
            env_vars["MCC_BEDE_MPIRUN"] = self.bede_mpi_command
            # bede-mpirun already owns the worker's MPI rank.  Tell the case
            # wrapper to exec NekRS directly instead of nesting another
            # mpirun through nrsmpi.
            env_vars["MCC_BEDE_MPI_PARENT"] = "1"
        else:
            raise ValueError(f"Unsupported launcher: {self.launcher}")

        run_settings = self.exp.create_run_settings(
            exe=executable,
            exe_args=executable_args,
            run_command=run_command,
            run_args=run_args,
            env_vars=env_vars,
        )
        model = self.exp.create_model(
            name=model_name,
            run_settings=run_settings,
            enable_key_prefixing=True,
        )
        model.attach_generator_files(to_copy=[*self.case_files, str(snapshot)])
        self.exp.generate(model, overwrite=True)
        self._set_model_restart_file(model, snapshot.name)

        self._models[env_id] = model
        self._episode_snapshots[env_id] = snapshot
        self._episode_steps[env_id] = 0
        self._last_steps[env_id] = None
        self._last_observations[env_id] = None
        self._last_rewards[env_id] = None
        self._wall_plans[env_id] = None
        self._last_actions[env_id] = self.initial_action.copy()
        self._last_policy_actions[env_id] = self.initial_policy_action.copy()
        self._action_sequences[env_id] = 0
        self._first_wall_flux_hat[env_id] = self.first_wall_heat_flux_baseline
        self._last_nu[env_id] = np.nan
        self._last_first_wall_flux[env_id] = np.nan
        self._last_total_heating_power[env_id] = np.nan
        self._last_wall_fluxes[env_id] = np.nan
        self._last_times[env_id] = np.nan
        self._last_solver_steps[env_id] = -1
        self._last_raw_wall_maps[env_id] = None
        self._last_tbar[env_id] = np.nan
        self._last_kbar[env_id] = np.nan
        self._last_reward_components[env_id] = {}

        # Seed both request tensors before NekRS starts. The UDF publishes its
        # zero-control restart frame, then waits for sequence 1.
        self.client.put_tensor(
            self._database_key(model_name, ACTION_KEY),
            self.initial_action.copy(),
        )
        self.client.put_tensor(
            self._database_key(model_name, ACTION_SEQUENCE_KEY),
            np.asarray([0], dtype=np.int64),
        )
        self.client.put_tensor(
            self._database_key(model_name, SHUTDOWN_KEY),
            np.asarray([0], dtype=np.int64),
        )
        self.exp.start(model, block=False, summary=False)

    def _select_snapshot(self, env_id: int = 0) -> Path:
        """Draw checkpoints from a shuffled cycle before reusing any one."""
        if getattr(self, "_snapshot_schedules", None) is not None:
            schedule = self._snapshot_schedules[env_id]
            index = self._snapshot_schedule_indices[env_id]
            self._snapshot_schedule_indices[env_id] += 1
            return schedule[index % len(schedule)]
        if self.snapshot_path is not None:
            return self.snapshot_path
        assert self.snapshots_dir is not None
        files = sorted(path for path in self.snapshots_dir.iterdir() if path.is_file())
        unexpected = [
            path.name for path in files
            if re.fullmatch(r"blanket0\.f\d+", path.name) is None
        ]
        if unexpected:
            raise ValueError(
                "Snapshot directory contains noncanonical files: "
                + ", ".join(unexpected)
            )
        snapshots = files
        if not snapshots:
            raise FileNotFoundError(
                f"No checkpoint files found in snapshot directory: {self.snapshots_dir}"
            )
        if not self._snapshot_queue:
            self._snapshot_queue = snapshots.copy()
            self._snapshot_rng.shuffle(self._snapshot_queue)
        return self._snapshot_queue.pop()

    @staticmethod
    def _set_model_restart_file(model, snapshot_name: str) -> None:
        """Canonicalize and reference a staged ``blanket0.fNNNNN`` restart."""
        model_directory = Path(model.path)
        snapshot_match = re.fullmatch(
            r"(?i)(?:blanketzik|blanket)0\.f(\d+)", snapshot_name
        )
        if snapshot_match is None:
            raise ValueError(
                "Restart filename must look like blanket0.fNNNNN (legacy "
                f"blanketZik names are also recognized): {snapshot_name}"
            )
        canonical_snapshot_name = f"blanket0.f{snapshot_match.group(1)}"
        staged_snapshot = model_directory / snapshot_name
        canonical_snapshot = model_directory / canonical_snapshot_name
        if staged_snapshot != canonical_snapshot:
            if canonical_snapshot.exists():
                raise FileExistsError(
                    f"Canonical restart already exists: {canonical_snapshot}"
                )
            staged_snapshot.rename(canonical_snapshot)
        if not canonical_snapshot.is_file():
            raise FileNotFoundError(
                f"Staged restart was not generated: {canonical_snapshot}"
            )

        parameter_files = sorted(model_directory.glob("*.par"))
        if not parameter_files:
            raise FileNotFoundError(
                f"No NekRS .par file generated in model directory: {model_directory}"
            )

        start_from_pattern = re.compile(
            r"(?im)^[ \t]*#?[ \t]*startFrom[ \t]*=.*$"
        )
        general_pattern = re.compile(r"(?im)^\[GENERAL\][ \t]*$")
        for parameter_file in parameter_files:
            contents = parameter_file.read_text()
            replacement = f"startFrom = {canonical_snapshot_name}"
            if start_from_pattern.search(contents):
                contents = start_from_pattern.sub(replacement, contents, count=1)
            else:
                general_match = general_pattern.search(contents)
                if general_match is None:
                    raise ValueError(
                        f"NekRS parameter file has no [GENERAL] section: "
                        f"{parameter_file}"
                    )
                insert_at = general_match.end()
                contents = (
                    contents[:insert_at]
                    + "\n"
                    + replacement
                    + contents[insert_at:]
                )
            parameter_file.write_text(contents)

    def _frame_key(self, env_id: int, field: str) -> str:
        return self._key(env_id, RANK_ZERO_PREFIX + field)

    def _get_wall_plans(self, env_id: int) -> tuple[_WallReconstructionPlan, ...]:
        plans = self._wall_plans[env_id]
        if plans is not None:
            return plans
        plans = tuple(
            _build_wall_plan(
                self.client.get_tensor(self._frame_key(env_id, wall + "Transverse")),
                self.client.get_tensor(self._frame_key(env_id, wall + "Z")),
                self.wall_shape,
            )
            for wall in WALL_CHANNELS
        )
        self._wall_plans[env_id] = plans
        return plans

    def _read_wall_maps(self, env_id: int) -> np.ndarray:
        channels = []
        for wall, plan in zip(WALL_CHANNELS, self._get_wall_plans(env_id)):
            raw_heat_flux = self.client.get_tensor(
                self._frame_key(env_id, wall + "HeatFlux")
            )
            channels.append(plan.reconstruct(raw_heat_flux))
        return np.stack(channels)

    def _prepare_observation(self, wall_maps: np.ndarray) -> np.ndarray:
        """Standardize one complete three-wall observation without fitted data.

        A single mean and standard deviation are taken over all channels and
        spatial points. This preserves relative amplitudes between walls while
        making every CNN input order-one. The first-wall routing reward retains
        an absolute physical performance signal outside the actor observation.
        """
        wall_maps = np.asarray(wall_maps, dtype=np.float64)
        if not np.all(np.isfinite(wall_maps)):
            raise ValueError("Heat-flux observation contains non-finite values")
        if getattr(self, "observation_scaling", "per_frame") == "fixed":
            mean = float(self.fixed_wall_mean)
            standard_deviation = float(self.fixed_wall_std)
        else:
            mean = float(wall_maps.mean())
            standard_deviation = float(wall_maps.std())
        scale = max(standard_deviation, 1e-8)
        return np.ascontiguousarray((wall_maps - mean) / scale, dtype=np.float32)

    def _first_wall_heat_flux_reward(
        self,
        env_id: int,
        first_wall_heat_flux: float,
        total_heating_power: float,
        actions: np.ndarray,
    ) -> float:
        """Reward heat redirected away from the plasma-facing x-minus wall."""
        if total_heating_power <= 0:
            raise ValueError("totalHeatingPower must be positive")
        previous_actions = self._last_actions[env_id]
        self._first_wall_flux_hat[env_id] += self.ewma_alpha * (
            float(first_wall_heat_flux) - self._first_wall_flux_hat[env_id]
        )
        redirected_heat_fraction = (
            self.first_wall_heat_flux_baseline
            - self._first_wall_flux_hat[env_id]
        ) / float(total_heating_power)
        magnitude_cost = self.action_magnitude_regularization * float(
            np.mean(np.square(actions))
        )
        slew_cost = self.action_slew_regularization * float(
            np.mean(np.square(actions - previous_actions))
        )
        reward = (
            self.reward_scale * redirected_heat_fraction
            - magnitude_cost
            - slew_cost
        )
        self._last_actions[env_id] = np.asarray(actions, dtype=np.float64).copy()
        self._last_reward_components[env_id] = {
            "first_wall_heat_flux": float(first_wall_heat_flux),
            "first_wall_heat_flux_ewma": float(self._first_wall_flux_hat[env_id]),
            "first_wall_heat_flux_baseline": self.first_wall_heat_flux_baseline,
            "total_heating_power": float(total_heating_power),
            "redirected_heat_fraction": float(redirected_heat_fraction),
            "action_magnitude_cost": magnitude_cost, "action_slew_cost": slew_cost,
        }
        return float(reward)

    def _read_frame(self, env_id: int, wait_for_new: bool) -> tuple[np.ndarray, float]:
        """Read a consistent wall frame and compute its local physical reward."""
        step_key = self._frame_key(env_id, "step")
        time_key = self._frame_key(env_id, "time")
        # ``time`` is the UDF's commit marker and is published after every
        # field, action sequence, and solver step belonging to the frame.
        self._block_until_exists(time_key, self.startup_timeout)

        while True:
            committed_time = float(self.client.get_tensor(time_key).reshape(-1)[0])
            step = int(self.client.get_tensor(step_key).reshape(-1)[0])
            if wait_for_new and committed_time == self._last_times[env_id]:
                self._check_model_alive(env_id)
                time.sleep(self.poll_interval)
                continue

            wall_maps = self._read_wall_maps(env_id)
            nu = float(self.client.get_tensor(self._frame_key(env_id, "Nu")).reshape(-1)[0])
            tbar = float(self.client.get_tensor(self._frame_key(env_id, "Tbar")).reshape(-1)[0])
            kbar = float(self.client.get_tensor(self._frame_key(env_id, "Kbar")).reshape(-1)[0])
            first_wall_heat_flux = float(self.client.get_tensor(
                self._frame_key(env_id, "xMinusIntegratedHeatFlux")
            ).reshape(-1)[0])
            total_heating_power = float(self.client.get_tensor(
                self._frame_key(env_id, "totalHeatingPower")
            ).reshape(-1)[0])
            wall_fluxes = np.asarray([
                self.client.get_tensor(
                    self._frame_key(env_id, name + "IntegratedHeatFlux")
                ).reshape(-1)[0]
                for name in ("xMinus", "xPlus", "yMinus", "yPlus")
            ], dtype=np.float64)
            physical_time = committed_time
            actions = np.asarray(self.client.get_tensor(
                self._frame_key(env_id, "timeIntensity")), dtype=np.float64).reshape(-1)
            action_sequence = int(self.client.get_tensor(
                self._frame_key(env_id, "actionSequence")).reshape(-1)[0])
            # Re-read both commit marker and step to reject a mixture if a new
            # frame arrived during the relatively large wall-field reads.
            if (
                float(self.client.get_tensor(time_key).reshape(-1)[0]) != committed_time
                or int(self.client.get_tensor(step_key).reshape(-1)[0]) != step
            ):
                continue

            observation = self._prepare_observation(wall_maps)
            if actions.shape != (self.act_dim,):
                raise ValueError(f"Published action has shape {actions.shape}")
            if not np.all(np.isfinite([
                nu, tbar, kbar, first_wall_heat_flux, total_heating_power
            ])):
                raise ValueError(
                    f"Environment {env_id} published non-finite diagnostics: "
                    f"Nu={nu}, Tbar={tbar}, Kbar={kbar}, "
                    f"xMinusIntegratedHeatFlux={first_wall_heat_flux}, "
                    f"totalHeatingPower={total_heating_power}"
                )
            if wait_for_new and action_sequence != self._action_sequences[env_id]:
                raise RuntimeError(
                    f"Environment {env_id} published action sequence {action_sequence}, "
                    f"expected {self._action_sequences[env_id]}"
                )
            if wait_for_new:
                reward = self._first_wall_heat_flux_reward(
                    env_id, first_wall_heat_flux, total_heating_power, actions
                )
            else:
                reward = 0.0
                self._last_actions[env_id] = actions.copy()
            self._last_nu[env_id], self._last_tbar[env_id], self._last_kbar[env_id] = nu, tbar, kbar
            self._last_first_wall_flux[env_id] = first_wall_heat_flux
            self._last_total_heating_power[env_id] = total_heating_power
            self._last_wall_fluxes[env_id] = wall_fluxes
            self._last_times[env_id] = physical_time
            self._last_solver_steps[env_id] = step
            self._last_raw_wall_maps[env_id] = np.ascontiguousarray(
                wall_maps, dtype=np.float32
            )
            if observation.size != self.obs_dim:
                raise ValueError(
                    f"Environment {env_id} reconstructed {observation.size} observation "
                    f"values, but obs_dim={self.obs_dim}. Reconstructed shape: "
                    f"{observation.shape}"
                )
            self._last_steps[env_id] = step
            self._last_observations[env_id] = observation.copy()
            self._last_rewards[env_id] = reward
            return observation, reward

    def _restart_sim(
        self, env_id: int, preserve_current_directory: bool = False
    ) -> np.ndarray:
        """Launch the next episode and return its first committed frame."""
        self._cleanup_sim(
            env_id, preserve_directory=preserve_current_directory
        )

        self._episode_id[env_id] += 1
        self._episode_steps[env_id] = 0
        self._last_steps[env_id] = None
        self._last_observations[env_id] = None
        self._last_rewards[env_id] = None

        self._launch_sim(env_id)
        observation, _ = self._read_frame(env_id, wait_for_new=False)
        return observation

    def _restart_completed_sim(self, env_id: int) -> np.ndarray:
        """Restart after a normal episode boundary, recovering startup failures."""
        try:
            return self._restart_sim(env_id)
        except _SimulationFailed as failure:
            observation, _ = self._recover_failed_sim(env_id, failure)
            return observation

    def _recover_failed_sim(
        self, env_id: int, failure: _SimulationFailed
    ) -> tuple[np.ndarray, list[_SimulationFailed]]:
        """Replace a failed process, retaining diagnostics for every failure."""
        failures = [failure]
        for attempt in range(1, self.max_failure_restarts + 1):
            failed = failures[-1]
            print(
                f"Recovering NekRS environment {env_id} after {failed.status} "
                f"({failed.model_name}); preserving {failed.model_directory}. "
                f"Replacement attempt {attempt}/{self.max_failure_restarts}."
            )
            try:
                observation = self._restart_sim(
                    env_id, preserve_current_directory=True
                )
                return observation, failures
            except _SimulationFailed as replacement_failure:
                failures.append(replacement_failure)

        failed_models = ", ".join(item.model_name for item in failures)
        raise RuntimeError(
            f"NekRS environment {env_id} failed and all "
            f"{self.max_failure_restarts} replacement attempts failed; "
            f"preserved model directories for: {failed_models}"
        ) from failures[-1]

    def _gracefully_stop_bede_sim(self, env_id: int, model) -> bool:
        """Ask a blocked Bede worker to finalize MPI and exit successfully."""
        try:
            status = str(self.exp.get_status(model)[0]).upper()
            if status.endswith("COMPLETED"):
                return True
            if status.endswith(("FAILED", "CANCELLED")):
                return False

            # Shutdown is committed by the sequence write, exactly like a
            # regular action.  It releases the UDF's blocking exchange and
            # asks NekRS to finish through its normal final-step path.
            self.client.put_tensor(
                self._database_key(model.name, SHUTDOWN_KEY),
                np.asarray([1], dtype=np.int64),
            )
            self.client.put_tensor(
                self._database_key(model.name, ACTION_KEY),
                self._last_actions[env_id].copy(),
            )
            self._action_sequences[env_id] += 1
            self.client.put_tensor(
                self._database_key(model.name, ACTION_SEQUENCE_KEY),
                np.asarray(
                    [self._action_sequences[env_id]], dtype=np.int64
                ),
            )

            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                status = str(self.exp.get_status(model)[0]).upper()
                if status.endswith("COMPLETED"):
                    return True
                if status.endswith(("FAILED", "CANCELLED")):
                    return False
                time.sleep(self.poll_interval)
        except Exception:
            return False
        return False

    def _cleanup_sim(self, env_id: int, preserve_directory: bool = False) -> None:
        """Release one episode's process, database, and filesystem resources.

        Cleanup is deliberately idempotent: a normally completed Slurm step
        may no longer be stoppable, and some tensors may never have been
        published if the simulation exited during startup.
        """
        model = self._models[env_id]
        if model is None:
            return

        stopped_cleanly = (
            self.launcher == "bede_mpirun"
            and self._gracefully_stop_bede_sim(env_id, model)
        )
        if not stopped_cleanly:
            try:
                self.exp.stop(model)
            except Exception:
                # Completed models are already absent from the launcher.
                # Cleanup must still continue so their database entries do not
                # accumulate.
                pass

        for key in MODEL_TENSOR_KEYS:
            database_key = self._database_key(model.name, key)
            try:
                if self.client.tensor_exists(database_key):
                    self.client.delete_tensor(database_key)
            except Exception:
                # The database may itself be shutting down from close().
                pass

        model_directory = Path(model.path)
        expected_name = f"nekrs_env_{env_id}_ep_{self._episode_id[env_id]}"
        if (
            model.name != expected_name
            or model_directory.name != expected_name
            or re.fullmatch(r"nekrs_env_\d+_ep_\d+", expected_name) is None
        ):
            raise RuntimeError(
                "Refusing to delete unexpected SmartSim model directory: "
                f"{model_directory}"
            )
        if model_directory.is_symlink():
            raise RuntimeError(
                f"Refusing to delete symlinked SmartSim model directory: {model_directory}"
            )
        if model_directory.exists() and not preserve_directory:
            shutil.rmtree(model_directory)

        self._models[env_id] = None
        self._episode_snapshots[env_id] = None

    def reset(self) -> np.ndarray:
        """Return initial frames, restarting active environments if necessary."""
        observations = np.empty(
            (self.n_envs, *self.observation_shape), dtype=np.float32
        )
        for env_id in range(self.n_envs):
            if self._last_steps[env_id] is None:
                try:
                    observations[env_id], _ = self._read_frame(
                        env_id, wait_for_new=False
                    )
                except _SimulationFailed as failure:
                    observations[env_id], _ = self._recover_failed_sim(
                        env_id, failure
                    )
            else:
                observations[env_id] = self._restart_sim(env_id)
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        policy_actions = np.asarray(actions, dtype=np.float64)
        if policy_actions.shape != (self.n_envs, self.act_dim):
            raise ValueError(
                f"Expected actions with shape {(self.n_envs, self.act_dim)}, "
                f"got {policy_actions.shape}"
            )
        if not np.all(np.isfinite(policy_actions)):
            raise ValueError("actions must contain only finite values")
        if self._actions_pending:
            raise RuntimeError("step_async() called before the previous step_wait()")

        physical_actions = self.policy_to_physical_action(policy_actions)
        for env_id in range(self.n_envs):
            self.client.put_tensor(
                self._key(env_id, ACTION_KEY),
                physical_actions[env_id],
            )
            self._action_sequences[env_id] += 1
            self.client.put_tensor(
                self._key(env_id, ACTION_SEQUENCE_KEY),
                np.asarray([self._action_sequences[env_id]], dtype=np.int64),
            )
        self._pending_policy_actions = policy_actions.copy()
        self._pending_actions = physical_actions.copy()
        self._actions_pending = True

    def step_wait(self):
        if not self._actions_pending:
            raise RuntimeError("step_wait() called without step_async()")

        observations = np.empty(
            (self.n_envs, *self.observation_shape), dtype=np.float32
        )
        rewards = np.empty(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [{} for _ in range(self.n_envs)]

        try:
            for env_id in range(self.n_envs):
                try:
                    observation, reward = self._read_frame(
                        env_id, wait_for_new=True
                    )
                except _EpisodeCompleted:
                    # This fallback prevents a normal NekRS completion from
                    # crashing training. For exact RL transitions, configure
                    # max_episode_steps so completion is anticipated on the
                    # final committed frame rather than detected one call late.
                    terminal_observation = self._last_observations[env_id]
                    if terminal_observation is None:
                        raise RuntimeError(
                            f"NekRS environment {env_id} completed before "
                            "publishing an initial frame"
                        )

                    rewards[env_id] = float(self._last_rewards[env_id] or 0.0)
                    dones[env_id] = True
                    infos[env_id] = {
                        "terminal_observation": terminal_observation.copy(),
                        "TimeLimit.truncated": True,
                        "episode_end_reason": "nekrs_completed_before_new_frame",
                    }
                    terminal_physical_action = self._last_actions[env_id].copy()
                    completed_episode_id = self._episode_id[env_id]
                    completed_episode_steps = self._episode_steps[env_id]
                    completed_snapshot = self._episode_snapshots[env_id]
                    observations[env_id] = self._restart_completed_sim(env_id)
                    infos[env_id]["terminal_physical_action"] = (
                        terminal_physical_action
                    )
                    infos[env_id].update(
                        {
                            "env_id": env_id,
                            "episode_id": completed_episode_id,
                            "episode_step": completed_episode_steps,
                            "snapshot": str(completed_snapshot),
                        }
                    )
                    continue
                except _SimulationFailed as failure:
                    terminal_observation = self._last_observations[env_id]
                    if terminal_observation is None:
                        raise RuntimeError(
                            f"NekRS environment {env_id} failed before "
                            "publishing an initial frame"
                        ) from failure

                    terminal_physical_action = self._last_actions[env_id].copy()
                    failed_episode_id = self._episode_id[env_id]
                    failed_episode_steps = self._episode_steps[env_id]
                    observations[env_id], failures = self._recover_failed_sim(
                        env_id, failure
                    )
                    rewards[env_id] = self.simulation_failure_penalty
                    dones[env_id] = True
                    infos[env_id] = {
                        "terminal_observation": terminal_observation.copy(),
                        "TimeLimit.truncated": False,
                        "episode_end_reason": "nekrs_failed",
                        "model_status": failure.status,
                        "failed_model": failure.model_name,
                        "failed_snapshot": (
                            str(failure.snapshot)
                            if failure.snapshot is not None
                            else None
                        ),
                        "failure_directory": str(failure.model_directory),
                        "recovery_failures": len(failures) - 1,
                        "terminal_physical_action": terminal_physical_action,
                        "env_id": env_id,
                        "episode_id": failed_episode_id,
                        "episode_step": failed_episode_steps,
                        "snapshot": (
                            str(failure.snapshot)
                            if failure.snapshot is not None
                            else None
                        ),
                    }
                    continue

                rewards[env_id] = reward
                self._episode_steps[env_id] += 1
                transition_diagnostics = dict(
                    self._last_reward_components[env_id]
                )
                transition_diagnostics.update(
                    {
                        "env_id": env_id,
                        "episode_id": int(self._episode_id[env_id]),
                        "Tbar": float(self._last_tbar[env_id]),
                        "Kbar": float(self._last_kbar[env_id]),
                        "Nu": float(self._last_nu[env_id]),
                        "action_sequence": int(self._action_sequences[env_id]),
                        "episode_step": int(self._episode_steps[env_id]),
                        "snapshot": str(self._episode_snapshots[env_id]),
                        "solver_step": int(self._last_solver_steps[env_id]),
                        "physical_time": float(self._last_times[env_id]),
                        "x_plus_integrated_heat_flux": float(
                            self._last_wall_fluxes[env_id, 1]
                        ),
                        "y_minus_integrated_heat_flux": float(
                            self._last_wall_fluxes[env_id, 2]
                        ),
                        "y_plus_integrated_heat_flux": float(
                            self._last_wall_fluxes[env_id, 3]
                        ),
                        "allocation_gpu": int(self._gpu_id(env_id)),
                        "applied_physical_action": self._last_actions[
                            env_id
                        ].copy(),
                    }
                )

                horizon_reached = (
                    self.max_episode_steps is not None
                    and self._episode_steps[env_id] >= self.max_episode_steps
                )

                if horizon_reached:
                    dones[env_id] = True
                    infos[env_id] = {
                        "terminal_observation": observation.copy(),
                        "terminal_raw_wall_maps": self._last_raw_wall_maps[
                            env_id
                        ].copy(),
                        "TimeLimit.truncated": self.episode_horizon_truncated,
                        "episode_end_reason": "max_episode_steps",
                        "terminal_physical_action": self._last_actions[
                            env_id
                        ].copy(),
                    }

                    # SB3 VecEnv semantics require the observation returned
                    # alongside done=True to be the first observation of the
                    # automatically reset next episode.
                    observations[env_id] = self._restart_completed_sim(env_id)
                else:
                    observations[env_id] = observation

                infos[env_id].update(transition_diagnostics)

            for env_id, info in enumerate(infos):
                info["policy_action"] = self._pending_policy_actions[env_id].copy()
                info["physical_action"] = self._pending_actions[env_id].copy()
                self._last_policy_actions[env_id] = self._pending_policy_actions[
                    env_id
                ].copy()

        finally:
            self._pending_policy_actions = None
            self._pending_actions = None
            self._actions_pending = False

        return observations, rewards, dones, infos

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def _check_model_alive(self, env_id: int) -> None:
        model = self._models[env_id]
        if model is None:
            raise RuntimeError(f"Environment {env_id} has no active model")

        status = str(self.exp.get_status(model)[0]).upper()

        if status.endswith("COMPLETED"):
            raise _EpisodeCompleted(
                f"NekRS environment {env_id} completed normally"
            )

        if status.endswith(("FAILED", "CANCELLED")):
            raise _SimulationFailed(
                env_id=env_id,
                status=status,
                model_name=model.name,
                model_directory=Path(model.path),
                snapshot=self._episode_snapshots[env_id],
            )

    def _block_until_exists(self, key: str, timeout: float | None) -> None:
        start = time.monotonic()
        while not self.client.tensor_exists(key):
            # A failed simulation can never publish this tensor.  Report its
            # status immediately instead of hiding the launcher error behind
            # the (potentially long) startup timeout.
            for env_id in range(self.n_envs):
                model = self._models[env_id]
                if model is not None and key.startswith(f"{model.name}."):
                    self._check_model_alive(env_id)
                    break
            if timeout is not None and time.monotonic() - start > timeout:
                for env_id in range(self.n_envs):
                    model = self._models[env_id]
                    if model is not None and key.startswith(f"{model.name}."):
                        raise _SimulationFailed(
                            env_id=env_id,
                            status="STARTUP_TIMEOUT",
                            model_name=model.name,
                            model_directory=Path(model.path),
                            snapshot=self._episode_snapshots[env_id],
                        ) from TimeoutError(
                            f"Timed out waiting for SmartRedis tensor {key}"
                        )
                raise TimeoutError(f"Timed out waiting for SmartRedis tensor {key}")
            time.sleep(self.poll_interval)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for env_id in range(self.n_envs):
            self._cleanup_sim(env_id)
        if hasattr(self, "db"):
            try:
                self.exp.stop(self.db)
            except Exception:
                pass


from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnvBase


class SB3MARLSmartSimVecEnv(SB3VecEnvBase):
    """Expose one physical simulation's ten electrodes as virtual agents.

    Rows are ordered by ``(simulation, agent)``. The ten agents share a policy
    and receive the same physical reward. Each heat-flux view and previous
    joint action are shifted so the acting electrode occupies index zero.
    """

    N_AGENTS = 10
    ACTION_NORMALIZATION = ACTION_HIGH

    def __init__(self, core: SmartSimVecEnv, observation_space, action_space):
        self.core = core
        if not hasattr(observation_space, "spaces"):
            raise ValueError("Expected a Dict MARL observation space")
        expected_keys = {"heat_flux", "previous_actions"}
        if set(observation_space.spaces) != expected_keys:
            raise ValueError(
                f"Expected MARL observation keys {expected_keys}, got "
                f"{set(observation_space.spaces)}"
            )
        heat_flux_space = observation_space.spaces["heat_flux"]
        previous_actions_space = observation_space.spaces["previous_actions"]
        expected_heat_flux_shape = (3, *core.wall_shape)
        if heat_flux_space.shape != expected_heat_flux_shape:
            raise ValueError(
                "Expected heat-flux observation shape "
                f"{expected_heat_flux_shape}, got {heat_flux_space.shape}"
            )
        if previous_actions_space.shape != (self.N_AGENTS,):
            raise ValueError(
                "Expected previous_actions observation shape "
                f"{(self.N_AGENTS,)}, got {previous_actions_space.shape}"
            )
        if action_space.shape != (1,):
            raise ValueError(
                f"Expected per-agent action space with shape (1,), got "
                f"{action_space.shape}"
            )

        if core.wall_shape[1] % self.N_AGENTS != 0:
            raise ValueError(
                "The periodic wall dimension must be divisible by the number "
                "of agents"
            )
        self._heat_flux_dtype = heat_flux_space.dtype
        self._action_observation_dtype = previous_actions_space.dtype
        self._agent_shift = core.wall_shape[1] // self.N_AGENTS
        super().__init__(core.n_envs * self.N_AGENTS, observation_space, action_space)

    def _agent_observations(self, heat_fluxes, previous_actions):
        heat_fluxes = np.asarray(heat_fluxes, dtype=self._heat_flux_dtype)
        previous_actions = np.asarray(previous_actions, dtype=np.float64)
        expected_heat_flux_shape = (self.core.n_envs, 3, *self.core.wall_shape)
        expected_action_shape = (self.core.n_envs, self.N_AGENTS)
        if heat_fluxes.shape != expected_heat_flux_shape:
            raise ValueError(
                f"Expected heat fluxes with shape {expected_heat_flux_shape}, "
                f"got {heat_fluxes.shape}"
            )
        if previous_actions.shape != expected_action_shape:
            raise ValueError(
                f"Expected previous actions with shape {expected_action_shape}, "
                f"got {previous_actions.shape}"
            )

        centred_heat_fluxes = []
        relative_actions = []
        for env_id in range(self.core.n_envs):
            for agent_idx in range(self.N_AGENTS):
                # Moving electrode i back by i tenth-periods aligns it with
                # electrode 0's position in the unshifted view.
                centred_heat_fluxes.append(
                    np.roll(
                        heat_fluxes[env_id],
                        shift=-agent_idx * self._agent_shift,
                        axis=2,
                    )
                )
                relative_actions.append(
                    np.roll(previous_actions[env_id], shift=-agent_idx)
                    / self.ACTION_NORMALIZATION
                )
        return {
            "heat_flux": np.ascontiguousarray(
                centred_heat_fluxes, dtype=self._heat_flux_dtype
            ),
            "previous_actions": np.ascontiguousarray(
                relative_actions, dtype=self._action_observation_dtype
            ),
        }

    def _single_agent_observation(
        self, heat_flux, previous_actions, agent_idx
    ):
        return {
            "heat_flux": np.ascontiguousarray(
                np.roll(
                    heat_flux,
                    shift=-agent_idx * self._agent_shift,
                    axis=2,
                ),
                dtype=self._heat_flux_dtype,
            ),
            "previous_actions": np.ascontiguousarray(
                np.roll(previous_actions, shift=-agent_idx)
                / self.ACTION_NORMALIZATION,
                dtype=self._action_observation_dtype,
            ),
        }

    def reset(self):
        heat_fluxes = self.core.reset()
        return self._agent_observations(heat_fluxes, self.core._last_actions)

    def step_async(self, actions):
        actions = np.asarray(actions)
        expected_shape = (self.num_envs, 1)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape}, got {actions.shape}"
            )
        self.core.step_async(actions.reshape(self.core.n_envs, self.N_AGENTS))

    def step_wait(self):
        observations, rewards, dones, infos = self.core.step_wait()
        virtual_infos = []
        for env_id, info in enumerate(infos):
            for agent_idx in range(self.N_AGENTS):
                agent_info = info.copy()
                if "terminal_observation" in agent_info:
                    terminal = np.asarray(agent_info["terminal_observation"])
                    terminal_action = np.asarray(
                        agent_info.pop("terminal_physical_action")
                    )
                    agent_info["terminal_observation"] = (
                        self._single_agent_observation(
                            terminal, terminal_action, agent_idx
                        )
                    )
                virtual_infos.append(agent_info)
        return (
            self._agent_observations(observations, self.core._last_actions),
            np.repeat(rewards, self.N_AGENTS, axis=0),
            np.repeat(dones, self.N_AGENTS, axis=0),
            virtual_infos,
        )

    def close(self):
        self.core.close()

    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self.core, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        self._get_indices(indices)
        setattr(self.core, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(
            "The environments are external NekRS processes, not Python objects"
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
