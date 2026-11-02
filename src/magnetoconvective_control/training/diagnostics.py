"""Compact rollout diagnostics; no field storage or additional environment steps."""

from contextlib import contextmanager
import csv
import json
from importlib.metadata import version
import os
from pathlib import Path
import sys

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO


def append_rows(path, rows):
    if not rows:
        return
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if header:
            writer.writeheader()
        writer.writerows(rows)


def policy_means(model, buffer):
    """Replay stored actor states deterministically, without advancing live states."""
    means = []
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    try:
        with th.no_grad():
            for step in range(buffer.buffer_size):
                observations = {
                    key: th.as_tensor(value[step], device=model.device)
                    for key, value in buffer.observations.items()
                }
                states = tuple(th.as_tensor(value[step], device=model.device) for value in (
                    buffer.hidden_states_pi, buffer.cell_states_pi,
                ))
                distribution, _ = model.policy.get_distribution(
                    observations, states,
                    th.as_tensor(buffer.episode_starts[step], device=model.device),
                )
                means.append(distribution.mode().cpu().numpy())
    finally:
        model.policy.set_training_mode(was_training)
    return np.asarray(means)


def complete_returns(rewards, dones, valid_terminal, gamma):
    """Return-to-go only where a genuine, non-failed terminal is observed.

    NaN marks rollout cuts, time-limit bootstraps, and failed episode tails.
    Episodes may start before this buffer; their observed suffix is still exact.
    """
    result = np.full_like(rewards, np.nan, dtype=np.float64)
    tail = np.full(rewards.shape[1], np.nan)
    for step in reversed(range(len(rewards))):
        tail = np.where(dones[step], np.where(valid_terminal[step], 0.0, np.nan), tail)
        tail = rewards[step] + gamma * tail
        result[step] = tail
    return result


class TrainingDiagnostics:
    PHYSICS = (
        "first_wall_heat_flux", "first_wall_heat_flux_ewma", "total_heating_power",
        "Nu", "Tbar", "Kbar", "physical_time", "solver_step", "action_sequence",
    )

    def __init__(self, model, directory, run_id):
        self.model = model
        self.directory = Path(directory) / "diagnostics" / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rollout = 0
        self.steps = []
        self.agents = 10

    def begin(self):
        self.rollout += 1
        self.steps = []
        self.start_timesteps = int(self.model.num_timesteps)
        self.policy_version = int(self.model._n_updates)

    def identity(self):
        return dict(run_id=self.run_id, rollout=self.rollout,
                    policy_version=self.policy_version, start_timesteps=self.start_timesteps,
                    end_timesteps=int(self.model.num_timesteps))

    def record(self, local):
        infos = local["infos"][::self.agents]
        get = lambda key, default=np.nan: np.asarray([i.get(key, default) for i in infos])
        done = np.asarray(local["dones"], dtype=bool)[::self.agents].copy()
        reason = get("episode_end_reason", "")
        truncated = get("TimeLimit.truncated", False).astype(bool)
        self.steps.append({
            **{key: get(key) for key in self.PHYSICS},
            "episode_id": get("episode_id", -1), "episode_step": get("episode_step", -1),
            "snapshot": get("snapshot", "").astype(str), "end_reason": reason,
            "done": done, "truncated": truncated,
            "valid_terminal": done & ~truncated & (reason == "max_episode_steps"),
            "initial_first_wall_heat_flux": np.where(
                get("episode_step", -1) == 1, get("initial_first_wall_heat_flux"), np.nan),
            "physical_reward": np.asarray(local["rewards"])[::self.agents].copy(),
            "applied_actions": np.asarray([
                i.get("applied_physical_action", np.full(self.agents, np.nan)) for i in infos
            ]),
        })

    def finish(self):
        buffer = self.model.rollout_buffer
        data = {key: np.asarray([step[key] for step in self.steps]) for key in self.steps[0]}
        shape = (buffer.buffer_size, buffer.n_envs // self.agents, self.agents)
        for name in ("actions", "values", "advantages", "returns", "log_probs", "episode_starts", "rewards"):
            data[name] = getattr(buffer, name).reshape(shape).copy()
        data["policy_means"] = policy_means(self.model, buffer).reshape(shape)
        data["policy_std"] = self.model.policy.log_std.detach().exp().cpu().numpy().copy()
        data["return_to_go"] = complete_returns(
            data["physical_reward"], data["done"], data["valid_terminal"], self.model.gamma)
        # Includes timeout reward corrections. A rollout-cut bootstrap is separate.
        data["timeout_reward_correction"] = data["rewards"] - data["physical_reward"][..., None]
        data["rollout_bootstrap_value"] = self.model._last_rollout_values.copy()
        data["rollout_bootstrap_mask"] = ~data["done"][-1]
        data.update(self.identity(), gamma=self.model.gamma, gae_lambda=self.model.gae_lambda,
                    schema_version=1, complete_rollout=True)
        self._save(data)
        self._summarize(data)
        self.steps.clear()

    def save_partial(self):
        if self.steps:
            data = {key: np.asarray([step[key] for step in self.steps]) for key in self.steps[0]}
            data.update(self.identity(), schema_version=1, complete_rollout=False)
            self._save(data)
            self.steps.clear()

    def _save(self, data):
        path = self.directory / f"rollout_{self.rollout:04d}.npz"
        with path.with_suffix(".partial").open("wb") as stream:
            np.savez_compressed(stream, **data)
        os.replace(path.with_suffix(".partial"), path)

    def _summarize(self, data):
        rows = []
        low, high = self.model.action_space.low, self.model.action_space.high
        for env in range(data["physical_reward"].shape[1]):
            for phase in sorted(set(((data["episode_step"][:, env] - 1) // 50).tolist())):
                if phase < 0:
                    continue
                mask = ((data["episode_step"][:, env] - 1) // 50) == phase
                mean = lambda x: float(np.mean(x[mask, env]))
                mc = data["return_to_go"][mask, env, None] - data["values"][mask, env]
                valid = np.isfinite(mc)
                error = data["advantages"][mask, env] - mc
                sampled = data["actions"][mask, env]
                applied = data["applied_actions"][mask, env]
                rows.append({
                    **self.identity(), "env_id": env, "phase_start": int(phase * 50 + 1),
                    "phase_stop": int((phase + 1) * 50), "physical_steps": int(mask.sum()),
                    "reward_mean": mean(data["physical_reward"]),
                    "raw_flux_mean": mean(data["first_wall_heat_flux"]),
                    "filtered_flux_mean": mean(data["first_wall_heat_flux_ewma"]),
                    "value_mean": mean(data["values"]), "advantage_mean": mean(data["advantages"]),
                    "target_mean": mean(data["returns"]),
                    "complete_return_samples": int(valid.sum()),
                    "gae_mc_rmse": float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else np.nan,
                    "policy_mean": mean(data["policy_means"]),
                    "policy_mean_rms": float(np.sqrt(mean(data["policy_means"] ** 2))),
                    "policy_spatial_std": float(data["policy_means"][mask, env].std(axis=-1).mean()),
                    "action_std": float(data["policy_std"].mean()),
                    "sample_clip_fraction": float(np.mean((sampled < low) | (sampled > high))),
                    "applied_action_rms": float(np.sqrt(np.mean(applied ** 2))),
                    "spatial_action_std": float(np.mean(applied.std(axis=-1))),
                })
        append_rows(self.directory / "phases.csv", rows)

    @contextmanager
    def optimization(self):
        """Observe SB3's existing epochs and backward passes, never replay them."""
        model, records, gradients = self.model, [], []
        buffer, policy = model.rollout_buffer, model.policy
        original_get, original_evaluate = buffer.get, policy.evaluate_actions
        evaluated = []
        epoch = 0

        def evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            evaluated[:] = [x.detach() if x is not None else None for x in result]
            return result

        def before_step(optimizer, args, kwargs):
            # SB3 already clipped gradients: explicitly report the post-clip norm.
            norms = [p.grad.detach().norm() for group in optimizer.param_groups
                     for p in group["params"] if p.grad is not None]
            gradients.append(float(th.stack(norms).norm().cpu()))

        def batches(*args, **kwargs):
            nonlocal epoch
            epoch += 1
            records.clear()
            gradients.clear()
            try:
                for batch in original_get(*args, **kwargs):
                    evaluated.clear()
                    try:
                        yield batch
                    finally:
                        if evaluated:
                            values, log_prob, _ = evaluated
                            mask = batch.mask > 1e-8
                            adv = batch.advantages
                            if model.normalize_advantage:
                                adv = (adv - adv[mask].mean()) / (adv[mask].std() + 1e-8)
                            log_ratio = log_prob - batch.old_log_prob
                            ratio = log_ratio.exp()
                            clip = model.clip_range(model._current_progress_remaining)
                            if model.clip_range_vf is not None:
                                vf_clip = model.clip_range_vf(model._current_progress_remaining)
                                values = batch.old_values + (values.flatten() - batch.old_values).clamp(-vf_clip, vf_clip)
                            records.append([
                                float(-th.minimum(adv * ratio, adv * ratio.clamp(1 - clip, 1 + clip))[mask].mean()),
                                float(((batch.returns - values.flatten()) ** 2)[mask].mean()),
                                float((ratio - 1 - log_ratio)[mask].mean()),
                                float(((ratio - 1).abs() > clip).float()[mask].mean()),
                            ])
            finally:
                if records:
                    metrics = dict(zip(("policy_loss", "value_loss", "approx_kl", "clip_fraction"),
                                       np.mean(records, axis=0)))
                    append_rows(self.directory / "optimization.csv", [{
                        **self.identity(), "epoch": epoch, **metrics,
                        "optimizer_steps": len(gradients),
                        "gradient_norm_post_clip_mean": float(np.mean(gradients)) if gradients else np.nan,
                        "gradient_norm_post_clip_max": max(gradients, default=np.nan),
                    }])

        hook = policy.optimizer.register_step_pre_hook(before_step)
        buffer.get, policy.evaluate_actions = batches, evaluate
        try:
            yield
        finally:
            buffer.get, policy.evaluate_actions = original_get, original_evaluate
            hook.remove()


class DiagnosticRecurrentPPO(RecurrentPPO):
    """Use the upstream optimizer; associate stdout with the update just completed."""
    _diagnostics = None
    _pending_log_iteration = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_diagnostics", "_pending_log_iteration", "_last_rollout_values"]

    def collect_rollouts(self, *args, **kwargs):
        # Capture the final value query already made by SB3, without an extra forward pass.
        original = self.policy.predict_values
        def predict_values(*args, **kwargs):
            values = original(*args, **kwargs)
            self._last_rollout_values = values.detach().cpu().numpy().reshape(-1).copy()
            return values
        self.policy.predict_values = predict_values
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self.policy.predict_values = original

    def dump_logs(self, iteration=0):
        self._pending_log_iteration = iteration

    def train(self):
        if self._diagnostics is None:
            super().train()
        else:
            with self._diagnostics.optimization():
                super().train()
            self.logger.record("train/rollout", self._diagnostics.rollout)
            self.logger.record("train/collection_policy_version", self._diagnostics.policy_version)
        if self._pending_log_iteration is not None:
            super().dump_logs(self._pending_log_iteration)
            self._pending_log_iteration = None


def write_provenance(model, directory, root, core):
    from ..cli.train import _git_revision

    metadata = {
        "git": _git_revision(root), "python": sys.version,
        "packages": {name: version(name) for name in (
            "stable-baselines3", "sb3-contrib", "torch", "numpy", "gymnasium", "smartredis", "smartsim")},
        "model_seed": model.seed,
        "snapshot_selection": "SystemRandom shuffled cycles; actual selections saved in rollout records",
        "snapshot_seed": None,
        "snapshot_schedules": core._snapshot_schedules,
        "model": {name: getattr(model, name) for name in (
            "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "normalize_advantage",
            "ent_coef", "vf_coef", "max_grad_norm", "target_kl")},
        "learning_rate": model.policy.optimizer.param_groups[0]["lr"],
        "policy_kwargs": model.policy_kwargs,
    }
    for name in ("run_contract.json", "metadata.json", "action_mapping.json"):
        path = Path(directory).parents[1] / name
        if path.exists():
            metadata[name] = json.loads(path.read_text())
    resolved = Path(directory).parents[1] / "config.resolved.yaml"
    if resolved.exists():
        metadata["resolved_config_yaml"] = resolved.read_text()
    (Path(directory) / "provenance.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
