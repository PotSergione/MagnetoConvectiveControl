"""Exercise the real recurrent learner on tiny environments; never launch CFD."""

import csv
import random

import numpy as np
import pytest

th = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from magnetoconvective_control.training.diagnostics import (
    DiagnosticRecurrentPPO, complete_returns,
)
from magnetoconvective_control.training.train import TrainingCheckpointCallback


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Dict({"x": gym.spaces.Box(-1, 1, (1,), dtype=np.float32)})
    action_space = gym.spaces.Box(-0.25, 0.25, (1,), dtype=np.float32)

    def __init__(self, horizon=4, truncated=False, failed=False):
        self.episode = -1
        self.calls = 0
        self.horizon, self.truncated, self.failed = horizon, truncated, failed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_id = 0
        return {"x": np.zeros(1, dtype=np.float32)}, {}

    def step(self, action):
        self.calls += 1
        self.step_id += 1
        done = self.step_id == self.horizon
        return ({"x": np.array([self.step_id / 4], dtype=np.float32)},
                float(self.episode + 1), done and not self.truncated, done and self.truncated, {
                    "episode_id": self.episode, "episode_step": self.step_id,
                    "snapshot": "test_snapshot", "episode_end_reason": (
                        "nekrs_failed" if self.failed else "max_episode_steps") if done else "",
                    "initial_first_wall_heat_flux": 0.02, "first_wall_heat_flux": 0.01,
                    "first_wall_heat_flux_ewma": 0.012, "total_heating_power": 0.05,
                    "applied_physical_action": np.zeros(10),
                })


def learner(cls):
    env = DummyVecEnv([TinyEnv for _ in range(10)])
    model = cls("MultiInputLstmPolicy", env, n_steps=4, batch_size=20,
                n_epochs=2, gamma=0.997, gae_lambda=1, seed=42, device="cpu",
                policy_kwargs={"lstm_hidden_size": 4, "net_arch": {"pi": [4], "vf": [4]}})
    return model, env


def test_complete_returns_excludes_cuts_failures_and_timeouts():
    rewards = np.ones((5, 2))
    dones = np.array([[0, 0], [1, 1], [0, 0], [0, 1], [0, 0]], dtype=bool)
    valid = np.zeros_like(dones)
    valid[1, 0] = True
    result = complete_returns(rewards, dones, valid, 0.5)
    np.testing.assert_allclose(result[:2, 0], [1.5, 1])
    assert np.isnan(result[2:, 0]).all()
    assert np.isnan(result[:, 1]).all()


def test_diagnostics_preserve_training_and_record_exact_rollouts(tmp_path):
    th.set_num_threads(1)
    reference, reference_env = learner(RecurrentPPO)
    reference.learn(80)
    expected = {k: v.clone() for k, v in reference.policy.state_dict().items()}
    rng = (th.get_rng_state().clone(), np.random.get_state(), random.getstate())

    model, env = learner(DiagnosticRecurrentPPO)
    callback = TrainingCheckpointCallback(tmp_path, checkpoint_interval=10000, metrics_interval=25)
    model.learn(80, callback=callback)
    callback.save_final()
    for k, value in model.policy.state_dict().items():
        th.testing.assert_close(value, expected[k], rtol=0, atol=0)
    assert th.equal(th.get_rng_state(), rng[0])
    assert np.random.get_state()[0] == rng[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:]
    assert random.getstate() == rng[2]
    assert [e.calls for e in env.envs] == [8] * 10

    directory = callback.diagnostics.directory
    for index in (1, 2):
        with np.load(directory / f"rollout_{index:04d}.npz", allow_pickle=False) as data:
            assert data["actions"].shape == (4, 1, 10)
            assert data["physical_reward"].shape == (4, 1)
            assert data["physical_reward"].mean() == index
            np.testing.assert_allclose(data["returns"],
                                       np.broadcast_to(data["return_to_go"][..., None], (4, 1, 10)), rtol=1e-6)
            assert not data["rollout_bootstrap_mask"].any()
            assert data["timeout_reward_correction"].max() == 0
            assert not any("hidden" in key or "observations" in key for key in data.files)
    with (directory / "optimization.csv").open() as f:
        epochs = list(csv.DictReader(f))
    assert len(epochs) == 4
    assert [int(r["optimizer_steps"]) for r in epochs] == [2] * 4
    assert all(float(r["gradient_norm_post_clip_max"]) <= model.max_grad_norm + 1e-6 for r in epochs)
    with (tmp_path / "reward_metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == 2  # No duplicate final row or overlapping windows.
    assert [float(r["reward_mean"]) for r in metrics] == [1, 2]
    assert [float(r["recent_episode_reward_mean"]) for r in metrics] == [4, 8]
    assert float(metrics[-1]["cumulative_episode_reward_mean"]) == 6

    restored = DiagnosticRecurrentPPO.load(tmp_path / "latest.zip", env=env)
    assert restored._diagnostics is None
    reference_env.close()
    env.close()


def test_misaligned_episodes_timeouts_and_failures(tmp_path):
    th.set_num_threads(1)
    factories = ([lambda: TinyEnv(3, truncated=True)] * 10
                 + [lambda: TinyEnv(3)] * 10 + [lambda: TinyEnv(3, failed=True)] * 10)
    env = DummyVecEnv(factories)
    model = DiagnosticRecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=4, batch_size=60, n_epochs=1,
        gamma=0.997, gae_lambda=1, seed=42, device="cpu",
        policy_kwargs={"lstm_hidden_size": 4, "net_arch": {"pi": [4], "vf": [4]}},
    )
    callback = TrainingCheckpointCallback(tmp_path, 10000, 120)
    model.learn(120, callback=callback)
    with np.load(callback.diagnostics.directory / "rollout_0001.npz") as data:
        assert np.isnan(data["return_to_go"][:, 0]).all()  # timeout
        assert np.isnan(data["return_to_go"][:, 2]).all()  # failure
        np.testing.assert_allclose(data["return_to_go"][:3, 1], [1 + .997 + .997**2, 1 + .997, 1])
        assert np.isnan(data["return_to_go"][3, 1])  # next episode cut by buffer
        assert data["rollout_bootstrap_mask"].all()
        assert np.any(data["timeout_reward_correction"][2, 0] != 0)
        assert np.all(data["timeout_reward_correction"][:, 1:] == 0)
    env.close()


def test_interrupted_rollout_retains_physics_without_inventing_advantages(tmp_path):
    class StopEarly(TrainingCheckpointCallback):
        def _on_step(self):
            super()._on_step()
            return self.model.num_timesteps < 20

    model, env = learner(DiagnosticRecurrentPPO)
    callback = StopEarly(tmp_path, 10000, 40)
    model.learn(40, callback=callback)
    callback.save_final()
    with np.load(callback.diagnostics.directory / "rollout_0001.npz") as data:
        assert not data["complete_rollout"]
        assert data["physical_reward"].shape == (2, 1)
        assert "advantages" not in data
    with (tmp_path / "reward_metrics.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["complete_rollout"] == "False"
    env.close()
