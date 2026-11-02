"""Portable configuration and submission tests."""

import os
from pathlib import Path

import pytest

from magnetoconvective_control.cli.submit import build_submission_command
from magnetoconvective_control.config import (
    compose,
    export_compatibility_environment,
    merge,
    parallel_plan,
)


PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("variant,seed,epochs,noise", [
    ("reference", 5489, 5, -2.3), ("replica", 5490, 5, -2.3),
    ("one_epoch", 5489, 1, -2.3), ("low_noise", 5489, 5, -3.506557897319982),
])
def test_credit_campaign_resolves_and_exports(variant, seed, epochs, noise, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    config = compose(
        PROJECT_ROOT / f"configs/experiments/bede_v100_ppo_200_credit_{variant}.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_power9_four_gpu.yaml",
        PROJECT_ROOT / "configs/scales/bede_power9_four_gpu.yaml",
    )
    export_compatibility_environment(config)
    assert parallel_plan(config).physical_environments == 4
    expected = {
        "RL_GAMMA": .997, "RL_GAE_LAMBDA": 1, "RL_LOG_STD_INIT": noise,
        "RL_SEED": seed, "RL_SNAPSHOT_SEED": seed,
        "RL_N_STEPS": 200, "RL_MAX_EPISODE_STEPS": 200,
        "RL_N_EPOCHS": epochs, "RL_BATCH_SIZE": 1000,
        "RL_TOTAL_TIMESTEPS": 150000, "RL_METRICS_INTERVAL": 8000,
    }
    for key, value in expected.items():
        assert float(os.environ[key]) == value
    assert os.environ["RL_EPISODE_HORIZON_TRUNCATED"] == "False"


def test_credit_launcher_delegates_with_isolated_paths_and_lock(tmp_path):
    import fcntl
    import subprocess

    launcher = PROJECT_ROOT / "cluster/slurm/bede_power9_four_gpu_credit.slurm"
    scripts = tmp_path / "checkout/cluster/slurm"
    scripts.mkdir(parents=True)
    (scripts / "bede_power9_four_gpu_train.slurm").write_text(
        '#!/bin/bash\nprintenv MCC_EXPERIMENT MCC_WORK_DIR\n'
    )
    environment = {
        **os.environ, "MCC_PROJECT_ROOT": str(tmp_path / "checkout"),
        "MCC_STORAGE_ROOT": str(tmp_path / "storage"), "SLURM_JOB_ID": "12345",
    }
    for variant in ("reference", "replica", "one_epoch", "low_noise"):
        result = subprocess.run(["bash", str(launcher), variant], env=environment,
                                text=True, capture_output=True, check=True)
        experiment, work = result.stdout.splitlines()
        assert experiment.endswith(f"bede_v100_ppo_200_credit_{variant}.yaml")
        assert work == str(tmp_path / f"storage/training/work/credit-{variant}-12345")
    invalid = subprocess.run(["bash", str(launcher), "typo"], env=environment, capture_output=True)
    assert invalid.returncode == 2
    with (tmp_path / "storage/training/.credit-reference.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        duplicate = subprocess.run(["bash", str(launcher), "reference"], env=environment,
                                   text=True, capture_output=True)
        assert duplicate.returncode == 1
        assert "already running" in duplicate.stderr


def test_scale_override_changes_only_parallel_shape():
    experiment = {"run": {"name": "test"}, "ppo": {"rollout_steps": 1000}}
    machine = {"launcher": {"type": "slurm", "gpu_resource": "gpu:1"}}
    scale = {
        "parallelism": {
            "physical_environments": 4,
            "simulations_per_gpu": 1,
        },
        "resources": {"nodes": 4, "gpus_per_node": 1},
        "smartsim": {"interface": "ib0"},
    }
    config = merge(merge(experiment, machine), scale)
    plan = parallel_plan(config)
    assert plan.physical_environments == 4
    assert plan.required_gpus == 4
    assert plan.nodes == 4
    assert config["ppo"]["rollout_steps"] == 1000


def test_multi_node_loopback_is_rejected():
    config = {
        "parallelism": {"physical_environments": 2},
        "resources": {"nodes": 2, "gpus_per_node": 1},
        "launcher": {"type": "slurm"},
        "smartsim": {"interface": "lo"},
    }
    with pytest.raises(ValueError, match="loopback"):
        parallel_plan(config)


def test_multi_node_placeholder_interface_is_rejected():
    config = {
        "parallelism": {"physical_environments": 2},
        "resources": {"nodes": 2, "gpus_per_node": 1},
        "launcher": {"type": "slurm"},
        "smartsim": {"interface": "CHANGE_ME"},
    }
    with pytest.raises(ValueError, match="routable"):
        parallel_plan(config)


def test_bede_mpirun_multi_node_loopback_is_rejected():
    config = {
        "parallelism": {"physical_environments": 2},
        "resources": {"nodes": 2, "gpus_per_node": 1},
        "launcher": {"type": "bede_mpirun"},
        "smartsim": {"interface": "lo"},
    }
    with pytest.raises(ValueError, match="loopback"):
        parallel_plan(config)


def test_bede_mpirun_rejects_multiple_gpus_per_node():
    config = {
        "parallelism": {"physical_environments": 2},
        "resources": {"nodes": 1, "gpus_per_node": 2},
        "launcher": {"type": "bede_mpirun"},
        "smartsim": {"interface": "ib0"},
    }
    with pytest.raises(ValueError, match="gpus_per_node=1"):
        parallel_plan(config)


def test_csf3_launcher_remains_local():
    config = compose(
        PROJECT_ROOT / "configs/experiments/first_wall_ppo.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/csf3_four_gpu.yaml",
        PROJECT_ROOT / "configs/scales/four_gpu.yaml",
    )
    plan = parallel_plan(config)

    assert config["launcher"]["type"] == "local"
    assert plan.physical_environments == 4
    assert plan.required_gpus == 4


def test_scale_parallelism_overrides_machine_default(monkeypatch):
    monkeypatch.setenv("RL_N_ENVS", "1")
    monkeypatch.setenv("RL_SIMULATIONS_PER_GPU", "3")
    config = {
        "launcher": {
            "simulations_per_gpu": 3,
            "mpi_command": "/opt/bin/bede-mpirun",
        },
        "parallelism": {
            "physical_environments": 4,
            "simulations_per_gpu": 1,
        },
    }
    export_compatibility_environment(config)
    assert os.environ["RL_N_ENVS"] == "4"
    assert os.environ["RL_SIMULATIONS_PER_GPU"] == "1"
    assert os.environ["MCC_BEDE_MPIRUN"] == "/opt/bin/bede-mpirun"


def test_insufficient_gpu_allocation_is_rejected():
    config = {
        "parallelism": {
            "physical_environments": 4,
            "simulations_per_gpu": 1,
        },
        "resources": {"nodes": 2, "gpus_per_node": 1},
        "launcher": {"type": "local"},
    }
    with pytest.raises(ValueError, match="needs 4 GPUs"):
        parallel_plan(config)


def test_submission_command_carries_scale_to_batch_controller(tmp_path):
    experiment = tmp_path / "experiment.yaml"
    machine = tmp_path / "machine.yaml"
    scale = tmp_path / "scale.yaml"
    for path in (experiment, machine, scale):
        path.write_text("test: true\n")
    config = {
        "parallelism": {
            "physical_environments": 2,
            "simulations_per_gpu": 1,
        },
        "resources": {
            "nodes": 2,
            "gpus_per_node": 1,
            "gpu_resource_per_node": "gpu:1",
        },
        "launcher": {"type": "slurm", "gpu_resource": "gpu:1"},
        "smartsim": {"interface": "ib0"},
        "scheduler": {
            "submit_command": "ghbatch",
            "account": "project",
            "partition": "gh",
        },
    }
    command = build_submission_command(
        config, PROJECT_ROOT, experiment, machine, scale
    )
    assert command[:5] == [
        "ghbatch",
        "--nodes=2",
        "--gres=gpu:1",
        "--account=project",
        "--partition=gh",
    ]
    assert command[-6:] == [
        "--experiment",
        str(experiment),
        "--machine",
        str(machine),
        "--scale",
        str(scale),
    ]


def test_bede_stack_smoke_is_one_gpu_one_update():
    config = compose(
        PROJECT_ROOT / "configs/experiments/stack_smoke.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_stack_smoke.yaml",
        PROJECT_ROOT / "configs/scales/one_gpu.yaml",
    )
    plan = parallel_plan(config)

    assert plan.physical_environments == 1
    assert plan.required_gpus == 1
    assert plan.nodes == 1
    assert config["launcher"]["type"] == "local"
    assert config["smartsim"]["interface"] == "lo"
    assert config["ppo"] == {
        "rollout_steps": 2,
        "batch_size": 20,
        "epochs": 1,
        "learning_rate": 1.0e-4,
    }
    assert config["training"]["total_timesteps"] == 20


def test_bede_power9_stack_smoke_is_one_gpu_one_update():
    config = compose(
        PROJECT_ROOT / "configs/experiments/stack_smoke.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_power9_stack_smoke.yaml",
        PROJECT_ROOT / "configs/scales/one_gpu.yaml",
    )
    plan = parallel_plan(config)

    assert plan.physical_environments == 1
    assert plan.required_gpus == 1
    assert plan.nodes == 1
    assert config["launcher"]["type"] == "local"
    assert config["launcher"]["cpus_per_simulation"] == 4
    assert config["smartsim"]["interface"] == "lo"
    assert config["runtime"]["startup_timeout"] == 1200
    assert config["training"]["total_timesteps"] == 20


def test_bede_terminal_pilot_disables_horizon_bootstrap(monkeypatch):
    config = compose(
        PROJECT_ROOT
        / "configs/experiments/bede_v100_ppo_200_terminal_pilot.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_power9_four_gpu.yaml",
        PROJECT_ROOT / "configs/scales/bede_power9_four_gpu.yaml",
    )
    export_compatibility_environment(config)

    assert config["training"]["episode_steps"] == 200
    assert config["training"]["episode_horizon_truncated"] is False
    assert config["training"]["total_timesteps"] == 80000
    assert os.environ["RL_EPISODE_HORIZON_TRUNCATED"] == "False"


def test_bede_power9_two_gpu_stack_smoke_is_multi_node_one_update(monkeypatch):
    monkeypatch.setenv("SMARTSIM_INTERFACE", "fabric0")
    monkeypatch.setenv("MCC_BEDE_MPIRUN", "/opt/bede/bin/bede-mpirun")
    config = compose(
        PROJECT_ROOT / "configs/experiments/stack_smoke_two_gpu.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_power9_two_gpu_stack_smoke.yaml",
        PROJECT_ROOT / "configs/scales/two_gpu.yaml",
    )
    plan = parallel_plan(config)

    assert plan.physical_environments == 2
    assert plan.required_gpus == 2
    assert plan.nodes == 2
    assert plan.gpus_per_node == 1
    assert config["launcher"]["type"] == "bede_mpirun"
    assert "mpi_command" not in config["launcher"]
    export_compatibility_environment(config)
    assert os.environ["MCC_BEDE_MPIRUN"] == "/opt/bede/bin/bede-mpirun"
    assert "smartsim" not in config
    assert config["runtime"]["startup_timeout"] == 1200
    assert config["training"]["total_timesteps"] == 40


def test_bede_two_gpu_stack_smoke_is_multi_node_one_update(monkeypatch):
    monkeypatch.setenv("SMARTSIM_INTERFACE", "fabric0")
    monkeypatch.setenv("MCC_BEDE_MPIRUN", "/opt/bede/bin/bede-mpirun")
    config = compose(
        PROJECT_ROOT / "configs/experiments/stack_smoke_two_gpu.yaml",
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/machines/bede_two_gpu_stack_smoke.yaml",
        PROJECT_ROOT / "configs/scales/two_gpu.yaml",
    )
    plan = parallel_plan(config)

    assert plan.physical_environments == 2
    assert plan.required_gpus == 2
    assert plan.nodes == 2
    assert config["launcher"]["type"] == "bede_mpirun"
    assert "mpi_command" not in config["launcher"]
    export_compatibility_environment(config)
    assert os.environ["MCC_BEDE_MPIRUN"] == "/opt/bede/bin/bede-mpirun"
    assert "smartsim" not in config
    assert config["ppo"] == {
        "rollout_steps": 2,
        "batch_size": 40,
        "epochs": 1,
        "learning_rate": 1.0e-4,
    }
    assert config["training"]["total_timesteps"] == 40
