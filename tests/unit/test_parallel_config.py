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


def test_bede_power9_two_gpu_stack_smoke_is_multi_node_one_update(monkeypatch):
    monkeypatch.setenv("SMARTSIM_INTERFACE", "fabric0")
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
    assert config["launcher"]["mpi_command"] == "bede-mpirun"
    assert "smartsim" not in config
    assert config["runtime"]["startup_timeout"] == 1200
    assert config["training"]["total_timesteps"] == 40


def test_bede_two_gpu_stack_smoke_is_multi_node_one_update(monkeypatch):
    monkeypatch.setenv("SMARTSIM_INTERFACE", "fabric0")
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
    assert config["launcher"]["mpi_command"] == "bede-mpirun"
    assert "smartsim" not in config
    assert config["ppo"] == {
        "rollout_steps": 2,
        "batch_size": 40,
        "epochs": 1,
        "learning_rate": 1.0e-4,
    }
    assert config["training"]["total_timesteps"] == 40
