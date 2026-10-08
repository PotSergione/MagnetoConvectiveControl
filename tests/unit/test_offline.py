"""Dependency-free contract tests; these never launch SmartSim or NekRS."""

import ast
from dataclasses import dataclass
from math import exp
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
ENV_PATH = PROJECT_ROOT / "src" / "magnetoconvective_control" / "envs" / "smartsim_env.py"
CASE_PATH = PROJECT_ROOT / "cases" / "toroidal_gr1e9_ha1750"


def _load_selected(*names):
    tree = ast.parse(ENV_PATH.read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = {"np": np, "dataclass": dataclass, "exp": exp}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


def test_wall_reconstruction_preserves_linear_field():
    _, _, build = _load_selected(
        "_WallReconstructionPlan", "_cluster_axis", "_build_wall_plan"
    )
    transverse, z, values = [], [], []
    for zz in np.linspace(0.0, 1.0, 5):
        for xx in np.linspace(-1.0, 1.0, 4):
            for _ in range(2):  # duplicate SEM boundary nodes
                transverse.append(xx)
                z.append(zz)
                values.append(2.0 * xx + 3.0 * zz)
    plan = build(np.array(transverse), np.array(z), (8, 10))
    reconstructed = plan.reconstruct(np.array(values))
    expected = 2.0 * plan.transverse_coordinates[:, None] + 3.0 * plan.z_coordinates[None, :]
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)


def test_first_wall_heat_flux_reward_level_and_action_costs():
    tree = ast.parse(ENV_PATH.read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SmartSimVecEnv")
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_first_wall_heat_flux_reward")
    namespace = {"np": np}
    exec(compile(ast.Module([method], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    class Dummy:
        pass
    Dummy._first_wall_heat_flux_reward = namespace["_first_wall_heat_flux_reward"]
    env = Dummy()
    env.ewma_alpha = 1.0 - np.exp(-0.1)
    env.first_wall_heat_flux_baseline = 0.018
    env.reward_scale = 1.0
    env.action_magnitude_regularization = 0.0
    env.action_slew_regularization = 0.0
    env._first_wall_flux_hat = np.array([0.018])
    env._last_actions = [np.zeros(10)]
    env._last_reward_components = [{}]
    assert env._first_wall_heat_flux_reward(0, 0.018, 0.054, np.zeros(10)) == 0.0
    reward = env._first_wall_heat_flux_reward(0, 0.012, 0.054, np.zeros(10))
    expected = env.ewma_alpha * (0.018 - 0.012) / 0.054
    np.testing.assert_allclose(reward, expected)
    assert env._first_wall_heat_flux_reward(0, 0.012, 0.054, np.zeros(10)) > reward


def test_observation_standardizer_is_self_contained():
    tree = ast.parse(ENV_PATH.read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SmartSimVecEnv")
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_observation")
    namespace = {"np": np}
    exec(compile(ast.Module([method], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    values = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    scaled = namespace["_prepare_observation"](object(), values)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled.mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(scaled.std(), 1.0, atol=1e-7)


def test_static_solver_python_contract():
    env = ENV_PATH.read_text()
    udf = (CASE_PATH / "blanket.udf").read_text()
    par = (CASE_PATH / "blanket.par").read_text()
    for field in ("yMinus", "yPlus", "xPlus"):
        for suffix in ("HeatFlux", "Transverse", "Z"):
            assert suffix in env
            assert suffix in udf
    assert 'N_AGENTS = 10' in env
    assert 'axis=2' in env
    assert 'redisPublishInterval = 0.1' in par
    assert 'randomAction = false' in par
    assert 'xMinusIntegratedHeatFlux' in env
    assert 'xMinusIntegratedHeatFlux' in udf
    assert 'totalHeatingPower' in env
    assert 'totalHeatingPower' in udf
    assert 'MPI_Allgatherv' in udf
    assert 'MPI_Bcast' in udf
    publish = udf[udf.index("static void publishToRedis"):udf.index("static void fetchNextAction")]
    assert publish.index("if (redisRank != 0) return;") < publish.index("ensureRedisClient();")
    assert publish.index("ensureRedisClient();") < publish.index("redisClient->put_tensor")
    runner = (CASE_PATH / "run_case.sh").read_text()
    assert "BLANKET_CMAKE" in runner


def test_ten_agent_cyclic_views_and_action_alignment():
    tree = ast.parse(ENV_PATH.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SB3MARLSmartSimVecEnv")
    method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "_agent_observations")
    namespace = {"np": np}
    exec(compile(ast.Module([method], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    class Dummy:
        pass
    env = Dummy()
    env.N_AGENTS = 10
    env.ACTION_NORMALIZATION = 0.25
    env._heat_flux_dtype = np.float32
    env._action_observation_dtype = np.float32
    env._agent_shift = 2
    env.core = Dummy()
    env.core.n_envs = 1
    env.core.wall_shape = (3, 20)
    heat_flux = np.arange(3 * 3 * 20).reshape(1, 3, 3, 20)
    actions = np.arange(10, dtype=float)[None] / 40.0
    observed = namespace["_agent_observations"](env, heat_flux, actions)
    assert observed["heat_flux"].shape == (10, 3, 3, 20)
    for agent in range(10):
        np.testing.assert_array_equal(
            observed["heat_flux"][agent], np.roll(heat_flux[0], -2 * agent, axis=2)
        )
        np.testing.assert_allclose(
            observed["previous_actions"][agent], np.roll(actions[0], -agent) / 0.25
        )


def test_restart_name_is_canonicalized_in_generated_case():
    tree = ast.parse(ENV_PATH.read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SmartSimVecEnv")
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_set_model_restart_file")
    method.decorator_list = []
    namespace = {"Path": Path, "re": __import__("re")}
    exec(compile(ast.Module([method], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "blanketZik0.f00260").write_bytes(b"restart")
        (root / "blanket.par").write_text("[GENERAL]\n# startFrom = old0.f00001\n")
        model = type("Model", (), {"path": str(root)})()
        namespace["_set_model_restart_file"](model, "blanketZik0.f00260")
        assert (root / "blanket0.f00260").read_bytes() == b"restart"
        assert "startFrom = blanket0.f00260" in (root / "blanket.par").read_text()
