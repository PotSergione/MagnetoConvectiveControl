from pathlib import Path

import numpy as np

from magnetoconvective_control.feasibility import (
    _actions_for_identification,
    _evaluation_layout,
    _exact_sign_test_p,
    _identification_layout,
    _training_snapshot_groups,
    _wall_modal_projection,
)


def _snapshots() -> list[Path]:
    return [Path(f"blanket0.f{250 + 10 * index:05d}") for index in range(5)]


def test_training_omits_each_snapshot_once_and_uses_every_snapshot_four_times():
    snapshots = _snapshots()
    groups = _training_snapshot_groups(snapshots)
    assert len(groups) == 5
    assert all(len(group) == 4 for group in groups)
    assert [set(snapshots) - set(group) for group in groups] == [
        {snapshot} for snapshot in snapshots
    ]
    assert {snapshot: sum(snapshot in group for group in groups) for snapshot in snapshots} == {
        snapshot: 4 for snapshot in snapshots
    }


def test_identification_and_evaluation_cover_five_fields_as_declared():
    snapshots = _snapshots()
    identification = _identification_layout(snapshots)
    assert list(map(len, identification)) == [4, 4]
    assert set(snapshots).issubset({item for block in identification for item in block})

    evaluation = _evaluation_layout(snapshots)
    assert list(map(len, evaluation)) == [4, 4, 4]
    primary = [item for block in evaluation for item in block if item[1] in {"learned", "reference"}]
    for snapshot in snapshots:
        assert primary.count((snapshot, "learned")) == 1
        assert primary.count((snapshot, "reference")) == 1


def test_identification_actions_have_frozen_segments_and_safe_bounds():
    config = {
        "campaign": {
            "seed": 5489, "transitions": 1000,
            "safe_action_range": [-0.03, 0.03],
        },
        "reference": {"action": 0.01},
        "identification": {
            "warmup_steps": 100, "excitation_steps": 500,
            "target_perturbation_rms": 0.005,
        },
    }
    actions = _actions_for_identification(config, 0, [False] * 4)
    assert actions.shape == (4, 1000, 10)
    np.testing.assert_allclose(actions[:, :100], 0.01)
    np.testing.assert_allclose(actions[:, 600:], 0.01)
    assert np.max(actions) <= 0.03 + 1e-12
    assert np.min(actions) >= -0.03 - 1e-12
    assert not np.allclose(actions[:, 100:600], 0.01)


def test_wall_projection_is_compact_and_constant_maps_only_have_uniform_mode():
    maps = np.stack([np.full((128, 160), value) for value in (1.0, 2.0, 3.0)])
    projection = _wall_modal_projection(maps)
    assert projection.shape == (30,)
    np.testing.assert_allclose(projection[[0, 10, 20]], [1.0, 2.0, 3.0])
    mask = np.ones(30, dtype=bool)
    mask[[0, 10, 20]] = False
    np.testing.assert_allclose(projection[mask], 0.0, atol=1e-12)


def test_five_pair_sign_test_cannot_reach_two_sided_five_percent():
    assert _exact_sign_test_p(5, 5) == 0.0625
