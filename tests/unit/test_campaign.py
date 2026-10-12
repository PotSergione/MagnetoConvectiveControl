import gzip
from pathlib import Path

import numpy as np

from magnetoconvective_control.campaign import (
    TRANSITION_FIELDS,
    TrajectoryWriter,
    _arm_mapping,
    _multisine_sequences,
    electrode_to_modal,
    modal_to_electrode,
    real_cyclic_fourier_basis,
)


def test_real_fourier_basis_is_rms_orthonormal_and_invertible():
    basis = real_cyclic_fourier_basis()
    np.testing.assert_allclose(basis @ basis.T / 10.0, np.eye(10), atol=1e-12)
    coefficients = np.linspace(-0.01, 0.01, 10)
    np.testing.assert_allclose(
        electrode_to_modal(modal_to_electrode(coefficients)), coefficients,
        atol=1e-12,
    )


def test_multisine_is_deterministic_orthogonal_and_bounded():
    first = _multisine_sequences(5489, 10, 500)
    second = _multisine_sequences(5489, 10, 500)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first / 500.0, np.eye(10), atol=1e-12)
    assert np.max(np.abs(first)) < 3.0


def test_arm_assignment_rotates_over_all_gpus():
    assignments = np.asarray([_arm_mapping(block) for block in range(4)])
    for arm in range(4):
        gpu_positions = np.argmax(assignments == arm, axis=1)
        assert sorted(gpu_positions.tolist()) == [0, 1, 2, 3]


def test_trajectory_writer_commits_table_and_compressed_chunks(tmp_path: Path):
    writer = TrajectoryWriter(tmp_path / "trajectory", chunk_steps=2)
    row = {field: 0 for field in TRANSITION_FIELDS}
    maps = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    for _ in range(3):
        writer.append(row, maps)
    writer.complete()
    directory = tmp_path / "trajectory"
    assert (directory / "COMPLETE").is_file()
    assert len((directory / "trajectory.csv").read_text().splitlines()) == 4
    chunks = sorted(directory.glob("wall_maps_*.npy.gz"))
    assert len(chunks) == 2
    with gzip.open(chunks[0], "rb") as stream:
        values = np.load(stream, allow_pickle=False)
    assert values.shape == (2, 3, 4, 5)
