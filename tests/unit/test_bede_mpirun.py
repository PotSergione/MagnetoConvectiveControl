"""Tests for Bede's MPI-based remote worker adapter."""

import subprocess
from pathlib import Path

import pytest

from magnetoconvective_control.launchers.bede_mpirun import (
    build_launch_command,
    discover_allocation_hosts,
    parse_allocation_hosts,
)


def test_parse_allocation_hosts_preserves_order_and_removes_duplicates():
    assert parse_allocation_hosts("gh006,gh005 gh006") == ("gh006", "gh005")


@pytest.mark.parametrize("value", ["", "gh005;hostname", "/tmp/node"])
def test_parse_allocation_hosts_rejects_empty_or_unsafe_values(value):
    with pytest.raises(ValueError):
        parse_allocation_hosts(value)


def test_build_launch_command_targets_one_gpu_rank_and_exports_present_values(
    tmp_path: Path,
):
    command = build_launch_command(
        mpi_command="/opt/bede/bin/bede-mpirun --verbose",
        host="gh006.bede.dur.ac.uk",
        executable="run_case.sh",
        working_directory=tmp_path,
        environment={
            "SSDB": "10.0.0.1:12345",
            "SR_DB_TYPE": "STANDALONE",
            "SSKEYIN": "nekrs_env_1_ep_0",
            "PATH": "/usr/bin",
            "NEKRS_CACHE_DIR": "",  # Empty optional values are not exported.
        },
    )

    assert command[:11] == [
        "/opt/bede/bin/bede-mpirun",
        "--verbose",
        "--bede-par",
        "1ppg:none",
        "--host",
        "gh006.bede.dur.ac.uk",
        "-np",
        "1",
        "--wdir",
        str(tmp_path.resolve()),
        "-x",
    ]
    assert command[11:] == [
        "SSDB",
        "-x",
        "SR_DB_TYPE",
        "-x",
        "SSKEYIN",
        "-x",
        "PATH",
        str((tmp_path / "run_case.sh").resolve()),
    ]


def test_discover_allocation_hosts_accepts_mpi_labeled_output():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "[1,0]<stdout>:__MCC_HOST__gh005\n"
                "[1,1]<stdout>:__MCC_HOST__gh006\n"
            ),
            stderr="",
        )

    assert discover_allocation_hosts("/opt/bin/bede-mpirun", runner=runner) == (
        "gh005",
        "gh006",
    )
    assert calls[0][0][:3] == [
        "/opt/bin/bede-mpirun",
        "--bede-par",
        "1ppn:none",
    ]
    assert calls[0][1]["timeout"] == 60


def test_discover_allocation_hosts_reports_probe_failure():
    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="allocation failed")

    with pytest.raises(RuntimeError, match="allocation failed"):
        discover_allocation_hosts("bede-mpirun", runner=runner)
