"""Tests for Bede's MPI-based remote worker adapter."""

import signal
import subprocess
from pathlib import Path

import pytest

from magnetoconvective_control.launchers.bede_mpirun import (
    build_launch_command,
    discover_allocation_hosts,
    parse_allocation_hosts,
    supervise_launch,
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
            "MCC_BEDE_MPI_PARENT": "1",
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
        "-x",
        "MCC_BEDE_MPI_PARENT",
        "/bin/bash",
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
                "gh005\n"
                "[1,1]<stdout>:gh006.bede.dur.ac.uk\n"
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
    assert calls[0][0][-2:] == ["/bin/hostname", "-s"]
    assert calls[0][1]["timeout"] == 60


def test_discover_allocation_hosts_reports_probe_failure():
    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="allocation failed")

    with pytest.raises(RuntimeError, match="allocation failed"):
        discover_allocation_hosts("bede-mpirun", runner=runner)


def test_supervise_launch_preserves_unexpected_worker_failure():
    class FailedProcess:
        def poll(self):
            return 7

        def wait(self):
            return 7

    calls = []

    def popen(command, env):
        calls.append((command, env))
        return FailedProcess()

    def set_signal_handler(_signum, _handler):
        return signal.SIG_DFL

    assert supervise_launch(
        ["bede-mpirun", "worker"],
        {"PATH": "/usr/bin"},
        popen=popen,
        set_signal_handler=set_signal_handler,
    ) == 7
    assert calls == [(["bede-mpirun", "worker"], {"PATH": "/usr/bin"})]


def test_supervise_launch_treats_requested_shutdown_as_success():
    handlers = {}

    class RunningProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 143

    process = RunningProcess()

    def popen(_command, env):
        assert env["PATH"] == "/usr/bin"
        return process

    def set_signal_handler(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    assert supervise_launch(
        ["bede-mpirun", "worker"],
        {"PATH": "/usr/bin"},
        popen=popen,
        set_signal_handler=set_signal_handler,
    ) == 0
    assert process.terminated
