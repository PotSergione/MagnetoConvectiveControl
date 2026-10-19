"""Launch one generated NekRS case on a selected Bede GPU node.

SmartSim supervises this adapter with its local launcher.  The adapter replaces
itself with ``bede-mpirun`` so process status, signals, and output continue to
flow through SmartSim while Bede performs the actual inter-node placement.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
# Values required by the NekRS wrapper or SmartRedis client on the remote node.
# ``-x NAME`` asks Open MPI to copy the value from the adapter environment;
# absent optional values are deliberately omitted.
REMOTE_ENVIRONMENT = (
    "SSDB",
    "SR_DB_TYPE",
    "SSKEYIN",
    "SSKEYOUT",
    "OCCA_CACHE_DIR",
    "NEKRS_CACHE_DIR",
    "NEKRS_HOME",
    "NEKRS_UDF_INCLUDES",
    "NEKRS_UDF_LIBS",
    "BLANKET_CMAKE",
    "PATH",
    "MCC_BEDE_MPI_PARENT",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "PYTHONNOUSERSITE",
    "SR_LOG_FILE",
    "SR_LOG_LEVEL",
    "CC",
    "CXX",
    "FC",
)


def parse_allocation_hosts(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return validated, ordered, unique host names from config or an env var."""
    if isinstance(value, str):
        candidates = re.split(r"[\s,]+", value.strip())
    else:
        candidates = [str(item).strip() for item in value]

    hosts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if _HOST_PATTERN.fullmatch(candidate) is None:
            raise ValueError(f"Invalid Bede allocation host name: {candidate!r}")
        if candidate not in seen:
            seen.add(candidate)
            hosts.append(candidate)
    if not hosts:
        raise ValueError(
            "The bede_mpirun launcher requires at least one allocated node"
        )
    return tuple(hosts)


def discover_allocation_hosts(
    mpi_command: str,
    *,
    runner=subprocess.run,
) -> tuple[str, ...]:
    """Discover one short host name per allocated node through Bede MPI."""
    command = shlex.split(mpi_command)
    if not command:
        raise ValueError("Bede MPI launch command cannot be empty")
    probe = [
        *command,
        "--bede-par",
        "1ppn:none",
        "/bin/hostname",
        "-s",
    ]
    try:
        result = runner(
            probe,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(
            f"Could not discover Bede allocation hosts with {mpi_command!r}: "
            f"{detail.strip()}"
        ) from error

    discovered = []
    for line in result.stdout.splitlines():
        # Open MPI may label rank output as ``[job,rank]<stdout>:HOST``.
        # Direct hostname execution avoids shell-quoting differences between
        # Bede's aarch64 and ppc64le wrappers.
        candidate = line.rpartition("<stdout>:")[2].strip()
        if not candidate:
            candidate = line.strip()
        if _HOST_PATTERN.fullmatch(candidate) and any(
            character.isdigit() for character in candidate
        ):
            discovered.append(candidate.partition(".")[0])
    try:
        return parse_allocation_hosts(discovered)
    except ValueError as error:
        raise RuntimeError(
            "bede-mpirun did not report any allocated hosts; probe output was: "
            f"{result.stdout!r}"
        ) from error


def build_launch_command(
    mpi_command: str,
    host: str,
    executable: str | Path,
    working_directory: str | Path,
    environment: Mapping[str, str],
) -> list[str]:
    """Build the argv used for one single-rank, one-GPU Bede worker."""
    command = shlex.split(mpi_command)
    if not command:
        raise ValueError("Bede MPI launch command cannot be empty")
    validated_host = parse_allocation_hosts([host])[0]
    work = Path(working_directory).expanduser().resolve()
    program = Path(executable)
    if not program.is_absolute():
        program = work / program
    program = program.resolve()

    argv = [
        *command,
        "--bede-par",
        "1ppg:none",
        "--host",
        validated_host,
        "-np",
        "1",
        "--wdir",
        str(work),
    ]
    for name in REMOTE_ENVIRONMENT:
        if name in environment and environment[name]:
            argv.extend(["-x", name])
    # SmartSim copies generator files without preserving the executable bit.
    # Invoking the generated case through Bash therefore works on every node
    # without mutating the generated directory after placement.
    argv.extend(["/bin/bash", str(program)])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", help="Generated case executable")
    args = parser.parse_args()

    try:
        host = os.environ["MCC_BEDE_TARGET_HOST"]
    except KeyError as error:
        raise RuntimeError("MCC_BEDE_TARGET_HOST is not set") from error
    mpi_command = os.environ.get("MCC_BEDE_MPIRUN", "bede-mpirun")
    command = build_launch_command(
        mpi_command=mpi_command,
        host=host,
        executable=args.executable,
        working_directory=Path.cwd(),
        environment=os.environ,
    )
    print(
        f"Launching Bede worker on {host}: {shlex.join(command)}",
        flush=True,
    )
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()
