#!/usr/bin/env bash

set -euo pipefail

: "${NEKRS_HOME:?Set NEKRS_HOME to the NekRS installation prefix}"
export NEKRS_HOME

# Keep the same local CMake-selection boilerplate used by the MARL case.
local_tools="${PWD}/.local-tools"
mkdir -p "${local_tools}"
cmake_executable="${BLANKET_CMAKE:-$(command -v cmake)}"
ln -sfn "${cmake_executable}" "${local_tools}/cmake"
export PATH="${local_tools}:${NEKRS_HOME}/bin:${PATH}"

# This is a headless worker; stale forwarded-display credentials only produce
# noisy X11 authorization diagnostics in locally launched MPI children.
unset DISPLAY XAUTHORITY

# The Bede multi-node adapter launches this script as an MPI rank.  Replace
# that shell process with NekRS directly so there is no nested mpirun.  All
# other launchers retain the established nrsmpi path below.
if [[ "${MCC_BEDE_MPI_PARENT:-0}" == "1" ]]; then
    exec > >(tee blanket.log) 2>&1
    exec "${NEKRS_HOME}/bin/nekrs" --setup blanket
fi

nrsmpi blanket 1 2>&1 | tee blanket.log
