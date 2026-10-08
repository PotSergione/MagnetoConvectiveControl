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

nrsmpi blanket 1 2>&1 | tee blanket.log
