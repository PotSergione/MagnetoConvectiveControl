# Running locally

Install the Python package in an environment compatible with the local NekRS,
MPI, SmartSim and SmartRedis installations. Set:

```bash
export NEKRS_HOME=/path/to/nekrs
export RL_SNAPSHOTS_DIR=/path/to/snapshots
export MCC_ARTIFACT_ROOT=/path/to/runs
```

Validate configuration without starting NekRS:

```bash
mcc-train --experiment configs/experiments/first_wall_ppo.yaml \
  --machine configs/machines/local.yaml \
  --scale configs/scales/one_gpu.yaml \
  --dry-run
```

Remove `--dry-run` only after confirming the resolved paths and run directory.
