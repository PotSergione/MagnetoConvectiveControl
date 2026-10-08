# MagnetoConvectiveControl

Research software for feedback control of magnetoconvective liquid-metal
flows using NekRS, SmartSim and recurrent reinforcement learning.

The repository separates reusable control software, replaceable physical
cases, experiment configurations and external run artifacts. The initial case
is the toroidal blanket configuration at `Gr=1e9`, `Pr=0.0321` and `Ha=1750`.

## Repository scope

Tracked here:

- RL and SmartSim integration code;
- self-contained NekRS case definitions;
- versioned training and evaluation configurations;
- tests, analysis code and compact paper-facing results;
- manifests describing external snapshots and checkpoints.

Not tracked here:

- SmartSim work directories and telemetry;
- NekRS restart fields;
- training checkpoints, TensorBoard data and raw logs;
- machine-local build caches.

## Quick start

```bash
conda env create -f environment.yml
conda activate magnetoconvective-control
mcc-train \
  --experiment configs/experiments/first_wall_ppo.yaml \
  --machine configs/machines/local.yaml \
  --dry-run
```

A real run additionally requires compatible NekRS, MPI, SmartSim and
SmartRedis installations plus the snapshot dataset described under `data/`.

See `docs/running_locally.md`, `docs/running_on_slurm.md`, and
`docs/data_management.md` before launching an experiment.

## Status

This is a research codebase being prepared in support of a paper. Public
licensing and citation metadata must be finalized before the first public
GitHub release.
