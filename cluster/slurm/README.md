# Slurm execution

`train.slurm` deliberately contains only resource allocation and invocation of
the portable `mcc-train` command. Module loading and native-library paths are
site-specific and should be supplied by a private wrapper or shell environment.

The preferred interface is `mcc-submit`, which converts a scale profile into
Slurm resource arguments and then invokes `train.slurm`:

```bash
export NEKRS_HOME=/cluster/path/to/nekrs
mcc-submit \
  --experiment configs/experiments/first_wall_ppo.yaml \
  --machine configs/machines/cluster.local.yaml \
  --scale configs/scales/one_gpu.yaml
```

The autonomous four-GPU design campaign uses one 48-hour `gpuA` allocation:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nekrs-smartsim
python -m pip install 'PyYAML>=6'

export MCC_DESIGN_SNAPSHOTS_DIR="$HOME/scratch/MagnetoconvectiveControl/design_snapshots_v1"
export MCC_ARTIFACT_ROOT="$HOME/scratch/MagnetoconvectiveControl/campaigns"
sbatch cluster/slurm/four_gpu_campaign.slurm
```

The snapshot directory must resolve to exactly eight non-confirmatory
`blanket0.fNNNNN` files. Completed four-arm blocks and PPO rollouts are atomic;
submitting the same command again resumes the immutable campaign plan. The
launcher defaults to the CSF3 `nekrs-smartsim` Conda environment, sources
`$HOME/.local/nekMHD/activate.sh`, and uses the native SmartRedis client under
`$HOME/.local/smartredis-0.6.1`.

`train.slurm` runs the Python controller directly in the batch shell. Do not
wrap it in an outer `srun`: SmartSim owns the database and simulation job steps.

The Bede GH-specific one- and two-GPU stack checks are
`bede_stack_smoke.slurm` and `bede_two_gpu_stack_smoke.slurm`. The latter uses
two GH nodes and discovers its SmartSim interface from the kernel route between
them; it does not assume a site-specific network-device name. It uses
`bede-mpirun` for both allocation discovery and one-worker-per-node placement,
without requiring Slurm client commands on the aarch64 nodes. The original
POWER9/V100 nodes have their own one-GPU launcher,
`bede_power9_stack_smoke.slurm`, plus the genuine two-node integration launcher
`bede_power9_two_gpu_stack_smoke.slurm`. Both use isolated ppc64le installation
and cache defaults. See `docs/running_on_slurm.md` for submission and dependency
examples.

`bede_power9_v100_timing.slurm` and
`bede_power9_four_v100_timing.slurm` measure 100 physical transitions with one
and four local V100 workers. The production counterpart is
`bede_power9_four_gpu_campaign.slurm`; it resumes completed atomic blocks and
exits successfully when the next block would violate the allocation reserve.

The Bede POWER9/V100 PPO training run with 200-step episodes is submitted with:

```bash
export MCC_STORAGE_ROOT=/nobackup/projects/bdman39/sergione
sbatch -A bdman39 cluster/slurm/bede_power9_four_gpu_train.slurm
```

It uses `training.episode_steps: 200` and `ppo.rollout_steps: 200`, aligning
each PPO rollout with one complete episode from each of the four physical
environments. With ten virtual agents, every rollout contains 8,000 samples.
It uses all four local V100s and writes checkpoints under
`${MCC_STORAGE_ROOT}/training`.

### Credit-assignment campaign

Use `bede_power9_four_gpu_credit.slurm` with one variant argument:

| Variant | Model/snapshot seed | Epochs | Initial action sigma |
|---|---:|---:|---:|
| `reference` (A) | 5489 | 5 | exp(-2.3), approximately 0.10 |
| `replica` (B) | 5490 | 5 | approximately 0.10 |
| `one_epoch` (C) | 5489 | 1 | approximately 0.10 |
| `low_noise` (optional D) | 5489 | 5 | 0.03 |

All use gamma 0.997, GAE lambda 1, batch size 1000, learning rate 1e-4,
200-step terminal episodes and 200-step rollouts. The control interval remains
0.1. A 150,000-sample request completes 19 rollouts (152,000 samples), with
one metrics report per rollout. Five epochs give 40 optimizer steps per
rollout; one epoch gives eight.

From the Bede checkout after pulling the changes:

```bash
export MCC_STORAGE_ROOT=/nobackup/projects/bdman39/sergione
sbatch -A bdman39 --job-name=mcc-credit-a cluster/slurm/bede_power9_four_gpu_credit.slurm reference
sbatch -A bdman39 --job-name=mcc-credit-c cluster/slurm/bede_power9_four_gpu_credit.slurm one_epoch
sbatch -A bdman39 --job-name=mcc-credit-b cluster/slurm/bede_power9_four_gpu_credit.slurm replica
# Optional exploration-amplitude comparison:
sbatch -A bdman39 --job-name=mcc-credit-d cluster/slurm/bede_power9_four_gpu_credit.slurm low_noise
```

Each requests one four-V100 node for 43 hours and delegates setup to the
existing four-GPU training launcher. New run names are
`bede_v100_ppo_200_credit_VARIANT`; the versioned experiment YAMLs specify
the snapshot and artifact paths. Different variants can run concurrently:
SmartSim work and TensorBoard files are isolated in
`MCC_STORAGE_ROOT/training/work/credit-VARIANT-JOBID/`. A file lock prevents
simultaneous launches of the same variant. Existing batch-size jobs are unchanged.

Seeds govern policy initialization/sampling and independent per-worker shuffled
snapshot cycles. A worker recovery does not change other workers' snapshot
sequences. Reusing a variant with an existing checkpoint resumes it and requests
**another** 150,000 samples; use a new run name for a fresh repeat. On resume,
snapshot cycles restart from their seed, rather than restoring their previous
position. Actual selected snapshots are retained in the diagnostics.

The one-GPU Bede job uses SmartSim's local launcher inside its allocation;
`ghbatch` supplies the outer Slurm allocation, while Slurm client commands are
not available at the aarch64 compute-node call site.
