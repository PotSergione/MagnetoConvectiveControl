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
`bede_power9_stack_smoke.slurm`, and isolated ppc64le installation and cache
defaults. See `docs/running_on_slurm.md` for submission and dependency
examples.

The one-GPU Bede job uses SmartSim's local launcher inside its allocation;
`ghbatch` supplies the outer Slurm allocation, while Slurm client commands are
not available at the aarch64 compute-node call site.
