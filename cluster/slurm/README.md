# Slurm execution

`train.slurm` deliberately contains only resource allocation and invocation of
the portable `mcc-train` command. Module loading and native-library paths are
site-specific and should be supplied by a private wrapper or shell environment.

Example:

```bash
export NEKRS_HOME=/cluster/path/to/nekrs
export MCC_ARTIFACT_ROOT=/scratch/project/mcc/runs
export RL_SNAPSHOTS_DIR=/scratch/project/mcc/snapshots
sbatch cluster/slurm/train.slurm \
  configs/experiments/first_wall_ppo.yaml \
  configs/machines/slurm_a100.example.yaml
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
