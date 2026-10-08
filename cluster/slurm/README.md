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
