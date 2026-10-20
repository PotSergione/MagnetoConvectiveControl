# Five-snapshot Bede V100 feasibility protocol

This protocol is the deliberately reduced experiment used to locate the
control problem's island of feasibility. It asks whether a recurrent modal
policy can learn a sustained improvement over the inherited uniform `+0.01`
action. It does not establish scientific superiority, generalisation, or
independent-snapshot significance.

The five restart fields are `blanket0.f00250` through `blanket0.f00290` in
increments of ten nondimensional time units. They are temporally correlated.
Five paired comparisons have a best possible two-sided exact sign-test value
of 0.0625, which the final report records explicitly.

## Fixed 43-hour design

The allocation uses one POWER9 node and its four V100 GPUs. Ten full
four-environment blocks are run:

1. Two identification blocks. Every trajectory has 100 reference-action
   transitions, 500 bounded multisine transitions, and a 400-transition tail.
   All ten real cyclic electrode modes are excited. The first block uses four
   distinct snapshots; the second adds the fifth snapshot and three
   sign-reversed repeats. Only scalar trajectories and 30 wall-modal
   projections per transition are retained.
2. Five PPO rollouts. Each rollout contains four complete 1000-transition
   physical episodes and exactly one update. The omitted snapshot rotates, so
   every snapshot appears in four rollouts. The final checkpoint is frozen
   before evaluation.
3. Three evaluation blocks. The deterministic learned policy and constant
   `+0.01` reference are paired once on every snapshot. The two spare slots in
   the last block repeat the learned and reference trajectories on the fifth
   snapshot as a reproducibility diagnostic; they are not additional primary
   statistical units.

The measured four-V100 projection is about 32.1 hours for ten blocks. The
launcher requests 43 hours and enforces a five-hour no-new-block reserve. A
signal five minutes before the Slurm limit records an interruption safely.
Re-submitting the same command resumes from the last committed block.

The feasibility gate requires positive mean full-episode, post-transient, and
final-400 return improvements, with at least four wins out of five for both
the full and final-400 comparisons. This is a predeclared engineering gate,
not a hypothesis test.

## Submission

The default snapshot directory is `$MCC_STORAGE_ROOT/data/snapshots` and the
default durable output is `$MCC_STORAGE_ROOT/campaigns`:

```bash
cd "$HOME/MagnetoConvectiveControl"
git pull --ff-only

export MCC_STORAGE_ROOT=/nobackup/projects/PROJECT/USER
sbatch -A PROJECT cluster/slurm/bede_power9_v100_feasibility.slurm
```

Override `MCC_DESIGN_SNAPSHOTS_DIR` or `MCC_ARTIFACT_ROOT` only when needed.
Use identical values on a resubmission because the plan is immutable.

Successful simulation work directories and raw wall maps are removed. Durable
products are the immutable plan, compact CSV tables and wall-mode projections,
stage decisions, PPO checkpoints, paired evaluation summary, and
`campaign_result.json`.
