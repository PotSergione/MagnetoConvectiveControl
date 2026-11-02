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

## Training diagnostics (local and Slurm)

The standalone trainer writes compact diagnostics beneath
`RUN_DIRECTORY/diagnostics/RUN_ID/`:

| File | Contents |
|---|---|
| `rollout_NNNN.npz` | Physical measurements, actions, policy means/std, values, raw advantages, targets, log probabilities, and independently calculated discounted returns |
| `phases.csv` | Per-environment summaries for episode steps 1–50, 51–100, etc., within each rollout |
| `optimization.csv` | Per-epoch losses, KL, clipping, actual optimizer-step count, and gradient norms after SB3 gradient clipping |
| `provenance.json` | Installed versions, Git revision/dirty flag, model settings, seeds and existing run metadata |

Read archives with `numpy.load(path, allow_pickle=False)`. Physical quantities
have shape `(control_steps, physical_environments)`; agent quantities have
shape `(control_steps, physical_environments, 10)`. `actions` are sampled
policy actions **before** clipping, `policy_means` are deterministic policy
outputs on the same stored observations and recurrent states, and
`applied_actions` are the physical actions reported by the solver. Missing
solver measurements on failure transitions are NaN. Initial first-wall flux
is recorded at episode step 1 and in `episode_metrics.csv` at normal completion.

`return_to_go` uses raw rewards and the configured gamma, independently of GAE.
It is finite only for suffixes ending at a normal, genuine terminal boundary
observed in the buffer. Failed episodes, timeouts and unfinished rollout tails
are NaN, never zero-filled. `timeout_reward_correction`,
`rollout_bootstrap_value` (one value per virtual row) and
`rollout_bootstrap_mask` (one flag per physical environment) identify bootstrap
contributions. `returns` retains SB3's actual training target. Interrupted
collections preserve physical records with `complete_rollout=False`; they
contain no invented advantage/return estimates and are not phase-summarized.

Every artifact identifies the session, rollout, collection policy version and
sample range. `policy_version` is SB3's pre-collection `_n_updates` epoch
counter, not a count of gradient steps. Episode IDs, snapshots and end reasons
are recorded per physical transition. The existing unseeded snapshot selector
is explicitly labelled `SystemRandom`; the records retain its actual choices.
Diagnostics do not seed or consume the training RNG, change recurrent states,
or add CFD steps. Policy means require one additional deterministic actor pass
through the stored rollout, but no wall fields or recurrent states are saved.

Reward statistics now cover exactly one rollout; `metrics_interval` is retained
as a compatibility input but no longer splits collections. Recent episode
means use completions in that rollout; the separately labelled cumulative
mean covers the current training session. Legacy cumulative CSV columns remain
for compatibility. Stdout is dumped after optimization so reward and training
statistics refer to the same rollout, including the final update.

The records support physical, action and credit-assignment analysis, not full
gradient replay. Per-epoch losses/KL/clipping are averages over evaluated
minibatches, excluding recurrent padding. Diagnostics observe the upstream SB3
optimizer rather than providing a separate implementation of PPO.
