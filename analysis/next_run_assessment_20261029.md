# Assessment after Bede job 1112418

This is an offline assessment, not a launch configuration. No trainer, reward,
solver, or scheduler behavior was changed. Job 1112418 was stopped by the user
after the reported 200,000-sample checkpoint. Its stdout is available in the
conversation; its checkpoint, per-episode CSV, TensorBoard events, and worker
logs are not present in the inspected local result directories.

## Evidence inspected

- `docs/control_formulation.md`: physical objective, response times, original
  observation/reward design, and local confirmatory evaluation.
- `docs/four_gpu_design_campaign.md` and `docs/bede_v100_feasibility.md`:
  identification and learnability protocols. These describe planned procedures;
  their presence is not evidence that their decision gates passed.
- All three CSVs in `analysis/reference_results/grid_validation`, and the
  analysis notebook sources. These grid CSVs do not directly record target-wall
  flux or the ten actions. Inferring target-wall flux requires a thermal-storage
  balance, so they are not a replacement for direct action-to-target-flux
  identification.
- All twenty 1000-step paired evaluation trajectories under
  `evaluation_runs/feedback_vs_constant_275k`, plus provenance and summaries.
- Original 275k checkpoint metadata/weights and reward history under the
  archived `nekMHD/research_tests/fusionB/rl/models/` tree.
- The distinct local 300k scratch run, including its contract and checkpoint.
- Current and job-revision (`a053f27`) trainer/environment implementation,
  UDF action/reward exchange, policy feature extractor, and installed SB3
  recurrent rollout, timeout, advantage, and optimization implementation.

The separate e-fold analysis mentioned in another chat was not found in the
repository or its referenced historical Markdown. The numeric time scales
below distinguish documented response estimates from direct calculations.

## What the stopped run establishes

At 200k SB3 samples there are 20,000 physical transitions, 100 complete
200-step physical episodes, and 25 collected four-environment rollout batches.
The log at iteration 25 contains optimization diagnostics from the previous
training phase (120 epochs counted as 24 phases of five epochs), not evidence
of 120 independently collected batches.

`episode_reward_mean` averages up to 1000 completed physical episodes, so all
100 episodes contribute here. It is neither the latest episode return nor a
25-episode sliding average. From the rounded log values:

- first 20 episodes: mean 2.74;
- first 60 episodes: mean 2.36;
- first 100 episodes: mean 1.86;
- therefore episodes 61--100: mean approximately 1.11.

This is substantive deterioration of stochastic training performance, although
individual episodes do not decline monotonically. Snapshot-blocked analysis
and deterministic evaluation of the Bede checkpoint are still missing.

There is no clear numerical optimization explosion in the shown KL, clipping,
or action-standard-deviation metrics. High explained variance measures fitting
of the training return targets; it does not validate the objective, action
credit, or a continuing-value extrapolation at the reset boundary.

The recurring Redis errors cannot be classified from top-level stdout. The
worker exchange checks sequence and frame consistency; continued progress is
not proof that all I/O problems were harmless. Preserve the run's error logs
and inspect episode end reasons before reusing it as a clean comparison.

## Response and learning clocks

Control interval: 0.1 nondimensional time units.

| Quantity | Control steps | Nondimensional time | Basis |
|---|---:|---:|---|
| Large kinetic-energy response | about 2 | 0.2 | documented grid comparison |
| Detectable bulk thermal response | 10--15 | 1--1.5 | documented grid comparison |
| First-wall half-response | 23--31 | 2.3--3.1 | documented calibration |
| EWMA e-fold time | 10 | 1 | exact filter parameter |
| EWMA mean lag | 9.51 | 0.951 | `(1-alpha)/alpha` |
| GAE residual-weight e-fold time | 43.09 | 4.309 | `-1/log(.997*.98)` |
| Discount e-fold time | 332.83 | 33.283 | `-1/log(.997)` |
| Physical episode | 200 | 20 | Bede configuration |

The documented 23--31-step half-response would correspond to a 33--45-step
e-fold time *if* a first-order model applied. This is not a measured impulse
tail, and kinetic, Joule damping, target-flux, and bulk-temperature response
times must not be treated as interchangeable.

GAE residual weights at lags 25, 50, 100, 200 are approximately 0.560, 0.313,
0.0982, 0.00964. These are not hard credit cutoffs: the critic can represent
later consequences. The numbers do show how heavily distant effects depend on
the critic rather than directly sampled residuals.

In the paired deterministic trajectories, mean target-flux reductions versus
constant +0.01 are approximately:

| Window | Mean flux reduction |
|---|---:|
| 1--25 | -0.00000101 |
| 26--50 | +0.0001107 |
| 51--100 | +0.0004471 |
| 101--200 | +0.0002690 |
| 201--400 | -0.0002722 |

These are trajectory contrasts, not causal impulse responses. They support
testing the 200-step transient objective, and show why instantaneous velocity
response is not enough to select reward/credit time scales.

## Reward and exploration findings

Removing EWMA changes the local 200-step feedback-minus-constant advantage
from +0.98698 to +0.97149, and the full 1000-step advantage from -4.09401 to
-4.03063. A shared initial-flux reference or zero reference changes absolute
returns but leaves paired differences unchanged. Initial reset flux was not
saved, so the rescoring uses an explicitly labelled first-step proxy.

Reward filtering adds history and delay, and remains a useful ablation. The
rescoring does not prove equivalent learning, but it does rule out a simple
reversal of the evaluated physical objective caused by the filter/reference.

The evaluated 275k local checkpoint has sigma 0.0983436; Bede reports 0.0984.
The separate local 300k scratch checkpoint has sigma 0.0984341. The evaluated
policy's first-200-step deterministic action RMS is 0.0140684. Its mean
instantaneous spatial standard deviation is 0.000595, and 99.724% of squared
action magnitude lies in the spatially uniform component. This establishes
what that policy used, not that other modes are uncontrollable or useless.

The same noise amplitude supported local transient control, so noise collapse
is not an established explanation. Conversely, large independent noise does
not establish good exploration of useful persistent, coordinated actions.
Stored deterministic evaluation trajectories cannot settle that question.

The evaluated local policy used one physical environment, 256-step rollouts,
and minibatches of 256. The current run used four environments, 200-step
rollouts, and minibatches of 1000. The local 300k scratch run used 1000-step
rollouts/minibatches and is not the checkpoint in the ten-pair evaluation.

## First priority: align the episode objective

The original 1000-step design explicitly treated reset as a time-limit
truncation of a continuing task. The standalone Bede trainer still leaves
`episode_horizon_truncated=True`. RecurrentPPO therefore adds
`gamma * V(terminal_observation)` at each 200-step boundary, despite resetting
the physical simulation and recurrent state to another snapshot.

This is correct for the *continuing-task approximation*, but not for the
newly stated objective of optimizing only the first 200 control transitions.
No trajectories in the current training episodes supply actual post-200
continuation rewards from that terminal state. The continuation is estimated.
With gamma=.997 the 200-step discounted continuation coefficient is 0.5483
in a full discounted-return expression. GAE propagates terminal errors with
its own shorter residual weighting; 0.5483 must not be interpreted as the
terminal coefficient in every stored advantage.

The campaign environment already sets `episode_horizon_truncated=False`;
the standalone training entry point does not. The concrete first experiment
should use that finite-horizon semantic, not increase the learning rate or
rescale rewards on the assumption that either caused the failure.

This identifies an objective mismatch, not a proven cause of the decline.

## Recommended next training pilot

Use a new run identity and fresh initialization. Keep the four-GPU,
five-snapshot shuffled training setup and the same observation/ten-agent
architecture. First change only the horizon semantic:

- 200 control transitions per episode and 200 per rollout;
- true task termination at transition 200, with no terminal-value bootstrap;
- retain gamma=.997, lambda=.98, learning rate 1e-4, five epochs,
  minibatch 1000, log_std_init=-2.3, and the current reward/reference/scale;
- 80,000 virtual-agent samples (10 complete rollout batches, 40 physical
  episodes), with deterministic evaluation at initialization and completion;
- evaluate on the same five snapshots against constant +0.01, reporting raw
  target-wall flux over 1--200 and 101--200, alongside the unchanged reward.

This is a discounted finite-horizon task. Setting gamma=1 would be a separate
choice to weight all 200 steps equally; it is not required to remove the
unintended continuation. Likewise lambda=1 is a separate Monte Carlo-target
ablation, not an automatic consequence of a 200-step episode.

Use the existing 200k checkpoint in a matched deterministic evaluation before
discarding it. It may perform differently from its stochastic training policy.
The diagnostic does not require continuing its training.

The pilot's rollout collection projects to roughly 12 hours from the observed
71--72 minutes per batch; evaluation, compilation, and restart overhead are
additional. A plateau is not grounds for extending blindly: require positive
paired raw-flux improvement from the initial to the final policy, inspect the
distribution across snapshots, and compare with the constant baseline.
Five nearby snapshots support a learnability diagnostic, not broad significance.

If a subsequent reward ablation is needed, remove only EWMA and retain the
same fixed reference/heating-power scaling. Do not simultaneously change the
reference, scale, action noise, learning rate, observation, and action basis.

## Minimal telemetry needed for the pilot

Retain one row per physical transition with snapshot, step, applied actions,
deterministic means, sampling sigma, raw/filtered target flux, heating power,
and termination reason. Store the initial pre-action flux once per episode.
Retain compact summaries of sampled returns/advantages and the terminal
bootstrap contribution (which must be zero for true horizon terminations).
Do not retain raw wall fields or extra CFD checkpoints. Keep policy checkpoints,
episode CSVs, and brief failure excerpts. This is enough to distinguish policy
drift, action saturation, physical deterioration, and reward-target artifacts.
