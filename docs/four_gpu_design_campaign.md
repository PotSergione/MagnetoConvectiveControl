# Four-GPU control-design and learnability campaign

## Purpose

This campaign is the single pre-production compute investment used to decide
whether a substantially longer reinforcement-learning run is justified and,
if so, to fix its control and learning design. It uses four independent CFD
simulations in parallel. It is not a confirmatory policy evaluation and must
not consume the final confirmatory snapshot set.

The campaign must answer, from one reusable dataset:

1. which joint electrode directions have a repeatable effect on first-wall
   heat flux;
2. the amplitude range over which those effects are useful and approximately
   reproducible;
3. the delay, bandwidth, sign reversal, and long-time tail of the response;
4. the coarsest defensible policy decision interval;
5. whether the current observation makes the controlled quantity and its
   action-dependent future sufficiently observable;
6. the reward-filter, return-discount, GAE, and episode timescales implied by
   the measured physics;
7. the number of independent trajectories required to resolve a useful
   policy change; and
8. whether PPO can improve a deliberately learnable version of the task when
   trained from four independent physical trajectories per update.

The output is a machine-readable design decision, not only a collection of
plots.

## Statistical unit and accounting

A physical joint action followed by one CFD transition is one physical
transition. The ten cyclic agent views are symmetry-related components of the
same joint intervention and do not count as ten independent physical samples.

Every report must state all of the following separately:

- CFD solver steps;
- physical control transitions;
- virtual-agent rows presented to SB3;
- completed physical trajectories;
- distinct restart fields;
- approximately independent causal response windows; and
- GPU-hours.

Uncertainty estimates must resample restart-field blocks, never individual
transitions or virtual-agent rows. Temporal samples within one trajectory and
the ten agents within one physical transition are correlated.

## Frozen inputs

Before submission, freeze at least eight design snapshots that are not reserved
for final confirmation. Record file hashes and source times in the campaign
manifest. Snapshot spacing should be justified using the uncontrolled flux
autocorrelation where possible. If the available fields are closer than one
decorrelation time, that limitation must be carried into the uncertainty
analysis rather than hidden by treating them as independent.

All four arms in one block start from the same restart field. Arm assignment
to allocation-local GPU is rotated between blocks so a hardware-specific
effect cannot be confounded with an action protocol.

Use 1000 physical transitions at the existing `0.1` publication cadence for
all identification trajectories. This preserves the long response currently
under investigation. It does not require the eventual policy to act every
`0.1` time units.

## Action coordinates

Represent the ten electrode actions in a real cyclic Fourier basis:

- mode 0: uniform action;
- modes 1--4: cosine and sine pairs; and
- mode 5: the Nyquist alternating mode.

Normalize every basis vector to unit electrode RMS. Store both modal
coefficients and the resulting ten physical electrode actions. This makes the
input energy comparable between spatial modes and allows the final policy
action space to be selected by measured response rather than convenience.

The dynamic sequences must be deterministic, seeded, mutually orthogonal as
far as the finite horizon permits, and designed with a bounded crest factor.
Their frequency content should span periods of approximately 5, 10, 20, 40,
80, 160, and 320 control transitions. Exact sequences are frozen in the
campaign manifest before the corresponding stage starts.

## Stage A: sustained static response

### Question

Does a simple coherent action produce a sustained improvement, which sign and
amplitude are useful, and when does any initial improvement reverse?

### Design

For every design snapshot, run four paired 1000-transition trajectories:

| Arm | Electrode action |
|---|---|
| A0 | uniform `-0.01` |
| A1 | uniform `0.00` |
| A2 | uniform `+0.01` |
| A3 | uniform `+0.02` |

The amplitudes are deliberately concentrated near the useful scale observed
in earlier work rather than spread across the full clipping range. They may
only be changed before the campaign if a documented safety check contradicts
this range.

### Analysis

For each arm and paired difference from A1, report:

- instantaneous and EWMA first-wall heat flux;
- cumulative reward under candidate filters, without action penalties;
- means over transitions 1--100, 101--200, 201--400, 401--600,
  601--800, and 801--1000;
- `Nu`, `Tbar`, `Kbar`, total heating power, action magnitude, and slew;
- time of first detectable response;
- time and magnitude of any sign reversal; and
- between-snapshot distribution of each quantity.

Use paired snapshot-block contrasts. Select a provisional reference action
`a_ref` only if its full-horizon and final-400-transition effects are both
acceptable. Early transient benefit alone is insufficient.

Stage A also defines a minimum relevant improvement `delta_min` for the
learnability gate. Record both the absolute physical threshold and its value
relative to between-snapshot variability. A useful default is the larger of:

- one standard error of the paired snapshot-block contrast; and
- one quarter of the sustained improvement of the best acceptable static arm
  over zero action.

## Stage B: dynamic and spatial identification

### Question

Which spatial modes and temporal frequencies affect the objective, how much
memory does the action response have, and what observation is sufficient to
represent that response?

### Adaptive input level

After Stage A, choose the dynamic perturbation RMS within the locally useful
range around `a_ref`. The maximum electrode action must remain inside the
predeclared safe range. This is the only adaptive campaign choice; the rule,
not the selected number, is frozen before Stage A.

### Design

For every design snapshot, run four paired trajectories:

| Arm | Protocol |
|---|---|
| B0 | constant `a_ref` |
| B1 | `a_ref + u_low(t)` |
| B2 | `a_ref - u_low(t)` |
| B3 | `a_ref + u_high(t)` |

`u_low` persistently excites the uniform and first spatial harmonic.
`u_high` excites the remaining Fourier modes with the same total input RMS.
The sign-reversed B1/B2 pair separates approximately odd action response from
even or amplitude-dependent effects.

Use the following temporal structure:

- transitions 1--100: `a_ref` in all arms;
- transitions 101--600: frozen persistently exciting sequence; and
- transitions 601--1000: return to `a_ref` to expose the response tail.

### System-identification analysis

Fit regularized finite-impulse-response and low-order state-space models from
modal actions to:

- instantaneous first-wall flux;
- integrated flux on every wall;
- `Nu`, `Tbar`, and `Kbar`; and
- low-dimensional summaries of the spatial wall-flux fields.

Cross-validation is blocked by restart field. Report impulse responses,
frequency response, coherence, uncertainty across snapshots, and prediction
error at horizons 1, 10, 25, 50, 100, 200, and 400 transitions.

A spatial mode is retained only if it provides reproducible out-of-snapshot
predictive information or a paired integrated response exceeding
`delta_min`. Modes that merely explain within-trajectory noise are discarded.

Define the causal tail as the earliest time after an excitation at which the
remaining absolute integrated impulse response is less than five percent of
the total, with uncertainty. If this point is not reached within the recorded
horizon, the campaign must report the horizon as unresolved rather than select
a shorter episode by assumption.

### Observation analysis

Use the same trajectories to compare these nested observation sets:

1. per-frame standardized three-wall maps plus previous action;
2. fixed-physically-scaled three-wall maps plus previous action;
3. set 2 plus instantaneous first-wall flux;
4. set 3 plus the reward EWMA state; and
5. set 4 plus `Nu`, `Tbar`, or `Kbar` only where they add blocked
   out-of-snapshot predictive information.

The targets are current first-wall flux and future flux at the horizons above.
Compare a state-only predictor with an action-conditioned predictor for every
observation set. The action history must improve future prediction; otherwise
the data contain no resolved causal signal for policy learning at that
horizon.

Choose the smallest observation set whose blocked prediction error is within
five percent of the best candidate at both the fast-response and causal-tail
horizons. Absolute fluxes use fixed physical scaling fitted on the design set;
per-frame scaling must not be applied to the appended scalar state.

### Reward and timescale analysis

Because instantaneous flux is recorded, recompute candidate EWMA rewards
offline. Choose the shortest filter that suppresses unresolved high-frequency
flux variation without adding more than ten percent of the measured fast
response time as extra lag.

Express all learning parameters in physical time. For policy decision interval
`Delta_t`, use

```text
gamma        = exp(-Delta_t / tau_discount)
gamma*lambda = exp(-Delta_t / tau_trace)
alpha        = 1 - exp(-Delta_t / tau_reward)
```

Choose the coarsest decision interval that still supplies at least five policy
decisions across the fastest retained action response and does not alias a
retained response band. Aggregate, rather than average away, the physical
rewards inside one held-action interval so the return remains a time integral.

For a finite-horizon sustained-control objective, prefer an undiscounted return
or choose `tau_discount` so the final-window weight is explicitly acceptable.
Choose `tau_trace` no shorter than the action-dependent response that the
critic can resolve. Report the implied weights at the fast-response, reversal,
causal-tail, and terminal times.

## Stage C: four-environment learnability gate

### Purpose

Stage C tests whether the selected design produces an observable improvement
under PPO before a long production run. It is an overfit/optimization gate,
not a generalization claim.

### Configuration

Generate the Stage C configuration from the Stage A/B decision artifact:

- selected modal action coordinates;
- selected action hold and physical decision interval;
- selected observation scalars and fixed scaling;
- selected EWMA, `gamma`, and GAE lambda;
- four independent physical environments, one per GPU;
- one complete physical trajectory per environment in each PPO rollout; and
- a policy frozen across all four trajectories contributing to an update.

Use eight fixed design snapshots, with four running concurrently and the other
four in the following update. Repeat this two-update snapshot cycle five
times, for ten PPO rollouts and forty physical trajectories. Do not count the
ten symmetry-related electrode rows as independent trajectories.

Log per environment and per physical transition:

- raw and filtered fluxes;
- reward components;
- value predictions and, after rollout processing, returns and advantages;
- deterministic action mean, sampled action, log probability, and action
  standard deviation;
- modal and electrode action coordinates;
- recurrent episode-start flags; and
- snapshot identity and physical episode phase.

### Predeclared pass criteria

The learnability gate passes only if all of the following hold:

1. a snapshot-blocked trend in full-episode return is positive and the change
   over the pilot exceeds `delta_min`;
2. improvement is not confined to the first 200 transitions, and the final
   400-transition return does not trend downward;
3. the improvement is present on a majority of the fixed design snapshots,
   not generated by one exceptional trajectory;
4. deterministic policy means change in retained coherent action modes by more
   than their run-to-run estimation uncertainty;
5. PPO diagnostics remain finite, with nonzero but controlled KL divergence;
   and
6. the action-conditioned critic or auxiliary predictor retains predictive
   skill at the selected causal horizon.

Failure is informative. The campaign report must classify it as one or more
of: insufficient controllability, unresolved action effect, inadequate
observation, excessive trajectory variance, optimizer failure, or late-time
objective conflict. A failed gate does not authorize a longer run merely by
increasing total timesteps.

## Required telemetry

Persist one row per physical transition containing:

- campaign, stage, block, arm, snapshot, GPU, and action-sequence identifiers;
- solver step and physical time;
- instantaneous first-wall flux and all integrated wall fluxes;
- total heating power, `Nu`, `Tbar`, and `Kbar`;
- exact ten-component policy and physical actions;
- modal action coefficients;
- every reward component, while retaining enough raw data to recompute reward;
  and
- termination, truncation, and failure status.

Store raw, unstandardized wall-flux maps in a chunked compressed external
dataset. Record its schema, byte size, hashes, code revision, snapshot hashes,
and action-sequence hashes in a versioned manifest. Processed compact tables
may be committed; raw fields remain external according to the project data
policy.

## Execution and restart contract

The complete campaign is one named, restartable workflow with three stages.
Every trajectory has a frozen specification and an atomic completion marker.
Completed trajectories are never silently rerun. Failed attempts remain in
the audit trail and are replaced only by a new numbered attempt.

Stage B may begin only after the Stage A reference action and perturbation
amplitude have been written to the decision artifact. Stage C may begin only
after the complete Stage B analysis has generated and validated its training
configuration. These are analysis barriers within one campaign, not informal
manual choices.

## Approximate resource envelope

At the currently observed cost of roughly 1.5 hours per 1000-transition
trajectory:

- Stage A: eight blocks, 32 trajectories, approximately 12 wall-clock hours;
- Stage B: eight blocks, 32 trajectories, approximately 12 wall-clock hours;
- Stage C: ten rollouts, 40 trajectories, approximately 15 wall-clock hours;
- analysis barriers, smoke tests, and restart margin: approximately 9 hours.

The complete workflow therefore fits approximately one 48-hour, four-GPU
allocation while retaining margin for stragglers. These values must be updated
from a four-environment smoke test before submission.

## Implementation prerequisites

The current training entry point rejects more than one physical environment,
and the committed Slurm example requests one GPU. Before this campaign:

1. enable and test `N_ENVS=4` through the complete SmartSim, MARL, recurrent
   state, callback, and checkpoint path;
2. request and bind four GPUs with one CFD simulation per GPU;
3. verify that snapshot assignment is paired and deterministic for Stages A/B;
4. add the open-loop modal action-sequence controller;
5. add transition telemetry including raw wall maps and policy diagnostics;
6. test interruption and exact campaign resumption; and
7. benchmark four-environment throughput and database/CPU contention.

A one-block smoke test must verify action identity, GPU identity, SmartRedis
keys, recurrent row ordering, reward duplication only across the correct ten
agents, and atomic output completion before the full allocation is submitted.

## Final campaign products

The campaign is complete only when it produces:

- immutable input and artifact manifests;
- paired static-response tables;
- modal impulse and frequency responses with block-level uncertainty;
- observation sufficiency and future-flux prediction tables;
- selected action basis, decision interval, episode horizon, reward filter,
  `gamma`, and GAE lambda with physical-time justification;
- estimated independent-trajectory requirement and projected GPU budget;
- Stage C learnability decision and failure classification if applicable; and
- a generated, frozen configuration for the long run, or an explicit no-go
  decision.

## Executable workflow

The implementation is exposed as one restartable command:

```bash
mcc-campaign \
  --campaign configs/campaigns/four_gpu_design.yaml \
  --machine configs/machines/csf3_four_gpu.yaml
```

On CSF3 it is submitted through `cluster/slurm/four_gpu_campaign.slurm`. The
workflow freezes `plan.json`, runs a four-environment smoke test, advances
through the Stage A and B analysis barriers, generates the Stage C design, and
writes `campaign_result.json`. An unresolved action-conditioned signal at the
Stage B barrier is a completed no-go outcome and prevents speculative PPO.

Stage C uses four centralized recurrent policies acting in the retained real
Fourier coordinates. It therefore presents four independent physical rows to
SB3 rather than forty rows formed by multiplying each trajectory by ten cyclic
views. The ten electrode voltages are reconstructed deterministically from the
selected modal coefficients. Per-transition tables retain both coordinate
systems, while PPO tables retain sampled and deterministic actions, action
standard deviations, log probabilities, values, returns, advantages, and
episode-start flags.
