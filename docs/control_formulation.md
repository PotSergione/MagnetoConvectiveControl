# Reinforcement-learning setup for electromagnetic blanket control

## Purpose

This document records the physical and algorithmic rationale for the proposed
multi-agent reinforcement-learning (MARL) case.  It complements
[`Todo.md`](Todo.md), which is the implementation checklist.  The present file
defines the problem that should be implemented and preserves the conclusions
drawn from the StandardGrid and FinerGrid validation runs.

The objective is to use ten localized electrode pairs to redirect internally
generated heat away from the plasma-facing `x-` first wall and toward the
other cooled walls.  The primary scientific performance metric is the
integrated outward conductive heat flux through `x-`.  The internal Nusselt
number,

\[
  Nu=\frac{Q}{\overline T},
  \qquad
  \overline T=\frac{1}{V}\int_V T\,dV,
\]

where the volumetric heating `Q` is fixed, remains a thermal-safety diagnostic:
maximizing `Nu` is equivalent to minimizing the volume-mean temperature.
`Tbar`, `Nu`, and kinetic energy are diagnostics rather than the optimized
objective.

## Physical case

The starting point is the validated `Gr = 1e9`, `Pr = 0.0321`, `Ha = 1750`
case under [`../validation`](../validation).  The domain is periodic in `z`
with length `10*pi`.  Ten electrode pairs are uniformly distributed in `z` and
apply opposite electric potentials on the two `y`-normal walls.  The present
model is inductionless:

\[
  \mathbf j=-\nabla\phi+\mathbf u\times\mathbf B_0,
  \qquad
  \nabla^2\phi=\nabla\cdot(\mathbf u\times\mathbf B_0),
\]

\[
  \mathbf F_L=N\,\mathbf j\times\mathbf B_0,
  \qquad
  N=\frac{Ha^2}{\sqrt{Gr}}.
\]

An electrode action specifies a wall-potential amplitude, not a prescribed
volumetric force.  The current and Lorentz-force distributions are solved from
the instantaneous flow state.

## Grid validation and control viability

The relevant trajectories are:

- [`../validation/StandardGrid/statistics_Gr_1.00e+09.csv`](../validation/StandardGrid/statistics_Gr_1.00e+09.csv),
  the long uncontrolled StandardGrid run;
- [`../validation/StandardGrid/statistics_Gr_1.00e+09_controlled_.csv`](../validation/StandardGrid/statistics_Gr_1.00e+09_controlled_.csv),
  the controlled StandardGrid run;
- [`../validation/FinerGrid/statistics_Gr_1.00e+09_controlled_.csv`](../validation/FinerGrid/statistics_Gr_1.00e+09_controlled_.csv),
  the ongoing controlled FinerGrid run.

In the controlled files, the flow is uncontrolled up to `t = 300`, receives a
new seeded random ten-electrode action every `0.1` time units over `300 <= t <
400`, and is subsequently returned to zero forcing.  Both resolutions use the
same action sequence and reconciled action range.

### Uncontrolled statistics

Over `200 <= t <= 300`, the two controlled files are still in their
uncontrolled phase.  Their mean values are:

| Quantity | StandardGrid | FinerGrid | Relative difference |
|---|---:|---:|---:|
| `Nu` | 82.1613 | 81.6075 | -0.67% |
| `Tbar` | 0.0210530 | 0.0211933 | +0.67% |
| `Kbar` | 0.0040772 | 0.0035297 | -13.43% |

The primary thermal statistics agree at substantially better than one percent.
Kinetic energy is more resolution-sensitive, but this does not materially
alter the heat-transfer baseline.

The existing long uncontrolled StandardGrid trajectory provides the stationary
`Nu` safety reference. Over `500 <= t < 700`, the preliminary value is

\[
  Nu_0\simeq80.539.
\]

The adjacent StandardGrid block means are approximately `80.61` over `500 <= t
< 600` and `80.47` over `600 <= t < 700`.  The final baseline metadata must
record the accepted stationary interval and an uncertainty corrected for
temporal autocorrelation.  A new baseline run is required only if the final
grid or stationarity analysis contradicts the existing trajectory.

### Controlled statistics

At the time of this assessment, the FinerGrid run extends to `t = 388.5`,
providing 885 common controlled transitions over `300 < t <= 388.5`:

| Quantity | StandardGrid | FinerGrid | Relative difference |
|---|---:|---:|---:|
| Controlled mean `Nu` | 93.041 | 93.898 | +0.92% |
| Controlled mean `Tbar` | 0.018638 | 0.018492 | -0.79% |
| Controlled mean `Kbar` | 0.07549 | 0.09105 | +20.61% |

Relative to each grid's own `200`--`300` uncontrolled mean, the controlled
Nusselt enhancements are

\[
  \Delta Nu_{\rm Standard}=13.24\%,
  \qquad
  \Delta Nu_{\rm Finer}=15.06\%.
\]

The controlled mean response therefore differs by about `1.82` percentage
points, while the raw controlled means agree within about one percent.  During
the initial `300 < t <= 325` response, mean `Nu` differs by only `0.51%`, and
the StandardGrid--FinerGrid correlations are approximately `0.997` for both
`Nu` and `Kbar`.

The trajectories later lose pointwise synchronization, as expected for a
nonlinear chaotic system.  Identical wall-potential actions need not yield
identical resolved currents, Lorentz work, dissipation, or kinetic-energy
attractors.  The early near-perfect synchronization rules out an obvious
action-range, timing, or electrode-mapping error.  Mean transport statistics,
response direction, response timescale, and normalized control authority are
the appropriate grid-validation criteria; pointwise long-time agreement is
not.

The StandardGrid is consequently considered adequate for MARL development.
The completed FinerGrid run remains useful as final confirmation rather than a
likely decision-changing prerequisite.

## Empirical response timescales

One control transition is 100 solver steps with `dt = 1e-3`:

\[
  \Delta t_c=0.1.
\]

Direct StandardGrid controlled--uncontrolled comparison gives:

| Response | Approximate number of control transitions |
|---|---:|
| Large kinetic-energy response | 2 |
| Detectable `Tbar`/`Nu` response | 10--15 |
| Approximately 0.1% `Nu` response | 24 |
| Approximately 1% `Nu` response | 50 |
| Approximately 5% `Nu` response | 91 |

Within the random-control interval, the raw `Kbar`--`Nu` correlation is about
`0.90`, whereas correlations between their single-transition increments are
near zero.  The interpretation is causal rather than instantaneous:

\[
  \text{electrode action}
  \longrightarrow \text{rapid flow reorganization}
  \longrightarrow \text{delayed thermal transport}
  \longrightarrow \text{mean-}Nu\text{ response}.
\]

The control cadence should remain at `0.1`.  Slowing the controls would remove
useful bandwidth and would not reduce the CFD cost per physical transition.
Temporal credit should instead be handled through recurrence, reward
filtering, GAE, and sufficiently long rollouts and episodes.

## MARL formulation

### Parameter sharing and cyclic symmetry

Treat every electrode as an agent, but use one shared stochastic policy.  The
ten electrodes are equivalent under translations of `Lz/10`; the heat source
and geometry are invariant in `z`.  Agent-centred observations should expose
this exact cyclic symmetry rather than asking ten independent policies to
rediscover it.

Each physical CFD environment produces ten virtual agent rows.  These are ten
correlated, cyclically shifted views of one physical transition, not ten
independent simulations.  Logs and performance reports must distinguish SB3
virtual-agent timesteps from physical CFD transitions.

All agents receive the same global reward.  Each agent emits one scalar, and
the ten scalars are assembled into the physical joint electrode action.

### Observation

The spatial observation is a three-channel collection of outward wall-normal
heat fluxes on the three selected Dirichlet walls:

\[
  H_t\in\mathbb R^{3\times N_\perp\times N_z}.
\]

The channels correspond to `y-`, `y+`, and `x+`; `x-` is excluded.  The exact
wall selection must remain documented alongside the boundary IDs.  Heat flux
is

\[
  q''=-\kappa\nabla T\cdot\mathbf n,
\]

with one consistent positive-outward convention.

Agent `i` observes

\[
  o_i(t)=\left\{
    \operatorname{roll}_z(H_t,-iN_z/10),
    \operatorname{roll}(a_{t-1},-i)
  \right\}.
\]

The full previous ten-component joint action is required because every local
flux field depends on the forcing from all electrodes.  After rolling,
component zero always denotes the observing agent.  `N_z` must be divisible by
ten.

The three wall maps have different transverse coordinates: the `y`-normal
walls use `(x,z)`, while the `x`-normal wall uses `(y,z)`.  Reconstruction must
orient these consistently before stacking them as CNN channels.  CNN padding
is circular only along `z` and nonperiodic along the transverse coordinate.

For the minimal first implementation, standardize each complete three-wall
snapshot using one mean and standard deviation over all channels and spatial
points. This needs no fitted field statistics and preserves contrasts between
walls and the full spatial pattern. It removes the snapshot's global offset
and scale; absolute thermal performance remains represented by the `Nu`-based
reward. Fixed physical scaling remains a later ablation if retaining absolute
wall-flux amplitude proves important.

The current or running `Nu` may be appended as a scalar global observation.
`Kbar` may also be published as a fast diagnostic or optional observation, but
must not be rewarded directly.

### Why full fields are retained

The integrated heat fluxes in the controlled StandardGrid CSV do not strongly
or consistently explain `Nu`.  Over `300 <= t <= 400`, their approximate raw
correlations with `Nu` are:

| Aggregate signal | Correlation with `Nu` |
|---|---:|
| `QwallYMinus` | -0.22 |
| `QwallYPlus` | -0.42 |
| `QwallXPlus` | -0.16 |
| Sum of the three | -0.42 |

The correlations vary in sign and magnitude across subintervals, and their
single-step increment correlations are close to zero.  Control redistributes
heat among walls, and integration removes plume location, boundary-layer
structure, and electrode-relative transport organization.  This is a reason
to retain the spatial maps, not evidence that more scalar variables alone will
make the problem observable.

A future ablation should compare:

1. running `Nu` plus previous actions;
2. three integrated wall fluxes, running `Nu`, and previous actions;
3. full heat-flux fields, running `Nu`, and previous actions;
4. optionally, full fields plus `Kbar`.

### Policy architecture

Adapt the working shared MARL implementation in
[`../../csf3_RB/marl`](../../csf3_RB/marl):

- a CNN processes the three heat-flux maps;
- the shifted previous joint action bypasses the CNN and is concatenated with
  its spatial embedding;
- a shared recurrent PPO actor uses a 128-unit LSTM initially;
- the critic is recurrent as well;
- recurrent state remains attached to the same `(simulation, electrode)` row
  across physical transitions.

The LSTM represents hidden velocity, pressure, current, and interior thermal
state and connects fast spatial responses to delayed global reward.

## Reward

Boundary `1` (`x-`) is interpreted as the plasma-facing first wall. This is
consistent with the imposed volumetric heating `exp[-(x+1)]`, which is maximal
there and decays into the blanket. The reward compares its recent integrated
heat load with the stationary uncontrolled value:

\[
  r_t=C_q\frac{Q_{x-,0}-\widehat Q_{x-,t}}{Q_{\rm generation}}
      -\lambda_a\frac{1}{10}\sum_i a_{i,t}^2
      -\lambda_\Delta\frac{1}{10}\sum_i
       (a_{i,t}-a_{i,t-1})^2.
\]

The online first-wall load is an exponentially weighted moving average,

\[
  \widehat Q_{x-,t}=(1-\alpha)\widehat Q_{x-,t-1}+\alpha Q_{x-,t},
  \qquad
  \alpha=1-e^{-1/\tau_r}.
\]

Use `tau_r = 10` transitions initially, giving `alpha ~= 0.0952`; `tau_r = 15`
is the first planned sensitivity test. Initialize the filter at `Q_x-,0` on
every episode reset, making the initial routing reward zero. Calibration from
the random-forcing runs gives an uncontrolled first-wall flux near `0.0166913`
on StandardGrid and a controlled redirected fraction of order `0.1--0.16`.
Accordingly `C_q=1` already supplies a suitably scaled reward.

The UDF publishes the integrated `x-` flux directly. It is deliberately not
added as a policy observation: the actor retains the existing `y-`, `y+`, and
`x+` spatial maps plus previous actions. Direct publication avoids estimating
the instantaneous storage term from finite differences of `Tbar`. In a
stationary mean the equivalent balance is

\[
 Q_{x-}=Q_{\rm generation}-(Q_{x+}+Q_{y-}+Q_{y+}).
\]

Do not use a cumulative mean from the start of the simulation or only the
increment of the running mean:

- a cumulative mean progressively loses sensitivity to new actions;
- an increment reward tends to zero after a beneficial state is maintained.

The first-wall redistribution begins earlier than the bulk `Nu` response and
reaches roughly half of its calibrated response in `23--31` transitions. The
10-transition EWMA suppresses flux noise without imposing the much longer bulk
thermal delay.

The initial learnability experiment may use zero or small action penalties to
establish whether sustained `Nu` improvement can be learned.  Subsequent
scientific optimization should use a physically motivated electrical-power or
action-magnitude cost and a slew penalty.

## Temporal credit assignment

Use the following initial PPO parameters:

```text
gamma:       0.997
GAE lambda:  0.98
```

Their combined characteristic trace length is

\[
  \frac{1}{1-\gamma\lambda}\approx44
\]

physical transitions, consistent with the measured 25--50-transition thermal
response.  A common `gamma = 0.995`, `lambda = 0.95` configuration has a scale
of only about 18 transitions and is less appropriate here.

PPO rollout boundaries are optimizer boundaries only.  The CFD state, EWMA,
and LSTM state must continue across them unless the physical episode actually
resets.

## Episode and rollout length

Use a physical episode length of 1000 control transitions:

\[
  T_{\rm episode}=1000\times0.1=100.
\]

The validation data suggest that the first approximately 250 transitions (25
nondimensional time units) are dominated by a generic mixing transient that
random actuation can produce.  A 1000-transition episode leaves about 750
transitions in which policies can be distinguished by how well they sustain
and organize the enhanced transport.  It also matches the duration of the
existing `300 <= t < 400` random-control experiment.  A 750-transition episode
is the lower-cost fallback; 512 transitions leave only 262 post-transient
samples and are no longer the preferred production choice.

The initial training configuration is:

```text
physical CFD environments:  1, running sequentially
episode length:             1000 physical transitions
PPO rollout n_steps:        256 physical transitions per virtual row
virtual agents/environment: 10
samples per PPO update:     256 * 1 * 10 = 2560 SB3 samples
updates by 100k samples:    approximately 40
```

Each physical episode spans approximately four PPO updates.  The final rollout
of an episode crosses its time-limit reset in the usual vector-environment
manner; recurrent episode-start masks must reset the appropriate LSTM rows.
Do not restart the simulations at every policy update.  Doing so would discard
mature controlled states, increase restart overhead, and make the physical
horizon depend on an optimizer detail.

At the 1000-transition time limit, report a Gymnasium truncation rather than a
physical termination and bootstrap the value function from the terminal
observation.

## Stationary snapshot strategy

Generate a pool of initial states from a long uncontrolled initializing
simulation:

1. run through startup and establish stationarity;
2. save complete restart states only in the stationary interval;
3. space them by approximately one correlation time where practical;
4. partition them into training, validation, and test pools;
5. initialize each sequential episode from the next training snapshot.

A conservative initial spacing is `30--50` nondimensional time units, or
300--500 control transitions.  Closer snapshots remain useful but should not be
counted as fully independent states.

An initial target is:

```text
training snapshots:    10--20
validation snapshots:   3--5
test snapshots:         3--5
```

Use a shuffled queue without replacement so all training snapshots are exposed
before reshuffling.  The single physical environment loads the next snapshot
when its current episode ends.  Held-out validation and test snapshots must not
be sampled during training.

On reset:

- restore the complete stationary flow/thermal state;
- set the declared initial electrode action, normally zero;
- initialize `Q_x-_hat = Q_x-,0`;
- reset the agent LSTM states;
- begin a new 1000-transition episode.

With one physical environment, ten agents, and 100,000 SB3 samples, the run
contains approximately 10,000 sequential physical transitions, or ten complete
trajectories of length 1000 and ten completed restarting fields.  RecurrentPPO
normally completes its final rollout, so a nominal 100,000-sample request with
2560 samples per update will generally finish at 102,400 samples after 40
updates.  The physical environment then advances 10,240 transitions: ten
complete episodes plus 240 transitions into an eleventh.  Consequently, ten
episodes complete but eleven snapshot fields may be touched.  This rounding
should be reflected in run planning and logs.

## SmartSim integration

The SmartSim/SmartRedis architecture from `../../csf3_RB/marl` should remain
substantially unchanged:

- one NekRS process per physical environment;
- isolated run directories and key prefixes;
- one ten-component physical action tensor;
- one committed solver frame per control transition;
- `step`/`time` frame validation, with `time` written last;
- asynchronous `step_async`/`step_wait` behavior;
- bounded failure recovery and checkpoint randomization;
- ten virtual agent rows assembled around each physical process.

The required changes are physics-specific tensor names and shapes, ten-agent
dimensions, three heat-flux reconstructions, direct scalar `x-` flux
publication, and the stationary heat-routing EWMA reward. No redesign of process orchestration is
planned.

## Evaluation criteria

Training reward is not sufficient evidence of success. Evaluate deterministic
policies on held-out stationary snapshots and compare them with:

- zero control;
- representative uniform constant actions;
- seeded random actions;
- any useful simple patterned controls identified during validation.

Report:

- stationary or long-window mean `Q_x-` and its fraction of generated heat;
- redirected heat fraction relative to the uncontrolled baseline;
- `Nu` and autocorrelation-aware uncertainty as thermal-safety diagnostics;
- `Tbar`;
- action magnitude, slew, and eventually electrical power;
- `Kbar` and wall-flux fields as mechanism diagnostics;
- variability across held-out snapshots.

Use 1000--2000-transition evaluation episodes where affordable.  The principal
claim is sustained improvement in heat transfer, not short-term return or
pointwise reproduction of one chaotic trajectory.

## Held-out evaluation of the 275k policy

The first confirmatory campaign evaluated the deterministic 275,000-sample
checkpoint against uniform constant actuation `a_i = 0.01`. Ten uncontrolled
restart fields at `t = 310, 320, ..., 400` were frozen before evaluation. Each
controller was run for 1000 physical control transitions from every field,
with controller order counterbalanced across the ten pairs. The prespecified
primary statistic was the paired total-return difference

\[
  \Delta_i=R_{\mathrm{feedback},i}-R_{\mathrm{constant},i},
\]

tested by enumerating all `2^10` sign assignments. The campaign completed all
20 trajectories. Its primary results were:

| Statistic | Result |
|---|---:|
| Mean paired return difference | -4.0940 |
| Median paired return difference | -4.4988 |
| Feedback wins | 2 of 10 |
| Exact two-sided sign-flip `p` | 0.1621 |

This does not establish a statistically significant difference and provides
no evidence that this checkpoint is superior to constant `0.01` actuation.
The observed direction instead favours the constant controller. This negative
confirmatory result must remain distinct from subsequent exploratory
diagnostics.

### Physical consistency

The result is not an artefact of reward aggregation. Across the ten pairs, the
mean first-wall heat flux was `0.0165809` under feedback and `0.0163652` under
constant actuation. Feedback therefore produced approximately `1.32%` more
first-wall heat flux on average and was worse on this quantity in eight pairs.
The exact paired sign-flip test on mean flux gave `p = 0.1719`. Paired return
and first-wall-flux differences had correlation `-0.9998`, confirming that the
implemented reward represents the intended routing objective.

Feedback nevertheless increased global convection: relative to the constant
controller it produced higher mean `Nu` and `Kbar`, and lower `Tbar`, in all
ten pairs, while total heating power was unchanged. The failure is therefore
specific to heat routing. The policy drives stronger transport but, over the
full horizon, directs more heat toward the plasma-facing wall.

### Dependence on episode phase

The paired result changes qualitatively with control time:

| Cumulative horizon | Mean paired return difference | Feedback wins |
|---:|---:|---:|
| 100 | +0.373 | 9 of 10 |
| 200 | +0.987 | 8 of 10 |
| 400 | +0.023 | 5 of 10 |
| 600 | -1.407 | 5 of 10 |
| 800 | -4.219 | 3 of 10 |
| 1000 | -4.094 | 2 of 10 |

Feedback is consequently useful during the initial transient but fails to
maintain its advantage after its own actuation reorganizes the flow. In the
`601--700` block it underperformed the constant controller on all ten restart
fields. Short-horizon evaluation would therefore have produced a misleadingly
favourable conclusion.

### Nature of the learned action

The policy is state-dependent but acts predominantly through a spatially
uniform mode. Over the complete campaign:

- the pooled mean action was `0.01098`, close to the constant `0.01` baseline;
- mean spatial standard deviation across the ten electrodes was `0.000627`;
- mean pairwise electrode correlation was `0.990`;
- all ten actions shared one sign during `97.3%` of transitions;
- lag-one action correlation was approximately `0.999`;
- no action approached its clipping bound.

This is not one fixed open-loop schedule. Temporal action histories from
different restart fields had mean pairwise correlation `-0.011`, and
snapshot-specific temporal variance was about 10.7 times the variance of the
cross-snapshot mean schedule. The most precise interpretation is that the
policy learned state-dependent modulation of an almost uniform, slowly varying
electrode mode rather than strong spatially localized feedback.

### Working diagnosis for subsequent training

The physical episode length during training was 1000 transitions, but PPO
collected and updated after 256-transition rollouts. Thus one CFD episode
spanned approximately four policy updates. The agent did encounter mature
controlled states, but direct recurrent backpropagation and sampled advantage
estimation were confined to rollout segments; delayed effects across a
boundary depended on value-function bootstrapping. The poor explained
variance observed during training makes that bootstrap an important likely
failure point. Moreover, the deployed policy was frozen for all 1000 test
steps, whereas the policy parameters changed several times within a training
episode.

The loss of evaluation advantage around the first 200--400 transitions is
consistent with this rollout-horizon limitation. It is not yet proof of
causation, but it motivates the next controlled experiment: retain the same
physical objective and 1000-transition episode while collecting a complete
episode before each PPO update. The purpose is to preserve the useful initial
response while exposing the value learner directly to the adverse long-time
flow reorganization. Exact optimizer and memory settings should be fixed in a
new experiment configuration before that run begins.

Raw campaign artefacts are intentionally untracked under
`evaluation_runs/feedback_vs_constant_275k/`; the frozen plan, per-transition
trajectories, summaries, and primary `analysis.json` provide the local audit
record.

## Outstanding confirmation

The FinerGrid controlled run should be allowed to complete through `t = 400`
and inspected visually.  Its current thermal agreement already validates the
resolution choice for the intended purpose.  A planned impulse or PRBS test can
later refine the action-to-`Nu` response kernel and revisit `tau_r`,
`gamma*lambda`, and rollout length.  These are confirmation and tuning steps;
they do not presently expose a conceptual obstacle to implementing the MARL
case.

## Implementation status

The first implementation now lives directly in this directory:

- `blanket.udf` publishes globally assembled `yMinus`, `yPlus`, and `xPlus`
  heat-flux fields plus integrated `xMinus` flux, generated thermal power,
  `Nu`, `Tbar`, `Kbar`, the applied action, and an action
  sequence. Rank zero owns SmartRedis; MPI broadcasts each accepted action.
- `blanket_marl_env.py` reconstructs `(3,128,160)` maps, standardizes each
  complete observation, implements the first-wall EWMA reward above, cycles restart
  snapshots without replacement, and creates ten shifted virtual-agent views.
- `model.py` applies nonperiodic padding in the transverse dimension and
  circular padding in `z`, then appends the shifted ten-action history.
- `blanket_marl_train.py` and `blanket_train.slurm` encode one physical
  environment, 256-step rollouts, 1000-step episodes, and the documented PPO
  parameters. Resume metadata records both action mapping and the RL contract.

No training was started during implementation validation. Before the first
run, populate `snapshots/` with stationary restart fields. Observation
standardization is computed independently from each current three-wall field,
so no fitted normalization dataset is required; `Q_x-,0` is the reward
baseline statistic. Solver-coupled smoke tests remain necessary on the target
GPU/NekRS installation.
