# Confirmatory policy evaluation

The primary comparison is paired by initial condition: the deterministic
feedback policy and uniform constant action `0.01` each control the same ten
frozen states for 1000 control steps. Five pairs run feedback first and five
run the constant controller first. The pairing and order are frozen in
`plan.json` before simulation begins.

Freeze the completed generator checkpoints at times 310 through 400:

```bash
mcc-evaluate freeze /home/sergio/test_generator_blanket \
  /home/sergio/test_generator_blanket/selected_snapshots
```

Launch or resume the serial campaign:

```bash
mcc-evaluate run \
  --snapshots /home/sergio/test_generator_blanket/selected_snapshots \
  --checkpoint /path/to/checkpoint.zip \
  --case cases/toroidal_gr1e9_ha1750/run_case.sh \
  --output /path/to/evaluation_results \
  --plan-only
```

Inspect the resulting `plan.json`, then repeat the command without
`--plan-only` to execute it.

Completed attempts are never rerun. An interrupted or failed attempt remains
on disk and the next invocation creates a new numbered attempt. Analyse only
after all twenty runs complete:

```bash
mcc-evaluate analyse /path/to/evaluation_results
```

The reported primary statistic is the mean paired return difference,
feedback minus constant control, with an exact two-sided sign-flip test.
