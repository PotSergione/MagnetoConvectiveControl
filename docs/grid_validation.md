# Blanket validation cases

`StandardGrid` is a copy of the established blanket validation case. It
retains the existing paper-replication CSV files and plotting notebook. Its
10 x 10 x 43 spectral elements at polynomial order 6 provide approximately
61 x 61 x 259 unique GLL points.

`FinerGrid` is the intermediate grid-convergence case for Gr = 1e9. It uses
16 x 16 x 43 elements at the same polynomial order, providing approximately
97 x 97 x 259 unique GLL points. Its mesh has already been generated and
converted to `blanket.re2`.

Both cases expose `randomAction` under `[CASEDATA]` in `blanket.par`:

- `false` uses the four fixed `electrode0` through `electrode3` values and
  writes `statistics_Gr_<value>.csv`.
- `true` applies a deterministic sequence of independent electrode actions
  uniformly distributed over [-1, 1]. One action is held for each
  `nusseltInterval`, and output is written to
  `statistics_Gr_<value>_controlled_.csv`.

The fixed seed makes the random forcing identical across grid resolutions,
which permits a controlled grid-convergence comparison.
