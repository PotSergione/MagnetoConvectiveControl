# Toroidal duct, Gr=1e9, Ha=1750

This directory is a self-contained NekRS physical case used by the initial RL
experiments. The magnetic field is aligned with the periodic duct direction.
The case publishes three wall heat-flux maps and scalar diagnostics through
SmartRedis and consumes ten electrode actions.

The case contract is recorded in `case.yaml`. Generated fields, compilation
caches and solver logs must remain outside version control.
