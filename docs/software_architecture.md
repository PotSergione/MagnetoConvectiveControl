# Software architecture

The repository maintains four explicit boundaries:

1. `src/` contains reusable control and orchestration software.
2. `cases/` contains replaceable NekRS physical problems with a common contract.
3. `configs/` contains versioned scientific and machine configurations.
4. `runs/` contains generated artifacts and is never committed.

The command-line layer resolves an experiment and machine profile, writes the
resolved configuration and provenance metadata, and only then imports the
training implementation. Environment variables remain a compatibility layer
for SmartSim and cluster integration, not the primary experiment definition.

## Case contract

Each case provides `case.yaml`, a launcher, solver parameters, UDF and mesh.
The interface section declares action dimensions, observed wall channels and
reward inputs. New geometries should be introduced as new case directories,
not by adding geometry switches throughout the RL implementation.
