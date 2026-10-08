# Data management

The Git repository contains source, configurations, small processed tables and
manifests. It excludes restart fields, model checkpoints, raw SmartSim work
directories, TensorBoard events and solver logs.

Every published dataset should have a manifest containing:

- case and experiment identifiers;
- source or persistent download location;
- file names, byte sizes and SHA-256 hashes;
- creation code revision;
- licensing or access information.

Selected frozen artifacts may be deposited in Zenodo or institutional storage.
GitHub releases can point to those deposits but should not become the primary
storage location for multi-gigabyte run data.
