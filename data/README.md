# External data

Large restart fields, checkpoints and raw solver telemetry are intentionally
excluded from Git. Configure their locations with `RL_SNAPSHOTS_DIR` and
`MCC_ARTIFACT_ROOT`.

The `manifests/` directory records the provenance, hashes and expected names
of datasets used for published experiments. It does not contain the datasets.
