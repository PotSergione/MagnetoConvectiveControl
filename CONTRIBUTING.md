# Contributing

Changes to numerical behaviour require tests and an accompanying experiment or
case configuration update. Do not commit raw runs, restart fields, checkpoints,
machine-private paths or credentials.

Before proposing a change:

```bash
python -m pytest
python -m compileall -q src tests
```

Public contribution and licensing terms will be finalized before release.
