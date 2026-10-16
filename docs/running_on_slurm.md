# Running on Slurm

The committed Slurm script is intentionally site-neutral. A cluster deployment
must supply its module environment, compatible MPI/CUDA libraries, `NEKRS_HOME`,
snapshot location and durable artifact root.

After updating an existing editable installation, rerun
`python -m pip install -e .` once so the `mcc-submit` console entry point is
installed.

Scientific settings belong in `configs/experiments/`; scheduler and hardware
settings belong in `configs/machines/` or the private submission wrapper.
The number of independent physical environments belongs in `configs/scales/`.

Use one code path for both a one-GPU run and a multi-GPU run:

```bash
mcc-submit \
  --experiment configs/experiments/first_wall_ppo.yaml \
  --machine configs/machines/cluster.local.yaml \
  --scale configs/scales/four_gpu.yaml \
  --dry-run
```

Remove `--dry-run` after checking the generated submission command. The
machine-local file selects `sbatch` or the site's compatible wrapper, such as
`ghbatch`, and supplies account, partition, and a routable compute-node network
interface. Loopback is valid only when the SmartSim database and every worker
share one node.

For Bede, copy `configs/machines/bede.example.yaml` to the ignored
`configs/machines/bede.local.yaml`, fill in the interface and any private
storage settings, and leave the committed example free of account names and
absolute user paths.

Inspect candidate interfaces from a compute allocation with:

```bash
ip -o -4 addr show scope global
```

Use the interface name whose address is reachable between allocated compute
nodes; do not enter an IP address in `smartsim.interface`.

The batch controller is deliberately not wrapped in `srun`. SmartSim launches
one nonblocking `srun --nodes=1 --ntasks=1 --gres=gpu:1` step per physical
environment. Each step runs an independent one-rank NekMHD simulation; the
simulations do not form one distributed MPI job.

Raw wall maps remain transient in SmartRedis. Durable training output is
limited to resolved configuration and metadata, compact reward/episode metrics,
model checkpoints, and diagnostics retained for failed workers.

Training writes numbered atomic checkpoints. Slurm sites that issue warning
signals before termination should configure `--signal` so the Python process
can save its final checkpoint before preemption.
