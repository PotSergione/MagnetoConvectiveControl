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
absolute user paths. The Bede backend discovers the allocated host names with
`bede-mpirun`; a private wrapper does not need to construct `MCC_BEDE_HOSTS`.

Inspect candidate interfaces from a compute allocation with:

```bash
ip -o -4 addr show scope global
```

Use the interface name whose address is reachable between allocated compute
nodes; do not enter an IP address in `smartsim.interface`.

The batch controller is deliberately not wrapped in `srun`. With a conventional
Slurm machine profile, SmartSim launches one nonblocking `srun` step per
physical environment. On Bede GH, the `bede_mpirun` backend instead launches
one independent one-rank NekMHD process on each selected allocation host. In
both cases the simulations remain independent and do not form one distributed
NekMHD MPI job.

Raw wall maps remain transient in SmartRedis. Durable training output is
limited to resolved configuration and metadata, compact reward/episode metrics,
model checkpoints, and diagnostics retained for failed workers.

Training writes numbered atomic checkpoints. Slurm sites that issue warning
signals before termination should configure `--signal` so the Python process
can save its final checkpoint before preemption.

## One-GPU Bede stack smoke

The committed Bede smoke job exercises one real restart, SmartSim/SmartRedis,
one CUDA NekMHD worker, two physical control transitions, and one RecurrentPPO
optimizer update. It deliberately uses a single node, so loopback is the correct
database interface. The outer allocation is submitted with `ghbatch`, while
SmartSim uses its local launcher inside that node because Bede's aarch64 compute
environment does not expose the Slurm client commands required by SmartSim's
Slurm launcher.

The snapshot must be available as
`data/snapshots/blanket0.f00260`. Submit from a Bede login node with:

```bash
cd "$HOME/MagnetoConvectiveControl"
git pull --ff-only

export MCC_STORAGE_ROOT=/nobackup/projects/PROJECT/USER
ghbatch -A PROJECT cluster/slurm/bede_stack_smoke.slurm
```

`MCC_STORAGE_ROOT` is the only required site-private value. The script derives
the existing ARM Conda environment, NekMHD installation, caches and job-local
working directory from it. `MCC_PROJECT_ROOT`, `MCC_CONDA_BASE`,
`MCC_CONDA_ENV`, `NEKRS_HOME`, and `MCC_CACHE_ROOT` remain available as explicit
overrides.

The job preserves all generated files on failure. After success it removes only
its job-local SmartSim case copies and TensorBoard events, retaining compact
configuration, metadata, reward metrics and the reload-validated PPO checkpoint
under `MCC_STORAGE_ROOT/smoke/mcc-stack-JOBID/artifacts/bede_stack_smoke`.

### Two-node/two-GPU integration smoke

The two-GPU Bede launcher requests one GH200 on each of two nodes. Inside the
allocation it uses `bede-mpirun` to discover both nodes, resolves the peer,
asks the controller node's routing table which interface reaches that peer,
and verifies that interface and one GPU on both nodes. SmartSim supervises its
database locally on the controller. Each generated NekMHD model is wrapped by
`bede-mpirun --host NODE -np 1`, which provides remote placement without the
Slurm client commands unavailable on Bede's aarch64 compute nodes. No
interactive compute-node login or hardcoded network-device name is required.

This backend is selected only by `launcher.type: bede_mpirun`. CSF3 profiles
continue to use their existing `local` or `slurm` launchers and do not pass
through the Bede adapter. Host discovery is automatic when `MCC_BEDE_HOSTS` is
unset; the smoke launcher exports the same value explicitly because it also
uses the host list to determine the routable SmartSim interface.

Queue it behind a one-GPU smoke test so that it consumes two nodes only after
the foundational test passes:

```bash
cd "$HOME/MagnetoConvectiveControl"
git pull --ff-only

export MCC_STORAGE_ROOT=/nobackup/projects/PROJECT/USER
ghbatch -A PROJECT \
  --dependency=afterok:ONE_GPU_JOB_ID \
  cluster/slurm/bede_two_gpu_stack_smoke.slurm
```

This job uses Bede's `ghtest` partition and its 30-minute test limit. It runs
two independent NekMHD workers, collects two physical transitions from each,
performs one 40-sample PPO update, and reload-validates the final checkpoint.
The selected route and interface are recorded in Slurm output. Compact results
remain under
`MCC_STORAGE_ROOT/smoke/mcc-2gpu-stack-JOBID/artifacts/bede_two_gpu_stack_smoke`.
