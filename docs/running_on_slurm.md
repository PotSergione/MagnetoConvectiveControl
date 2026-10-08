# Running on Slurm

The committed Slurm script is intentionally site-neutral. A cluster deployment
must supply its module environment, compatible MPI/CUDA libraries, `NEKRS_HOME`,
snapshot location and durable artifact root.

Scientific settings belong in `configs/experiments/`; scheduler and hardware
settings belong in `configs/machines/` or the private submission wrapper.

Training writes numbered atomic checkpoints. Slurm sites that issue warning
signals before termination should configure `--signal` so the Python process
can save its final checkpoint before preemption.
