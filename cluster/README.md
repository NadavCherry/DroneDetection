# Cluster job scripts (BGU ISE/CS SLURM)

These are **site-specific**. They were written for the Ben-Gurion University SLURM cluster
and every one of them carries an absolute path to one account's home directory. That is not
an oversight to be tidied away — a SLURM batch script has to name a real filesystem, and
pretending otherwise would produce scripts that look portable and work nowhere.

What is here instead is an honest boundary:

| | portable? | how to adapt |
|---|---|---|
| the script **body** | yes | `ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}` — export `SPECKLOCK_ROOT` and nothing in the body needs editing |
| `#SBATCH --output` | **no** | SLURM parses `#SBATCH` directives *before* the shell runs, so an environment variable there is not expanded. These lines must be edited by hand. |
| partition, GPU type, module names | **no** | `--partition main`, `--gpus=rtx_4090:1` and `module load anaconda` are this cluster's names for things. |

So on another cluster:

```bash
export SPECKLOCK_ROOT=/your/path/to/SpeckLock
sed -i "s|/home/cherryn/projects/SpeckLock|$SPECKLOCK_ROOT|" cluster/*.sbatch   # the #SBATCH lines
# then review --partition, --gpus and `module load` in whichever script you are running
```

## Conventions these scripts follow

* **CPU work never holds a GPU.** Dataset building, split construction, scoring saved
  detections and statistical tests all run with no `--gpus` line at all. Several of them
  say so in a comment, because the temptation to reuse a GPU script is real.
* **`cluster/gpu_budget.sh` before every submission.** The account is capped at 7
  concurrent GPUs across *all* sessions, so the number has to be measured rather than
  assumed. That script counts via `scontrol show job`, which is the only form this SLURM
  build reports correctly — `squeue -o %b` prints `N/A` for array tasks and undercounts,
  and `squeue -O AllocTRES` is rejected outright.
* **Long runs carry `--requeue` and resume rather than restart.** A preempted 20-hour
  training job that silently began again at epoch 0 is worse than a failed one.
* **Guards that fail loudly.** Several scripts verify their own preconditions before
  spending GPU time — that a checkpoint's `args.yaml` names the dataset the task claims,
  that a "leak split" actually has overlapping clips, that detection files record the `dt`
  they were produced with. Each of those guards exists because its absence once produced a
  plausible, wrong number.

## What these produced

The reports under `docs/reports/` name the job that generated them. `dt_build` → `dt_train`
→ `dt_eval` → `dt_compare` is the dt ablation; `splitfix` → `leak_train` is the leakage
experiment in the YOLOMG/NPS investigation; `edge_bench` is the edge-model measurement.
