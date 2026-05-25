# Phase 1 — Operations Log

Chronological record of how Phase 1 was actually executed on the lab K8s pod
`han-adaptive-5bb485b87d-v8rp8` on 2026-05-25 / 26. This is the "what I did
and why" notebook; the experimental design and numbers are in
[`README.md`](./README.md).

## 0. Pre-flight environment audit

Before doing anything, I sanity-checked the container:

```bash
hostname                       # han-adaptive-5bb485b87d-v8rp8 → K8s pod
ls /.dockerenv                 # exists → confirmed inside a container
cat /etc/os-release | head -3  # Ubuntu 24.04
findmnt -T $HOME               # NFS from 192.168.23.22 → $HOME persists
                               #   across container rebuilds
nproc                          # 64 CPU cores
free -h                        # 377 GiB total, 174 GiB free
cat /sys/fs/cgroup/memory.max  # 60 GiB cgroup cap (per-pod)
df -h /                        # 937 GB, 683 GB free (overlay; ephemeral)
df -h ~/workspace              # 35 TB NFS, 6 TB free (persistent)
nvidia-smi                     # 4× A6000 (48 GB each), driver 595.58.03
```

Key takeaways that shaped the plan:
- **Persistent storage = `$HOME`** (NFS). All conda envs, repos, training logs
  go here so they survive a pod restart.
- **No sudo** (password prompt). Everything installs in `$HOME`.
- **Python 3.12 only** (system). Isaac Gym Preview 4 needs Python 3.6–3.8 →
  pulled in Miniconda for Python 3.8.
- **CUDA toolkit 12.9 already installed** at `/usr/local/cuda/`, but SATA
  pins cu116 PyTorch wheels — those bundle their own CUDA runtime, so the
  toolkit version mismatch does not matter.

## 1. Stack install

Performed in order; failures recorded so future-me knows what to expect.

| Step | Command (abbreviated) | Notes |
|---|---|---|
| Miniconda | `bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3` | First attempt I aborted (interrupted); the half-finished `~/miniconda3` had to be `rm -rf`'d before a clean reinstall |
| sata env | `conda create -n sata -c conda-forge --override-channels python=3.8 pip -y` | Used `conda-forge` to avoid the new Anaconda Terms-of-Service prompt on the `defaults` channel |
| PyTorch | `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu116` | Pip auto-resolved to `2.0.1+cu117` (no cu116 wheel for torch 2.0.1 on PyPI's `cu116` index); works fine on driver 595 / A6000 |
| Isaac Gym | `pip install -e ~/workspace/isaacgym/python` | User dropped `IsaacGym_Preview_4_Package.tar.gz` into `~/workspace/`; I moved + untarred in place |
| SATA repo | `git clone https://github.com/marmotlab/SATA` into `~/workspace/SATA` | Pinned commit: `8fc422af3fec463a408779b1685c2453d0040be8` |
| rsl_rl & legged_gym | `pip install -e rsl_rl && pip install -e legged_gym` | Editable installs so config edits don't need reinstall |
| Pinned extras | `pip install "numpy==1.21" "setuptools==59.5.0" tensorboard` | **Order matters**: legged_gym pulls in newer numpy/setuptools that must be downgraded *after* it; otherwise SATA's imports break |
| wandb | `pip install wandb` (no `wandb login`) | Discovered only after first sanity-check failure (see §3 below) |

## 2. Container quirk that bit us

Ubuntu 24.04's system Python is 3.12; there is no `libpython3.8.so.1.0` on
`LD_LIBRARY_PATH`. Importing `isaacgym` therefore fails immediately:

```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```

The conda env ships its own `libpython3.8.so.1.0` at `$CONDA_PREFIX/lib/`.
Permanent fix via a conda activate hook:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh <<'EOF'
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
# (matching deactivate hook also written)
```

After this, every `conda activate sata` automatically brings `libpython3.8`
into the loader path. Captured in [setup-sata.md Troubleshooting](../../docs/setup-sata.md#troubleshooting).

## 3. Sanity-check round 1 — false success

First run was deliberately small (50 iters, 2048 envs, headless) to prove
end-to-end. Launched in background with the bash pipeline
`python ... 2>&1 | tail -40`. The shell returned exit code 0 → I declared
success → looked at the output and saw the actual python script had died
with:

```
ModuleNotFoundError: No module named 'wandb'
```

Lesson: `tail`'s exit code masks the upstream's. **Never wrap a sanity check
in a pipe.** SATA's `rsl_rl/runners/on_policy_runner.py` has a hard
`import wandb` even when not actively logging. Fix: `pip install wandb`,
then export `WANDB_MODE=disabled` for runs that should not log there.

## 4. Sanity-check round 2 — pass

Same command without the `tail` wrapper, plus `WANDB_MODE=disabled`:

```bash
cd ~/workspace/SATA/legged_gym/legged_gym
WANDB_MODE=disabled python scripts/train.py \
  --task=go2_torque --headless \
  --num_envs=2048 --max_iterations=50
```

Result: 50 iters in 75 s, 33k steps/s, final mean reward ~3.5 (very low —
that's fine for 50 iters, the curriculum hasn't kicked in yet). The bio-inspired
reward components (`rew_motor_fatigue`, `rew_forward`, …) all reported sensible
values. gymtorch C++ extension compiled once and cached at
`~/.cache/torch_extensions/py38_cu117/` — subsequent runs load instantly.

I monitored this run with the Claude Monitor tool, watching the merged
stdout/stderr stream for `Learning iteration|Traceback|Error|CUDA out of memory|Killed`
patterns — broad enough to catch crashes, narrow enough not to flood me with
per-iteration noise.

## 5. Pre-launch GPU audit

Before launching the full 3-seed run, re-checked GPU state. Critical finding:

```
0: 0 % util,    1 MiB used   ← idle, ours
1: 33 % util,   8628 MiB     ← another tenant's job (PID 73325, not visible
                                from our pod namespace)
2: 0 %,         1 MiB        ← idle, ours
3: 0 %,         1 MiB        ← idle, ours
```

GPU 1 was actively used by a co-tenant. Even though `nvidia-smi` showed all 4
GPUs allocated to our pod, **someone else was already computing on GPU 1.**
Plan adjusted on the fly: 3 seeds (1, 2, 3) on GPUs 0, 2, 3 instead of 4
seeds on GPUs 0–3.

The launch script used `CUDA_VISIBLE_DEVICES=<physical_id>` per process; from
inside the pinned process, the GPU is always `cuda:0` (CUDA renumbers the
filtered set starting at 0).

## 6. Launch matrix

Three background processes, started ~3 s apart so log timestamps don't collide:

```bash
cd ~/workspace/SATA/legged_gym/legged_gym

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python scripts/train.py \
  --task=go2_torque --headless --num_envs=4096 --max_iterations=3000 \
  --seed=1 --run_name=ref_s1 &

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python scripts/train.py \
  --task=go2_torque --headless --num_envs=4096 --max_iterations=3000 \
  --seed=2 --run_name=ref_s2 &

CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled python scripts/train.py \
  --task=go2_torque --headless --num_envs=4096 --max_iterations=3000 \
  --seed=3 --run_name=ref_s3 &
```

## 7. Monitoring during the 65-min run

- `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader`
  every now and then to confirm GPUs warm (5.8 GB VRAM each, 38–41 % util,
  75–85 °C — A6000s, well within thermal budget). GPU 1 stayed
  at ~64 % / 87 °C — co-tenant kept working without contention.
- Tailed the three task output files for `Learning iteration|Traceback|Error`
  patterns via the Monitor tool — no errors throughout.

## 8. Completion

All three seeds finished cleanly (exit code 0 on the actual Python process):

| seed | Final mean reward | Wall time | Logs at `results/raw/...` |
|---:|---:|---:|---|
| 1 | 122 | 63.6 min | `May25_23-12-14_ref_s1` |
| 2 | 108 | 64.7 min | `May25_23-12-17_ref_s2` |
| 3 | 112 | 64.0 min | `May25_23-12-22_ref_s3` |

Reward variance is within normal seed-to-seed RL band (≈ ±5–10 % of mean).

## 9. What I'd do differently

- **Never wrap sanity-checks in `tail` / `head`** — see §3. Use Monitor with a
  grep filter instead, or `python ... 2>&1 | tee log` so exit code propagates.
- **Pre-flight GPU check is non-negotiable on shared boxes.** A Kubernetes
  `nvidia.com/gpu: 4` allocation doesn't guarantee that nobody else is using
  the physical card you can see.
- **Pin numpy/setuptools last.** `pip install -e legged_gym` quietly upgrades
  them; re-pinning after is the only reliable sequence.
- **`WANDB_MODE=disabled` should be in every training command** unless you've
  actively logged in. Save yourself the surprise during a 3-seed parallel run.

## 10. References

- The full env audit lives in this file's §0.
- Install sequence and ordering rules: [`docs/setup-sata.md`](../../docs/setup-sata.md).
- What the trained code does in detail: [`docs/training-internals.md`](../../docs/training-internals.md).
- Concept primer for the RL / bio-inspired terms used above:
  [`docs/concepts-primer.md`](../../docs/concepts-primer.md).
