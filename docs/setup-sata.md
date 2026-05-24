# Setting up SATA (Phase 1 — Reproduce)

Step-by-step setup to reproduce SATA on a **CUDA GPU server**. These steps
mirror the official repo's README (marmotlab/SATA) with prerequisites and notes
added.

> **This cannot run in the Claude Code web sandbox** — there is no GPU, and
> Isaac Gym Preview 4 must be downloaded manually from NVIDIA (EULA / developer
> login). Run everything below on your own CUDA machine.

## Prerequisites

- Linux (Ubuntu 20.04 recommended), an NVIDIA GPU + recent driver, CUDA toolkit.
- Python **3.8** (Isaac Gym Preview 4 supports 3.6–3.8).
- An NVIDIA Developer account (to download Isaac Gym).
- Reference run from the plan: ~20 min, 4096 envs (e.g., RTX 4090).

## 1. Environment + PyTorch

```bash
conda create -n sata python=3.8   # or venv/docker
conda activate sata

# CUDA 11.6 (versions used by the authors):
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --extra-index-url https://download.pytorch.org/whl/cu116
# (older CUDA 11.3 alternative is in the SATA README)
```

## 2. Isaac Gym Preview 4

1. Download from <https://developer.nvidia.com/isaac-gym> (requires login).
2. Unzip, then:
   ```bash
   cd isaacgym/python && pip install -e .
   ```
3. Verify:
   ```bash
   cd examples && python 1080_balls_of_solitude.py
   ```

> If vanilla Isaac Gym / RSL Legged Gym does not run, fix that first — SATA
> assumes a working Isaac-LeggedGym-RslRL stack. It is recommended to first try
> <https://github.com/leggedrobotics/legged_gym>.

## 3. Clone SATA and install its packages

```bash
git clone https://github.com/marmotlab/SATA
cd SATA
pip install -e rsl_rl       # the repo's customized rsl_rl
pip install -e legged_gym   # the repo's customized legged_gym
```

Pinned extras required by SATA:

```bash
pip install "numpy==1.21"        # must be >1.20 and <1.24
pip install tensorboard
pip install "setuptools==59.5.0"
pip install wandb
```

## 4. Train `go2_torque`

```bash
cd SATA/legged_gym/legged_gym
python scripts/train.py --task=go2_torque            # add --headless to disable GUI
# task config: legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py
```

GUI play controls: press `v` to pause/resume.

## 5. Play the trained policy

```bash
python scripts/play.py --task=go2_torque
```

## Troubleshooting (from the SATA README)

- **CUDA errors with many parallel envs** on some GPU/driver combos — reduce
  `num_envs`.
- The codebase targets the authors' hardware; it is a reference, not tuned for
  other robots.

## What maps to our project phases

| Phase | Action in this setup |
|-------|----------------------|
| 1. Reproduce | Steps 1–5 above → "simulation running" |
| 2. Ablation | Edit `go2_torque_config.py`: toggle fatigue, change torque limit, modify growth schedule; vary terrain |
| 3. Adaptive interpretation | Analyze the above behaviors; map mechanisms to control concepts (analogies) |
| 4. (Optional) Residual compensation | Preliminary exploration only, if time allows |
