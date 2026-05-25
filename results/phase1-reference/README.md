# Phase 1 — Reference Reproduction

Goal: confirm that SATA's published `go2_torque` reference config trains a
working torque-based locomotion policy on our hardware, with reward magnitudes
consistent across random seeds. This is the "simulation running" milestone of
the project plan.

## What was run

Three independent training runs, identical config except for the random seed,
launched in parallel (one per A6000):

| seed | GPU | run name | log dir under `results/raw/` |
|---:|---|---|---|
| 1 | A6000 #0 | `ref_s1` | `May25_23-12-14_ref_s1` |
| 2 | A6000 #2 | `ref_s2` | `May25_23-12-17_ref_s2` |
| 3 | A6000 #3 | `ref_s3` | `May25_23-12-22_ref_s3` |

(GPU 1 was occupied by another tenant; we deliberately avoided contention.)

Hyperparameters (unchanged from SATA published config):

| | value |
|---|---|
| Task | `go2_torque` |
| Parallel envs | 4096 |
| PPO iterations | 3000 |
| Rollout length per iter | 24 steps |
| PPO epochs × mini-batches | 5 × 4 |
| Clip ε | 0.2 |
| Target KL | 0.01 (adaptive LR) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |

Bio-inspired stack: `motor_fatigue=True`, `hill_model=True`,
`activation_process=True`, `control_type='TG'` (growth on).

## Results

| seed | Final mean reward (iter 3000) | Wall time | Total env steps |
|---:|---:|---:|---:|
| 1 | 122 | 63.6 min | 295 M |
| 2 | 108 | 64.7 min | 295 M |
| 3 | 112 | 64.0 min | 295 M |

**Mean ± std: 114 ± 6** at iteration 3000.

### Takeaway

- All three seeds converge to comparable reward → the published config
  reproduces reliably, not a one-shot result.
- Per-seed wall time variance (~1 min) reflects NFS / CPU contention from the
  three parallel runs; a solo seed runs ~10% faster (~58 min) per the
  pre-launch sanity check.
- VRAM use per run was ~5.8 GB / 48 GB available → comfortable headroom for
  4-up parallel ablation in Phase 2.

### Reward trajectory

Reward climbed from ~0 throughout training; the bio-inspired reward
components (`rew_motor_fatigue`, `rew_head_height`, `rew_forward`,
`rew_moving_y`, `rew_moving_yaw`) are all written per-iteration into each
run's tensorboard event file. View with:

```bash
tensorboard --logdir ~/workspace/SATA/legged_gym/logs --port 6006 --bind_all
# then http://<server>:6006
```

(Plots are not yet committed as static figures. They will be added when
[`analysis/`](../../analysis/) is created for Phase 2 comparison work.)

## Reproduction

On the lab server, with [`scripts/sata-env.sh`](../../scripts/sata-env.sh)
activated:

```bash
source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh

for s in 1 2 3; do
  gpu=$(( s == 1 ? 0 : s + 1 ))   # avoid GPU 1 if it's busy on your box
  CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled \
    python scripts/train.py --task=go2_torque --headless \
      --num_envs=4096 --max_iterations=3000 \
      --seed=$s --run_name=ref_s$s &
done
wait
```

Expected wall time: ~65 min for the three parallel runs (or ~58 min if a
seed runs alone on an otherwise-idle box).

## Related docs

- [`docs/training-internals.md`](../../docs/training-internals.md) — what the
  code actually does (action→torque pipeline, growth curriculum, PPO loop)
- [`docs/concepts-primer.md`](../../docs/concepts-primer.md) — definitions of
  the underlying terms (Hill model, PPO, GAE, etc.)
- [`docs/setup-sata.md`](../../docs/setup-sata.md) — how to install everything
