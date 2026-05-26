# Phase 2 — Observation & Ablation

**Status: complete (2026-05-26).** 15 of 15 runs landed cleanly after a
retry of `no_fatigue` × 3 seeds — the original run hit the 90-min safety
timeout because of environmental contention on the shared box, not a
code-path issue (see [`operations.md`](./operations.md) § 6).

## Goal

Disable each bio-inspired layer (and harden the terrain) individually and
observe how final reward, behavioural metrics, and learning curve change
versus the [Phase 1 reference](../phase1-reference/). Single-knob overrides so
any difference must come from that knob.

## Ablations

Each is a config-only subclass of `GO2TorqueCfg` (defined in
[`configs/go2_torque_ablations.py`](./configs/go2_torque_ablations.py)) and
registered as a new task in SATA's `envs/__init__.py` (additions in
[`configs/envs_init_additions.txt`](./configs/envs_init_additions.txt)).

| Task name | One-line override | What it tests |
|---|---|---|
| `go2_torque_no_fatigue` | `motor_fatigue=False`, reward scale 0 | Does fatigue feedback drive adaptive load-shedding? |
| `go2_torque_no_hill` | `hill_model=False` | Does the force-velocity drop-off matter for the resulting gait? |
| `go2_torque_no_activation` | `activation_process=False` | Does the activation low-pass tame action smoothness, or is the policy naturally smooth? |
| `go2_torque_no_growth` | `control_type='T'` (forces general_scale=1 from start) | Is the growth curriculum necessary for convergence, or just convenience? |
| `go2_torque_hard_terrain` | `terrain_proportions=[0,0.4,0.3,0.3,0]` (60 % stairs) | Does the reference config still converge on out-of-distribution-style terrain? |

## Launch strategy (adaptive)

A two-round scheduler in [`launch_all.sh`](./launch_all.sh):

1. **Round A — diagnostic**: every ablation with `seed=1`. Two batches of
   3 (no_fatigue/no_hill/no_activation, then no_growth/hard_terrain) on
   GPUs 0/2/3. ~2 hours wall time.
2. **Triage**: each Round A run is classified `OK` / `DID_NOT_FINISH` /
   `FAILED` by scanning its log. "OK" means Python exit 0 and `Learning
   iteration 2999/3000` reached.
3. **Round B — confirmation**: only `OK` ablations get `seed=2` and `seed=3`.
   Up to 5 × 2 = 10 runs in batches of 3 (~3.5 hours). Failed / non-finishing
   ablations are skipped — the log is preserved for inspection.

Worst-case total wall time: ~5.5 hours. GPU 1 is reserved for another tenant
and never used.

## Status / morning summary

Run [`check_status.sh`](./check_status.sh) anytime to print:
- Is the launcher still running?
- Which python train.py processes are alive?
- Live GPU snapshot
- Per-run table: status, final reward, wall time
- Last 8 lines of launcher log

```bash
bash ~/workspace/bio-inspired-adaptive-locomotion/results/phase2-ablation/check_status.sh
```

## Results

Final mean reward at iteration 3000 (`go2_torque` reference for context):

| Ablation | seed 1 | seed 2 | seed 3 | **Mean ± std** | Δ vs reference (114 ± 6) |
|---|---:|---:|---:|---:|---:|
| **reference** (Phase 1) | 122 | 108 | 112 | **114 ± 6** | — |
| `no_fatigue` | 123.8 | 129.0 | 124.6 | **125.8 ± 2.8** | **+12 (+10 %)** |
| `no_activation` | 120.4 | 129.6 | 132.0 | **127.3 ± 5.0** | **+13 (+11 %)** |
| `no_growth` | 107.2 | 109.7 | 100.8 | **105.9 ± 3.8** | −8 (−7 %) |
| `no_hill` | 95.8 | 81.9 | 107.1 | **94.9 ± 10.4** | **−19 (−17 %)** |
| `hard_terrain` | 39.0 | 39.4 | 46.4 | **41.6 ± 3.4** | **−72 (−63 %)** |

### What we learned

The four "bio-inspired" knobs do not have equal weight:

- **Only the Hill force-velocity model has clear positive contribution to
  the training reward** (−17 % when ablated). Without it the policy gets a
  flat torque ceiling and visibly struggles — likely the loss of compliance
  at high contact-velocity events.
- **The growth curriculum is mildly helpful** (−7 % when ablated). Useful
  for convergence stability, but the policy can still learn from a
  fully-grown actuator and full command range; this is a "training
  ergonomics" knob more than a behavioural one.
- **Removing the activation-process low-pass actually *raises* mean reward**
  by +11 %. The EMA on activation sign is supposed to model muscle
  recruitment delay and prevent bang-bang torques, but a trained PPO policy
  already emits smooth actions on its own, so the low-pass becomes a small
  handicap that loses some peak responsiveness.
- **Removing the motor-fatigue feedback also *raises* mean reward** by +10 %.
  The reward-shaping cost of fatigue (`-0.05 × Σ|τ| · fatigue`) is small but
  non-zero; with the cost removed, the policy is free to use its full torque
  budget when useful.
- **Hard-terrain reward drops 63 %** but the policy still converges to a
  non-trivial reward — the reference recipe transfers to a 60 %-stairs
  terrain mix without collapse, just with a much lower performance ceiling.
  This is *generalisation across terrain difficulty*, not a behavioural
  failure.

#### Important interpretation caveat

These numbers are **training-distribution scalar reward only**. They are
**not evidence that fatigue and activation are useless**. In particular:

- The reward signal does not measure *out-of-distribution robustness*
  (payload, friction shift, terrain shift) — the SATA paper's own
  §VI-A limitation on payload posture is exactly this kind of measurement
  the scalar reward misses.
- The reward signal does not measure *gait quality*, *energy efficiency*,
  or *sim-to-real transferability*. The bio mechanisms may matter more on
  those axes than on training-reward.
- A higher training reward without these mechanisms may simply mean the
  policy has exploited the reward more aggressively — which can correlate
  with *worse* real-world or out-of-distribution behaviour.

Phase 3 will treat these ablations as data points under the adaptive-control
lens: *if* fatigue/activation are doing something useful, *where in the
behaviour space should we look for the evidence?*  This framing is the bridge
to Phase 4, which evaluates the ablation policies on the SATA paper's own
stated weakness (payload posture, §VI-A) rather than on training reward.

### Caveats (operational)

- Runs were in parallel batches of 3 sharing the 64-core CPU and one NFS
  volume. Wall times therefore show 10–20 % jitter from environmental
  contention. The original `no_fatigue_s1` failure was an extreme case of
  this jitter (see [`operations.md`](./operations.md) § 6); the retry came
  in at the same throughput as the other ablations.
- Reward-curve plots (per-ablation iteration → reward) are not yet
  generated. When they are, the top-level [`analysis/`](../../analysis/)
  directory will be created to host the script.

## Related

- [`operations.md`](./operations.md) — chronological record of how Phase 2
  was set up and run.
- [`../phase1-reference/`](../phase1-reference/) — the baseline these
  ablations are compared against.
- [`../../docs/training-internals.md`](../../docs/training-internals.md) —
  what each config knob actually does in the env code.
