# Phase 2 — Observation & Ablation

**Status: in progress (launched 2026-05-26, overnight).** Results table will
be filled in when the launcher completes.

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

## Results (TBD)

Will be filled in when the launcher reports DONE. Expected outputs:
- Per-ablation final reward (mean ± std over surviving seeds)
- Cross-ablation comparison table vs Phase 1 reference (114 ± 6)
- Notes on which ablations broke and why (if any)
- (Eventually) reward-curve plots — these will trigger creation of the
  [`analysis/`](../../analysis/) directory at the top of the repo

## Related

- [`operations.md`](./operations.md) — chronological record of how Phase 2
  was set up and run.
- [`../phase1-reference/`](../phase1-reference/) — the baseline these
  ablations are compared against.
- [`../../docs/training-internals.md`](../../docs/training-internals.md) —
  what each config knob actually does in the env code.
