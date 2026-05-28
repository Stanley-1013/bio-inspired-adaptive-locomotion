# Phase 3 — Operations Log

How the 384-cell evaluation grid was produced on 2026-05-28/29.

## Grid definition

6 conditions (reference + 5 ablations) × 8 seeds × 8 scenarios = 384 cells,
64 episodes per cell. Each cell loads one trained policy
(`model_3000.pt`) and runs it under one scenario, recording ~15 metrics
aggregated over the 64 episodes. Harness:
[`../../analysis/eval_under_conditions.py`](../../analysis/eval_under_conditions.py).

Scenarios: `nominal`, `payload_{5,8,10,15}kg`, `push_x_{1p5,3,5}`.

## Correctness decisions (why this is not just play.py)

- **Each policy runs in its own training-time env config.** A `no_fatigue`
  policy is evaluated with `motor_fatigue=False` so its observation vector
  matches what it trained on. SATA's `scripts/play.py` instead force-sets
  `activation_process / hill_model / motor_fatigue = True` regardless of the
  loaded policy — which would feed an ablated policy an observation
  distribution it never saw. We do not use play.py for the grid for this
  reason.
- `use_test=True` → `general_scale=1` (fully grown) for every condition.
- Fixed forward command (vx=1) via point-range on `commands.ranges`, **not**
  via the `control_type='T'` shortcut (which would also rewrite the no_growth
  ablation's control path). Verified this keeps each ablation's knob intact.
- Deterministic inference policy (mean action, no sampling); no observation
  noise; per-scenario point-value domain randomisation so eval is reproducible.

## Isaac Gym one-sim-per-process constraint

Isaac Gym allows only **one `create_sim` per process**. An in-process loop
over cells crashes on the second cell. The batch driver therefore **spawns one
subprocess per cell** (`subprocess.run([... --out-csv ...])`); each child
creates its sim, runs, appends its CSV row, and exits. The CSV is append-only
and the driver skips `(condition, seed, scenario)` rows already present, so the
grid is **resumable** after any interruption.

## Two parallel launchers (GPU split)

The main grid ran on **cuda:0**; the eval harness pins sim+policy to one GPU.
GPUs 1–3 were otherwise idle, so a second launcher ran the two *added*
scenarios (`payload_8kg`, `push_x_1p5`) on **cuda:2** in parallel, writing a
separate CSV to avoid a concurrent-append race:

```bash
# main grid (6 original scenarios), cuda:0
nohup setsid python analysis/eval_under_conditions.py \
  --batch-csv results/phase3-bio-claims-and-robustness/raw_metrics.csv \
  --episodes 64 --num-envs 64 &

# supplementary (8 kg + gentle push), cuda:2, separate CSV
nohup setsid python analysis/eval_under_conditions.py \
  --batch-csv results/phase3-bio-claims-and-robustness/raw_metrics_suppl.csv \
  --scenarios payload_8kg push_x_1p5 --gpu 2 --episodes 64 --num-envs 64 &
```

Main grid: 15:08 → ~22:00 (~1.4 min/cell, 288 cells). Supplementary: 96 cells,
finished earlier. Both detached via `nohup setsid` to survive disconnects.

## Scenario tuning mid-phase

The first design only had `payload_{5,10,15}kg` and `push_x_{3,5}`. Two
adjustments after seeing early data:

1. **Added `payload_8kg`** — the real Go2 Pro payload limit. The original grid
   jumped from in-spec 5 kg straight to beyond-spec 10 kg; 8 kg gives a clean
   in-spec upper point and is where the reference begins to fall.
2. **Added `push_x_1p5`** (training-magnitude, 1.5 m/s, every 4 s). The
   original `push_x_3 / _5` (every 2 s) were so severe they crushed *every*
   condition to ~20–50 reward, losing discriminative power. The gentle push
   shows all conditions hold near nominal, confirming the harsh pushes were
   over-driven, not that the policies are fragile.

These were appended to the resumable grid; existing cells were not re-run.

## Metric-design correction (caught in analysis)

The fatigue-distribution metric (`fatigue_*_acrossdofs`) is identically zero
for the `no_fatigue` condition (its fatigue tensor never updates), so it cannot
serve as a cross-condition "load distribution" proxy. We instead read load
distribution from `torque_peak_std_acrossdofs` / `energy_std_acrossdofs`, which
are defined for every condition. (Recorded here so the zero column in the CSV
is not mistaken for a finding.)

Two more metric caveats found in an independent rigor review of the analysis:

- **The `survived` / `early_terminated` flags are uninformative — do not use
  them.** They were defined against `int(env.max_episode_length)` ≈ 4000 steps
  (the nominal-dt cap), but the variable-control-rate eval rollout completes a
  no-fall episode at ~3500–3580 steps, so `survived` is ~0 and
  `early_terminated` ~1 in *every* cell, including healthy nominal walking.
  Falls are read off **episode length relatively** instead (a no-fall rollout
  ≈ 3578 steps; a shorter, high-variance length = falling). Valid as a relative
  measure; the absolute "20 s = N steps" cap is not asserted.
- **`no_activation` nominal energy is NOT significantly different** from
  reference under the declared two-sided Welch test (p≈0.05). An earlier draft
  marked it significant using a one-sided p; corrected to n.s. The headline
  `no_activation` finding (peak torque 42.5 N·m) is unaffected.

## Analysis

Aggregation + Welch t-tests (8 v 8, unequal variance, Satterthwaite df) and
the three figures are produced by
[`../../analysis/plot_phase3.py`](../../analysis/plot_phase3.py). Headline
significant effects all have t > 7; see [`README.md`](./README.md) for the
numbers and verdicts.
