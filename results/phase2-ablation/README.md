# Phase 2 — Observation & Ablation

**Status: complete (2026-05-27).** Extended from 3 → **8 seeds per condition**
(48 runs total) after the 3-seed numbers turned out to be too statistically
thin to back the original narrative. With the larger sample, two of the
"clear effects" from the 3-seed analysis turn out **not significant** at all
(see Results below). Operational details and the schedule history are in
[`operations.md`](./operations.md).

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

## Results (8 seeds per condition)

Final mean reward at iteration 3000, averaged across 8 random seeds per
condition. Significance is Welch's two-sample t-test against the reference
with Satterthwaite-approximated degrees of freedom (different conditions
have very different variance so a pooled-variance test would be wrong).

| Condition | n | Mean ± std | Δ vs ref | % change | Welch t | df | **p** |
|---|--:|---:|---:|---:|---:|---:|---:|
| reference | 8 | **103.9 ± 15.9** | — | — | — | — | — |
| `no_fatigue` | 8 | **125.8 ± 2.8** | +21.9 | +21 % | +3.84 | 7.4 | **0.006 ✱✱** |
| `no_activation` | 8 | **128.3 ± 7.4** | +24.4 | +23 % | +3.94 | 9.9 | **0.003 ✱✱** |
| `no_hill` | 8 | **98.6 ± 13.4** | −5.3 | −5 % | −0.72 | 13.6 | 0.48 (n.s.) |
| `no_growth` | 8 | **102.5 ± 7.3** | −1.4 | −1 % | −0.22 | 9.9 | 0.83 (n.s.) |
| `hard_terrain` | 8 | **35.98 ± 6.1** | −67.9 | −65 % | −11.26 | 9.0 | **<0.001 ✱✱✱** |

### What changed when we went from 3 → 8 seeds

The original 3-seed analysis claimed *"only Hill model has clear positive
contribution; no_growth mildly hurts"*. **Both of those claims dissolved**
under the larger sample:

- `no_hill`'s apparent 17 % drop at n=3 became 5 % at n=8, and is statistically
  indistinguishable from reference noise (p = 0.48). The Hill ablation is
  high-variance — two of eight seeds reach 75 / 82, two reach 108 — and the
  3-seed estimate happened to sample the lower tail.
- `no_growth` is similarly noise-level at n=8 (p = 0.83).
- The reference itself is much noisier than the 3-seed estimate suggested:
  std grew from 6 to 16, dominated by seed 7 which trained normally to
  reward ≈ 100 then collapsed to 70 in the final 5 PPO iterations — a
  well-known PPO instability, real seed variance, not a data error
  (see [`operations.md`](./operations.md) § 7).

### What survived

Two effects strengthened with more data:

- **Ablating the activation low-pass raises reward by +24** (p = 0.003).
  The EMA on activation sign — meant to model muscle recruitment delay and
  prevent bang-bang torques — turns out to be a constraint on what a trained
  PPO policy can output. PPO already emits smooth actions on its own; the
  low-pass costs peak responsiveness without buying anything the policy
  needed.
- **Ablating the motor-fatigue feedback raises reward by +22** (p = 0.006).
  With the fatigue reward penalty (`-0.05 × Σ|τ| · fatigue`) gone and the
  fatigue observation always zero, the policy is free to use full torque
  budget whenever useful.

And one effect remains overwhelming (and unsurprising):

- **Hard-terrain (60 % stairs) reward collapses to ~36** (p < 0.001), about
  one-third of reference. The policy still converges to a non-trivial reward,
  i.e., the recipe transfers without total collapse, but with a much lower
  ceiling.

### Important interpretation caveat

These results are **training-distribution scalar reward only**. They are
**not evidence that the bio-inspired mechanisms are useless**. In particular:

- Training reward does not measure *out-of-distribution robustness* (payload,
  friction shift, terrain shift) — the SATA paper's own §VI-A limitation
  on payload posture is exactly the kind of measurement scalar reward misses.
- Training reward does not measure *gait quality*, *energy efficiency*, or
  *sim-to-real transferability*. The bio mechanisms may matter more on those
  axes than on training reward.
- A higher training reward without these constraints may simply mean the
  policy has exploited the reward more aggressively — which can correlate
  with *worse* real-world behaviour.

Phase 3 treats these ablations as data points under the adaptive-control
lens: *if* the bio constraints are doing something useful, *where in the
behaviour space should we look for the evidence?*  This framing is the
bridge to Phase 4, which evaluates the ablation policies on SATA's own
stated weakness (payload posture, §VI-A) instead of on training reward.

### Statistical methodology

- **n = 8 seeds per condition.** Standard for a small RL paper claim is 5;
  more is better given how high RL seed variance can be (Henderson et al.,
  "Deep RL that Matters", AAAI 2018).
- **Welch's two-sample t-test** with Satterthwaite-approximated df, because
  variances differ substantially across conditions (`no_fatigue` σ ≈ 3,
  reference σ ≈ 16). A pooled-variance test would understate the SE for
  comparisons against the noisy reference.
- **Final-iteration reward** is reported as-is rather than as
  "last-100-iters mean", to match the SATA paper's reporting convention.
  The PPO instability on `ref_s7` is the most extreme case where this
  matters; for `ref_s7` a last-100-iters mean would give ~97 instead of 70.

### Caveats (operational)

- Runs were in parallel batches of 3 sharing the 64-core CPU and one NFS
  volume. Wall times show 10–20 % jitter from environmental contention.
  The original 3-seed `no_fatigue_s1` failure was an extreme case
  (see [`operations.md`](./operations.md) § 6); all retries and the
  full 8-seed extension landed cleanly.

## Related

- [`operations.md`](./operations.md) — chronological record of how Phase 2
  was set up and run.
- [`../phase1-reference/`](../phase1-reference/) — the baseline these
  ablations are compared against.
- [`../../docs/training-internals.md`](../../docs/training-internals.md) —
  what each config knob actually does in the env code.
