# Phase 2 — Operations Log

Live notebook for how Phase 2 was set up and executed. Will be updated as the
overnight launcher progresses.

## 0. Pre-flight

Same pod, same conda env (`sata`), same NFS-backed `$HOME` as Phase 1. The
ablations modify the **SATA codebase** under `~/workspace/SATA/`, not this
project repo:

- New file: `SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_ablations.py`
  (5 config subclasses; archival copy at
  [`configs/go2_torque_ablations.py`](./configs/go2_torque_ablations.py))
- Patched: `SATA/legged_gym/legged_gym/envs/__init__.py` (imports + 5
  `task_registry.register(...)` lines; the additions are recorded in
  [`configs/envs_init_additions.txt`](./configs/envs_init_additions.txt))

These are local edits on top of SATA commit `8fc422a`. They are not pushed
upstream to marmotlab/SATA — they only need to exist on our lab pod.

## 1. Sanity-check before mass launch

Tested 3 of 5 ablations end-to-end with `--num_envs=2048 --max_iterations≤50`,
each on GPU 0:

| Ablation | What I looked for | Result |
|---|---|---|
| `no_fatigue` | Reward block omits `rew_motor_fatigue` line | ✅ confirmed (`motor_fatigue=0` reward override worked, fatigue tensor stays zero) |
| `no_growth` | No crash on `control_type='T'` branch (line 179) | ✅ ran 5 iters, ~45k steps/s (faster than ref because no curriculum scaling) |
| `hard_terrain` | Trimesh terrain init doesn't OOM or hang on 60% stairs | ✅ iter time 1.5 s (same as ref); generation completed without warnings |

`no_hill` and `no_activation` are even simpler boolean overrides — trusted by
analogy with `no_fatigue`.

## 2. Launch matrix design (adaptive)

Three GPUs available (0/2/3); GPU 1 holds another tenant's process (7.8 GB
allocated, idle but not ours to use). Naive plan = 5 batches of 3 = 5.5 h
unattended. Refined to two-round adaptive plan:

- **Round A** (diagnostic): 2 batches covering all 5 ablations × seed=1.
  ~2 h wall time.
- **Triage**: scan each Round A log for `Traceback|RuntimeError|OOM|Killed`
  and for completion marker `Learning iteration 2999/3000`. Classify each
  as `OK` / `DID_NOT_FINISH` / `FAILED`.
- **Round B** (confirmation): only `OK` ablations get seed=2 and seed=3.
  Up to 10 runs in 4 batches. ~3.5 h max.

If a particular ablation cannot learn at all, we save ~2 h by skipping its
remaining seeds. Logs are preserved either way.

Implementation: [`launch_all.sh`](./launch_all.sh) with helper functions
`run_batch()` and `classify_run()`. Per-run safety timeout 5400 s (90 min).

## 3. Launch invocation

Run detached via `nohup setsid` so the launcher and its children survive any
shell / Claude-session disconnect:

```bash
cd ~/workspace/bio-inspired-adaptive-locomotion/results/phase2-ablation
nohup setsid bash launch_all.sh </dev/null \
  > ~/workspace/SATA/legged_gym/logs/phase2_launcher.log 2>&1 &
disown
```

Verified after launch:
- `pgrep -af launch_all.sh` shows it running
- `nvidia-smi` shows GPUs 0/2/3 warming up (~5.8 GB / 38 % util each, like
  Phase 1)
- Tail of `phase2_launcher.log` shows the "batch start" message

## 4. Monitoring & morning

Status snapshot from any shell:

```bash
bash ~/workspace/bio-inspired-adaptive-locomotion/results/phase2-ablation/check_status.sh
```

Prints: launcher liveness, active train.py PIDs, GPU snapshot, per-run table
(status / final reward / wall time), tail of launcher log.

## 5. Overnight outcome (2026-05-26 03:07 → 09:30)

Launcher ran 6 h 22 min unattended; final summary lifted from
`phase2_launcher.log`:

| Round | Batch | Contents | Wall |
|---|---|---|---|
| A | 1 | no_fatigue:1, no_hill:1, no_activation:1 | 03:07 → 04:37 (90 min — timeout) |
| A | 2 | no_growth:1, hard_terrain:1 | 04:37 → 06:07 (90 min) |
| — | triage | classified Round A | a few seconds |
| B | 1 | no_hill:2, no_hill:3, no_activation:2 | 06:07 → 07:15 (68 min) |
| B | 2 | no_activation:3, no_growth:2, no_growth:3 | 07:15 → 08:20 (66 min) |
| B | 3 | hard_terrain:2, hard_terrain:3 | 08:20 → 09:29 (69 min) |

Triage outcome: 4 ablations marked `OK` (no_hill, no_activation, no_growth,
hard_terrain); 1 ablation marked `DID_NOT_FINISH` (no_fatigue) and its
seeds 2,3 skipped.

14 of 15 planned runs completed. Final-reward summary table is in
[`README.md`](./README.md). All Round-A and Round-B batches exited with
status 0 (no Tracebacks anywhere).

## 6. Why `no_fatigue_s1` hit the 90-min timeout

The other Round-A batch-1 runs finished in 64–65 min; `no_fatigue_s1` was
still at iter 2497/3000 when `timeout 5400` killed it. My first hypothesis
(that the `motor_fatigue=False` code path's `torch.zeros_like` was
allocating a tensor every step) **was wrong** — the retry (§7) ran in
63–64 min, identical to the other ablations. Looking back at the killed
log's per-iteration times tells the real story:

| iter | iter time (s) |
|---:|---:|
| 0–400 | 1.5–2.1 |
| 600–800 | **2.94–3.42** ← spikes |
| 1000 | 2.17 |
| 1200–1400 | 1.6–2.4 |
| 1600–1800 | **2.4–2.99** ← spikes |
| 2000–2400 | 1.1–2.4 |
| 2491–2497 | 1.1–1.4 ← back to normal |

The slowdown was **environmental contention** (the other K8s tenant on this
host, plus competition with our two batch-mates), not anything specific to
the `no_fatigue` code path. The retry caught the box at a quieter moment.

Lesson: per-run wall time on shared infrastructure is noisy enough that the
safety timeout needs ≥ 50 % headroom over the median observation, not just
20 %. Phase 1 had ~65-min runs alone, so 90 min looked like plenty of
margin — but with 3 jobs sharing the box plus a noisy neighbour, ~2 × the
median is needed.

## 7. `no_fatigue` retry (2026-05-26 10:28)

Re-ran no_fatigue × 3 seeds in parallel on GPUs 0/2/3, with the per-run
timeout extended to 120 min (`timeout 7200`):

```bash
# preserve the killed log first
mv ~/workspace/SATA/legged_gym/logs/phase2_launches/no_fatigue_s1.log \
   ~/workspace/SATA/legged_gym/logs/phase2_launches/no_fatigue_s1_killed_at_90min.log

# ad-hoc launcher (in /tmp; not committed since it is a one-shot retry)
nohup setsid bash /tmp/relaunch_no_fatigue.sh \
  > /tmp/relaunch_no_fatigue.log 2>&1 &
disown
```

Expected wall time ~105–115 min per run (87 min / 2497 iter × 3000 iter +
small parallel-contention overhead). ETA: ~12:30 CST.

Retry results (all 3 seeds exited 0 cleanly):

| seed | Final reward | Wall time |
|---:|---:|---:|
| 1 | 123.79 | 64.4 min |
| 2 | 128.96 | 63.3 min |
| 3 | 124.59 | 64.1 min |

**Mean ± std: 125.8 ± 2.8** at iteration 3000 — _higher_ than the Phase 1
reference (114 ± 6) by ~+12 (+10 %).  The seed-1 number is consistent with
what the killed log was already showing at iter 2497, so the result is
robust.

## 8. Extension to 8 seeds (2026-05-26 21:23 → 2026-05-27 09:02)

The 3-seed analysis produced borderline-significant t values (p ≈ 0.04–0.05)
on several ablations — too thin to back the "Hill model is the only knob
that helps" narrative without risking a reversal. We launched 5 more seeds
(s4–s8) per condition — including the reference — so the final dataset is
6 conditions × 8 seeds = 48 runs. See [`extend_to_8_seeds.sh`](./extend_to_8_seeds.sh).

| Round | Batch | Wall time | Contents |
|---|---|---|---|
| s=4 | a | 66 min | reference, no_fatigue, no_hill |
| s=4 | b | 69 min | no_activation, no_growth, hard_terrain |
| s=5 | a | 71 min | (same trio a) |
| s=5 | b | 70 min | (same trio b) |
| s=6 | a | 68 min | a |
| s=6 | b | 71 min | b |
| s=7 | a | 71 min | a |
| s=7 | b | 72 min | b |
| s=8 | a | 70 min | a |
| s=8 | b | 72 min | b |

Total: 11 h 39 min wall; all 30 runs exited 0 cleanly within the 120-min
safety timeout. Per-run wall settled at 63–68 min, very consistent.

### What the 8-seed data showed

- **Reference variance is much larger than the 3-seed estimate.** Std grew
  from 6 to 16, mean dropped from 114 to 104. Seed 7 in particular trained
  to ~100 then collapsed in the final 5 PPO iterations to 70 (a PPO
  late-training instability — not a data error; iteration 2999 reached
  cleanly, EXIT_CODE=0).
- **`no_hill` -17 % → -5 % (n.s.)**: the seed-1/2/3 sample happened to draw
  the lower tail of a high-variance distribution. With 8 seeds the
  ablation is statistically indistinguishable from reference (p = 0.48).
- **`no_growth` -7 % → -1 % (n.s.)**: similarly dissolved (p = 0.83).
- **`no_fatigue` +10 % → +21 %, `no_activation` +11 % → +23 %**: stronger
  evidence with more data (both now p ≤ 0.006).

### Why we re-tested the reference too (and why this mattered)

Adding seeds only to ablations would have kept the noisy 3-seed reference
as the comparison; both terms in Welch's SE contain `s²_baseline / N_baseline`,
so leaving the baseline at n=3 caps how much the SE can shrink. Bringing
reference to n=8 was what let `no_fatigue` go from t=3.1 (3v3) to t=3.8
(8v8), and what let us catch the seed-7 reference collapse that pushed
the 3-seed estimate optimistic.

## 9. Lessons for future phases

- **n=3 is too few for a paper claim in this codebase.** Two of the
  apparent 3-seed effects were sampling artifacts. n≥5 should be the
  minimum from Phase 3 onward; n=8 was comfortable here and used 11.5 h
  of wall on 3 GPUs.
- **Bring the baseline up with the ablations.** Welch's SE has both
  sample sizes in the denominator; leaving the baseline thin caps your
  statistical power.
- **Per-run safety timeout must allow for ablation-induced slowdown.**
  90 min cut off `no_fatigue_s1` during environmental jitter. 120 min
  proved safe across all 30 extension runs.
- **Adaptive triage paid off** in the first overnight launcher: skipping
  no_fatigue's seeds 2,3 after seed 1 didn't finish avoided ~3 h of GPU
  on what would also have been clipped runs.
- **Preserve killed logs** rather than overwriting them — the killed
  `no_fatigue_s1_killed_at_90min.log` is what let us diagnose the slowdown.
- **GPU 1 never touched** across the ~17 h total of all Phase 2 runs;
  neighbour tenant unaffected.
