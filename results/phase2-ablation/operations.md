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

## 5. Results (TBD)

Will be filled in after the launcher reports DONE. Plan: lift the per-run
final-reward table from `check_status.sh` output, compute mean ± std per
ablation (over surviving seeds), and add a "what we learned" paragraph
contrasting each ablation with Phase 1's 114 ± 6 baseline.
