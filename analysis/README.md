# analysis/

Reusable analysis, plotting, and recording scripts for this project. Outputs
(figures, extracted CSVs, video) land under [`../results/`](../results/) next to
the run they belong to, not here — this directory holds *code only*.

All scripts run from inside the `sata` conda env. The pure-plotting ones need
only `pandas` / `tbparse` / `matplotlib` (CPU); the eval / recording ones import
Isaac Gym and therefore need a GPU box. Unless noted, run from the repo root:

```bash
source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh
cd ~/workspace/bio-inspired-adaptive-locomotion
```

### Evaluation (GPU — Isaac Gym)

| Script | What it does | Output |
|---|---|---|
| [`eval_under_conditions.py`](./eval_under_conditions.py) | Evaluate every (condition, seed) checkpoint under nominal + OOD perturbations (payload / push), logging reward and actuator-feasibility metrics. Each policy runs in an env whose bio-knob config **matches its training config**. One subprocess per cell (Isaac Gym is one-sim-per-process). One CSV row per (policy, scenario). | `results/phase3-*/raw_metrics*.csv` |

### Plotting (CPU — `sata` env)

| Script | What it does | Output |
|---|---|---|
| [`plot_phase2_curves.py`](./plot_phase2_curves.py) | Read tensorboard event files from all 6 × 8 Phase 2 runs and plot reward curves | `results/phase2-ablation/plots/reward_curves_*.png` |
| [`plot_phase3.py`](./plot_phase3.py) | Merge the two Phase 3 eval CSVs → payload-degradation, nominal-feasibility (peak torque / energy / jerk vs hardware lines), and push-degradation figures | `results/phase3-*/plots/{payload_degradation,nominal_feasibility,push_degradation}.png` |
| [`plot_phase4.py`](./plot_phase4.py) | Reward-vs-payload for reference / no_fatigue / ref_residual + a peak-torque panel showing the residual stays in the actuator envelope | `results/phase4-*/plots/residual_payload.png` |

### Recording & viewing (GPU — Isaac Gym)

| Script | What it does | Output |
|---|---|---|
| [`record_demo.py`](./record_demo.py) | Record ONE focused disturbance clip (`--event push/leg_pull/vertical/stairs`) with slow-motion + camera zoom and post-hoc on-frame annotations. Real in-sim forces; overlays drawn after render. | `results/phase3-*/videos/ev_*.mp4` |
| [`record_video.py`](./record_video.py) | Record an mp4 of a (condition, seed) policy under a scenario via off-screen camera sensors (no viewer / VNC). Reuses `eval_under_conditions.py` loading, so the recorded policy runs in the same env it was evaluated in. | `results/*/videos/*.mp4` |
| [`view_policy.py`](./view_policy.py) | Open an Isaac Gym viewer and run a policy continuously for live watching over VNC. Respects each ablation's training-time bio-knob config (unlike SATA's `play.py`, which forces all knobs on). | — (interactive) |
| [`overlay.py`](./overlay.py) | Helper (pure numpy + PIL, no Isaac Gym): draws arrows / impact rings / subtitles onto rendered frames. Imported by the recorders; not run directly. | — (library) |
| [`mp4_to_gif.sh`](./mp4_to_gif.sh) | Two-pass palette conversion of an mp4 to a small GitHub-inline GIF, using the ffmpeg bundled with `imageio-ffmpeg`. `bash analysis/mp4_to_gif.sh <in.mp4> <out.gif> [width] [fps] [start] [dur]` | the GIF you name |

The GPU scripts assume the lab-server layout: SATA checkpoints and tensorboard
event files live under `~/workspace/SATA/legged_gym/logs/SATA/<timestamp>_<run_name>/`.
To run elsewhere, clone SATA to `~/workspace/SATA` or adjust `find_run_dir` in
[`eval_under_conditions.py`](./eval_under_conditions.py).
