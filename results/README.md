# Results

Phase-by-phase outcomes of this project. Each phase has its own subdirectory
with a `README.md` summarising what was run, the numbers, and how to reproduce.

Heavy artifacts (per-iteration checkpoints, tensorboard event files; ~850 MB
per phase) are not committed. On the lab server they live at
`~/workspace/SATA/legged_gym/logs/SATA/` and are exposed here via the
`raw` symlink (gitignored). Off-server readers can regenerate them by
following the reproduction steps inside each phase's README.

## Index

| Phase | Status | Summary |
|---|---|---|
| [phase1-reference](./phase1-reference/) | done (2026-05-26) | SATA reference reproduction (initial 3 seeds 114 ± 6; extended to 8 seeds gives 104 ± 16 — see Phase 2) |
| [phase2-ablation](./phase2-ablation/) | done (2026-05-27) | 5 ablations × 8 seeds on *training reward*. Removing fatigue/activation raises reward (expected sign of a sim-to-real constraint); hill/growth n.s. Real value assessed in Phase 3 |
| [phase3-bio-claims-and-robustness](./phase3-bio-claims-and-robustness/) | done (2026-05-29) | 384 cells. The ablations that won Phase 2 reward breach the hardware envelope (no_activation peak 42.5 N·m ≈ Go2 limit; no_fatigue 2.5× energy, 35× jerk) → bio knobs are feasibility constraints, not reward devices |
| [phase4-residual-compensation](./phase4-residual-compensation/) | done (2026-05-29) | Simple stance-gated height-PD residual on the frozen policy: small in-envelope payload trend (8 kg +37 %, but two-sided Welch p≈0.07, n.s.), harmless at nominal, capped because the frozen policy treats it as a disturbance |
| [phase5-trainability](./phase5-trainability/) | side note (2026-06-02) | Ran the all-biomech-off case once (5 seeds). Recorded as an observation — our knobs degenerate to scaled raw torque, not necessarily the paper's baseline, so it isn't a clean comparison either way |

## Local-only navigation

On the lab server, `results/raw` is a symlink into the SATA training logs:

```
results/raw/
├── May25_23-12-14_ref_s1/    # seed 1 (phase 1)
├── May25_23-12-17_ref_s2/    # seed 2 (phase 1)
└── May25_23-12-22_ref_s3/    # seed 3 (phase 1)
```

To recreate this symlink on a fresh checkout (lab server only):

```bash
ln -s ~/workspace/SATA/legged_gym/logs results/raw
```
