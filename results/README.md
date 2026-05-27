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
| [phase2-ablation](./phase2-ablation/) | done (2026-05-27) | 5 ablations × 8 seeds. Only `no_fatigue` (+22) and `no_activation` (+24) significant; Hill model & growth curriculum have no detectable effect on training reward |
| phase3-control-perspective | planned | Compare SATA's answers to adaptive-control questions |
| phase4-residual-compensation | optional | Body-height regulation residual under payload (motivated by SATA paper §VI-A) |

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
