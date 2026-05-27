# analysis/

Reusable analysis & plotting scripts for this project. Outputs (figures,
extracted CSVs) land under [`../results/`](../results/) next to the run
they belong to, not here — this directory holds *code only*.

| Script | What it does | Output |
|---|---|---|
| [`plot_phase2_curves.py`](./plot_phase2_curves.py) | Read tensorboard event files from all 6 × 8 Phase 2 runs and plot reward curves | `results/phase2-ablation/plots/reward_curves_*.png` |

Run from inside the `sata` conda env (which has `tbparse` + `matplotlib`):

```bash
source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh
cd ~/workspace/bio-inspired-adaptive-locomotion
python analysis/plot_phase2_curves.py
```

The scripts assume the lab-server layout: tensorboard event files live
under `~/workspace/SATA/legged_gym/logs/SATA/<timestamp>_<run_name>/`.
