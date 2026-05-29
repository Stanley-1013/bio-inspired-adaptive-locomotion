#!/usr/bin/env python3
"""
Phase 2 — plot the training-reward curves of all 48 runs (6 conditions × 8 seeds).

Outputs two PNGs into ../results/phase2-ablation/plots/ :

  reward_curves_overlay.png        — all 6 conditions on one axis; per-condition
                                     mean across seeds with ±1 std shaded band.
  reward_curves_per_condition.png  — 2×3 grid of subplots, one per condition;
                                     every seed shown as a thin line + mean as
                                     a thick line.

Reads tensorboard event files from $HOME/workspace/SATA/legged_gym/logs/SATA/
(run dirs named like  May25_23-12-14_ref_s1  or  May26_21-23-..._no_hill_s4).

Run from the sata conda env:
  source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh
  cd ~/workspace/bio-inspired-adaptive-locomotion
  python analysis/plot_phase2_curves.py
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tbparse import SummaryReader

# ---------- config ----------------------------------------------------------

LOGS_ROOT = Path.home() / "workspace/SATA/legged_gym/logs/SATA"
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results/phase2-ablation/plots"
)
SCALAR_TAG = "Train/mean_reward"

# Order matters for plotting (legend / subplot order).
CONDITIONS = [
    "reference",
    "no_fatigue",
    "no_hill",
    "no_activation",
    "no_growth",
    "hard_terrain",
]
# Run-name prefix per condition (as written by SATA's --run_name).
PREFIXES = {
    "reference": "ref",
    "no_fatigue": "no_fatigue",
    "no_hill": "no_hill",
    "no_activation": "no_activation",
    "no_growth": "no_growth",
    "hard_terrain": "hard_terrain",
}
COLORS = {
    "reference": "#1f77b4",
    "no_fatigue": "#2ca02c",
    "no_hill": "#d62728",
    "no_activation": "#9467bd",
    "no_growth": "#ff7f0e",
    "hard_terrain": "#8c564b",
}
SEEDS = list(range(1, 9))

# ---------- data loading ----------------------------------------------------

def find_run_dir(prefix: str, seed: int) -> Path | None:
    """Return the most recent run dir matching *_<prefix>_s<seed> (or None)."""
    pattern = str(LOGS_ROOT / f"*_{prefix}_s{seed}")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def load_reward_curve(run_dir: Path) -> pd.DataFrame:
    """Return DataFrame with columns ['step', 'reward'] for one run."""
    reader = SummaryReader(str(run_dir), pivot=False)
    df = reader.scalars
    sub = df[df["tag"] == SCALAR_TAG][["step", "value"]].rename(
        columns={"value": "reward"}
    )
    return sub.sort_values("step").reset_index(drop=True)


def load_all() -> dict[str, list[pd.DataFrame]]:
    """Return {condition_label: [df_seed1, df_seed2, ...]}, missing seeds skipped."""
    out: dict[str, list[pd.DataFrame]] = defaultdict(list)
    missing = []
    for cond in CONDITIONS:
        prefix = PREFIXES[cond]
        for seed in SEEDS:
            run_dir = find_run_dir(prefix, seed)
            if run_dir is None:
                missing.append(f"{cond}/s{seed}")
                continue
            try:
                curve = load_reward_curve(run_dir)
                if len(curve) < 100:
                    missing.append(f"{cond}/s{seed} (only {len(curve)} steps)")
                    continue
                out[cond].append(curve)
            except Exception as exc:
                missing.append(f"{cond}/s{seed} (read error: {exc})")
    if missing:
        print("WARNING: missing or short runs:")
        for m in missing:
            print(f"  {m}")
    return out


# ---------- per-condition mean / std ---------------------------------------

def aligned_matrix(curves: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Truncate all curves to the shortest, interpolate to common x grid.
    Returns (steps[N], values[K seeds, N steps])."""
    max_step = min(int(c["step"].max()) for c in curves)
    # All runs should share the same step values (PPO writes once per iter),
    # but defensively interpolate onto an evenly-spaced grid.
    grid = np.arange(0, max_step + 1)
    rows = []
    for c in curves:
        rows.append(np.interp(grid, c["step"].values, c["reward"].values))
    return grid, np.stack(rows, axis=0)


# ---------- plotting --------------------------------------------------------

def plot_overlay(data: dict[str, list[pd.DataFrame]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in CONDITIONS:
        if cond not in data:
            continue
        steps, mat = aligned_matrix(data[cond])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=1)
        ax.plot(steps, mean, label=f"{cond} (n={mat.shape[0]})",
                color=COLORS[cond], linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std,
                        color=COLORS[cond], alpha=0.18, linewidth=0)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Train/mean_reward")
    ax.set_title("Phase 2 ablations — per-condition mean ± std across 8 seeds")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_condition(data: dict[str, list[pd.DataFrame]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, cond in zip(axes, CONDITIONS):
        if cond not in data:
            ax.set_title(f"{cond} (no data)")
            continue
        steps, mat = aligned_matrix(data[cond])
        for row in mat:
            ax.plot(steps, row, color=COLORS[cond], alpha=0.35, linewidth=0.8)
        mean = mat.mean(axis=0)
        ax.plot(steps, mean, color=COLORS[cond], linewidth=2.2,
                label=f"mean (n={mat.shape[0]})")
        ax.set_title(cond)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes[:3]:
        ax.set_xlabel("")
    for ax in axes[3:]:
        ax.set_xlabel("PPO iteration")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Train/mean_reward")
    fig.suptitle("Phase 2 ablations — individual seed curves per condition",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- main ------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading scalars from {LOGS_ROOT}/ (tag = {SCALAR_TAG})")
    data = load_all()
    print()
    print("Per-condition seed counts loaded:")
    for cond in CONDITIONS:
        print(f"  {cond:15s} {len(data.get(cond, []))}")
    print()
    print(f"Writing plots to {OUT_DIR}/")
    plot_overlay(data, OUT_DIR / "reward_curves_overlay.png")
    plot_per_condition(data, OUT_DIR / "reward_curves_per_condition.png")
    print("done.")


if __name__ == "__main__":
    main()
