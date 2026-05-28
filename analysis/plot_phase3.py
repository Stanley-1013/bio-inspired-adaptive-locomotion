#!/usr/bin/env python3
"""
Phase 3 plots: merge the two eval CSVs and produce
  1. payload_degradation.png   — reward vs added payload, per condition,
                                  with in-spec (<=8 kg Go2 Pro) vs beyond-spec shading
  2. nominal_feasibility.png    — per-condition peak torque / energy / action jerk
                                  at nominal, against real-hardware reference lines
  3. push_degradation.png       — reward vs lateral push magnitude, per condition

Run inside the sata conda env (needs pandas + matplotlib):
  python analysis/plot_phase3.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PHASE3 = Path(__file__).resolve().parent.parent / "results/phase3-bio-claims-and-robustness"
OUT = PHASE3 / "plots"
CSVS = [PHASE3 / "raw_metrics.csv", PHASE3 / "raw_metrics_suppl.csv"]

CONDS = ["reference", "no_fatigue", "no_hill", "no_activation", "no_growth", "hard_terrain"]
COLORS = {
    "reference": "#1f77b4", "no_fatigue": "#2ca02c", "no_hill": "#d62728",
    "no_activation": "#9467bd", "no_growth": "#ff7f0e", "hard_terrain": "#8c564b",
}

# Real Unitree Go2: payload Air 7 / Pro 8 / EDU 12 kg; peak joint torque 45 N·m.
# SATA sim uses a conservative 23.5 N·m torque clip.
GO2_PRO_PAYLOAD = 8.0
SIM_TORQUE_CLIP = 23.5
GO2_PEAK_TORQUE = 45.0


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(c) for c in CSVS], ignore_index=True)
    df = df[df["n_episodes"].astype(float) > 0]  # drop any error rows
    return df


def agg(df, scenario_filter):
    """Return per-(condition,scenario) mean+std over seeds for given scenarios."""
    sub = df[df["scenario"].isin(scenario_filter)]
    g = sub.groupby(["condition", "scenario"])
    return g.agg(["mean", "std"])


def plot_payload(df):
    masses = {"nominal": 0, "payload_5kg": 5, "payload_8kg": 8,
              "payload_10kg": 10, "payload_15kg": 15}
    fig, ax = plt.subplots(figsize=(9, 6))
    # shading: in-spec (<=8 kg) vs beyond-spec
    ax.axvspan(0, GO2_PRO_PAYLOAD, color="green", alpha=0.06, label="in-spec (≤ Go2 Pro 8 kg)")
    ax.axvspan(GO2_PRO_PAYLOAD, 15, color="red", alpha=0.06, label="beyond rated payload")
    for c in CONDS:
        xs, ys, es = [], [], []
        for sc, mkg in masses.items():
            d = df[(df.condition == c) & (df.scenario == sc)]["reward"].astype(float)
            if len(d):
                xs.append(mkg); ys.append(d.mean()); es.append(d.std())
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=COLORS[c],
                    label=c, linewidth=1.8)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("added payload (kg)")
    ax.set_ylabel("episode reward (mean ± std, 8 seeds)")
    ax.set_title("Phase 3 — reward vs payload\n(fixed forward command, 20 s episodes)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "payload_degradation.png", dpi=140)
    plt.close(fig)
    print("wrote payload_degradation.png")


def plot_push(df):
    mags = {"nominal": 0, "push_x_1p5": 1.5, "push_x_3": 3.0, "push_x_5": 5.0}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(0, 1.5, color="green", alpha=0.06, label="≤ training push (1.5 m/s)")
    for c in CONDS:
        xs, ys, es = [], [], []
        for sc, mg in mags.items():
            d = df[(df.condition == c) & (df.scenario == sc)]["reward"].astype(float)
            if len(d):
                xs.append(mg); ys.append(d.mean()); es.append(d.std())
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=COLORS[c],
                    label=c, linewidth=1.8)
    ax.set_xlabel("lateral push velocity (m/s, applied every 2-4 s)")
    ax.set_ylabel("episode reward (mean ± std, 8 seeds)")
    ax.set_title("Phase 3 — reward vs lateral push")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "push_degradation.png", dpi=140)
    plt.close(fig)
    print("wrote push_degradation.png")


def plot_feasibility(df):
    nom = df[df.scenario == "nominal"]
    metrics = [
        ("torque_peak", "peak joint torque (N·m)", [SIM_TORQUE_CLIP, GO2_PEAK_TORQUE],
         ["sim clip 23.5", "Go2 peak 45"]),
        ("energy_per_step", "mech. energy per step (Σ|τ·ω|·dt)", [], []),
        ("mean_jerk", "action jerk |Δa|/dt (log scale)", [], []),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, ylabel, hlines, hlabels) in zip(axes, metrics):
        means = [nom[nom.condition == c][metric].astype(float).mean() for c in CONDS]
        stds = [nom[nom.condition == c][metric].astype(float).std() for c in CONDS]
        bars = ax.bar(range(len(CONDS)), means, yerr=stds, capsize=3,
                      color=[COLORS[c] for c in CONDS])
        for y, lab in zip(hlines, hlabels):
            ax.axhline(y, ls="--", color="k", lw=1)
            ax.text(len(CONDS) - 0.4, y, lab, va="bottom", ha="right", fontsize=8)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels(CONDS, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if metric == "mean_jerk":
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase 3 — nominal-walking feasibility metrics "
                 "(ablations that score higher reward exceed hardware-realisable regimes)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "nominal_feasibility.png", dpi=140)
    plt.close(fig)
    print("wrote nominal_feasibility.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"loaded {len(df)} rows")
    plot_payload(df)
    plot_push(df)
    plot_feasibility(df)
    print("done.")


if __name__ == "__main__":
    main()
