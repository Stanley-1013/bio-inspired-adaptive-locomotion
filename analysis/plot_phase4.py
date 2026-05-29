#!/usr/bin/env python3
"""
Phase 4 plot: reward vs payload for reference / no_fatigue / ref_residual,
plus a peak-torque panel showing the residual stays within the actuator
envelope. Merges the Phase 3 CSVs (reference, no_fatigue) with the Phase 4
residual CSV.

Run in the sata conda env: python analysis/plot_phase4.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
P3 = ROOT / "results/phase3-bio-claims-and-robustness"
P4 = ROOT / "results/phase4-residual-compensation"
OUT = P4 / "plots"
CSVS = [P3 / "raw_metrics.csv", P3 / "raw_metrics_suppl.csv", P4 / "raw_residual.csv"]

CONDS = ["reference", "no_fatigue", "ref_residual"]
COLORS = {"reference": "#1f77b4", "no_fatigue": "#2ca02c", "ref_residual": "#d62728"}
MASS = {"nominal": 0, "payload_5kg": 5, "payload_8kg": 8, "payload_10kg": 10}
GO2_PEAK, SIM_CLIP, PRO = 45.0, 23.5, 8.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(c) for c in CSVS], ignore_index=True)
    df = df[df["n_episodes"].astype(float) > 0]

    fig, (axr, axt) = plt.subplots(1, 2, figsize=(14, 5.5))
    axr.axvspan(0, PRO, color="green", alpha=0.06, label="in-spec (≤ Go2 Pro 8 kg)")
    for c in CONDS:
        xs, ys, es = [], [], []
        for sc, m in MASS.items():
            d = df[(df.condition == c) & (df.scenario == sc)]["reward"].astype(float)
            if len(d):
                xs.append(m); ys.append(d.mean()); es.append(d.std())
        axr.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                     color=COLORS[c], label=c)
    axr.set_xlabel("added payload (kg)"); axr.set_ylabel("episode reward (mean ± std, 8 seeds)")
    axr.set_title("Reward vs payload\nref_residual = reference + simple in-envelope height residual")
    axr.legend(fontsize=9); axr.grid(True, alpha=0.3)

    # peak torque panel
    for c in CONDS:
        xs, ys, es = [], [], []
        for sc, m in MASS.items():
            d = df[(df.condition == c) & (df.scenario == sc)]["torque_peak"].astype(float)
            if len(d):
                xs.append(m); ys.append(d.mean()); es.append(d.std())
        axt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                     color=COLORS[c], label=c)
    axt.axhline(GO2_PEAK, ls="--", color="k", lw=1); axt.text(0.2, GO2_PEAK-2, "Go2 peak 45 N·m", fontsize=8)
    axt.axhline(SIM_CLIP, ls=":", color="gray", lw=1); axt.text(0.2, SIM_CLIP+0.5, "sim clip 23.5", fontsize=8)
    axt.set_xlabel("added payload (kg)"); axt.set_ylabel("peak joint torque (N·m)")
    axt.set_title("Peak torque stays within the actuator envelope")
    axt.legend(fontsize=9); axt.grid(True, alpha=0.3); axt.set_ylim(0, 50)

    fig.suptitle("Phase 4 — does a simple classical residual recover §VI-A payload posture, within spec?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "residual_payload.png", dpi=140)
    print("wrote", OUT / "residual_payload.png")


if __name__ == "__main__":
    main()
