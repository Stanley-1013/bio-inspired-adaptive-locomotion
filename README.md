# Bio-inspired Adaptive Locomotion via Torque-based Learning

**A Case Study of SATA**

*A simulation-based study using adaptive and robust control questions as a lens.*

This project reproduces and studies **SATA** (Safe and Adaptive Torque-Based
Locomotion Policies Inspired by Animal Learning; Li et al., RSS 2025).

Rather than proposing a new controller, this work investigates how bio-inspired
torque control creates adaptive locomotion behaviors, and discusses these
mechanisms through **adaptive and robust control questions**.

Official implementation: https://github.com/marmotlab/SATA

This repository is intended for educational and research purposes.

**Keywords:** Embodied AI · Adaptive Control · RL · Torque Control · Locomotion

---

## Overview

This project takes SATA as its main subject. We first **reproduce** its
torque-based locomotion pipeline in simulation, then use **ablation
experiments** to observe how the biomechanical model, fatigue feedback, and
growth mechanism shape adaptive behavior. Finally — and this is the framing that
matters — we do **not** force SATA into classical control theory. Instead we let
**adaptive / robust control supply the questions** (how to stay stable under
unknown dynamics, reject disturbances, generalize) and observe **how SATA
answers them differently** — with learning and biomechanics rather than analytic
control laws. If time allows, we further test a simple **residual compensation**
term in boundary cases.

> Framing: the goal is **not** to claim SATA *is* adaptive control. SATA is a
> learning-based policy with no Lyapunov-based update law or online parameter
> estimation. We instead **compare two philosophies of handling adaptation** —
> classical adaptive/robust control vs. learning + bio-inspired mechanisms —
> using the former's questions as a lens.

## Repository

```
docs/   progress-report deck (pptx + pdf), design brief, SATA setup guide
scripts/  setup + deck-build scripts, and the pptxgenjs deck generator
.claude/  SessionStart hook (auto-runs scripts/setup.sh on web sessions)
```

- **Progress report:** `docs/20260525_progress_report_v1.pptx` / `.pdf`
  (spec in `docs/progress-report-design-brief.md`).
- **Reproduce SATA (needs a GPU box):** see `docs/setup-sata.md`.
- **Rebuild the deck:** `bash scripts/setup.sh` then `bash scripts/build-deck.sh`.
  (Deck build uses pptxgenjs + LibreOffice; SATA itself does **not** run in the
  Claude Code web sandbox — no GPU / Isaac Gym.)

## 1. Motivation & Problem

Position control struggles on unknown and unstructured terrains because of its
rigidity and limited adaptability. SATA addresses this by learning **torque
policies** and incorporating **bio-inspired mechanisms** (muscle activation,
fatigue, growth) to achieve safe and adaptive locomotion.

## 2. Purpose

- Reproduce SATA in simulation (Isaac Gym + Unitree Go2).
- Analyze how the bio-inspired designs produce adaptive behaviors.
- Compare how SATA and classical adaptive / robust control address the **same
  adaptation questions** (unknown dynamics, disturbances, generalization).
- *(Optional, if time allows)* Explore a lightweight **residual compensation**
  term to handle boundary cases.

**Key contribution:** **contrast two philosophies of adaptation** —
learning-based locomotion vs. classical adaptive/robust control — through
understanding and experiments.

## 3. System Overview (SATA)

```
Command (velocity, direction, heading)
        │
        ▼
   Torque Policy ◄──────────────┐
        │                       │
        ▼                       │
 Biomechanical Layer            │ State Feedback
  • Activation (NN model)       │ (base states,
  • Fatigue feedback            │  fatigue states,
  • Torque limit (Growth)       │  termination)
        │                       │
        ▼                       │
   Torque Output                │
        │                       │
        ▼                       │
 Robot Simulator (Go2) ─────────┘
```

The policy is trained with RL; the biomechanical layer adds bio-inspired
adaptation between the learned policy and the simulated robot.

## 4. Research Questions

1. **Why torque?** Why is torque-based locomotion more suitable than
   position-based locomotion for achieving compliance and generalization?
2. **Which components?** Which components in SATA contribute to adaptive
   behaviors (muscle activation, fatigue feedback, growth mechanism, torque
   scheduling)?
3. **Shared problems?** How does SATA address problems *traditionally studied in
   adaptive control* (stability under unknown dynamics, disturbance rejection,
   generalization) — and how does its learning-based answer differ from an
   analytic control law?
4. **Residual compensation?** Could a simple residual compensation term improve
   performance under extreme conditions *without* modifying the RL policy?

## 5. Methodology & Work Plan

| Phase | Focus | Key activities | Output |
|-------|-------|----------------|--------|
| **1. Reproduction** | Set up the pipeline | Set up server/container; install Isaac Gym + SATA; run `go2_torque` training; verify play in simulation | Working simulation pipeline |
| **2. Observation & Ablation** | Modify baseline and compare | Toggle fatigue model; change torque limit; modify growth schedule; test different terrains | Behavioral insights & data |
| **3. Control Perspective** | Use adaptive-control questions as a lens; compare answers | See tables below | Comparison & discussion |
| **4. Residual Compensation** *(optional, if time allows)* | Explore a lightweight residual term | `τ_total = τ_SATA + τ_comp` to compensate disturbances (friction, payload, terrain change) | Preliminary observations in boundary cases |

**Phase 3 — adaptive control supplies the *questions*; SATA gives a different
kind of *answer*:**

| Adaptation question | Classical adaptive / robust control | SATA (learning + biomechanics) |
|---------------------|-------------------------------------|--------------------------------|
| Unknown dynamics | online parameter estimation | learns a robust policy in sim |
| Disturbances | explicit compensation term | internal state (fatigue) feedback |
| Generalization | stability proofs / guarantees | empirical, across terrains |
| Control output | analytic control law | neural torque policy |

We also note where SATA's mechanisms *rhyme with* adaptive-control vocabulary —
a lens for discussion, not equivalence:

| SATA mechanism | Adaptive-control idea it echoes |
|----------------|---------------------------------|
| Growth (torque limit / curriculum) | gain scheduling |
| Fatigue feedback | internal feedback |
| Torque modulation | robustness / compliance |

*SATA has no Lyapunov-based update law or online parameter estimation — these
are framing lenses, not formal equivalences.*

## 6. Expected Outcomes

- Reproduce and understand the SATA framework in simulation.
- Explain how bio-inspired mechanisms lead to adaptive locomotion.
- Contrast how SATA and classical adaptive/robust control answer the same
  adaptation questions, being clear about where the framings align and diverge.
- Provide a baseline for future work on residual/robust compensation and
  real-world studies.

## 7. Experiment Setup

| Item | Configuration |
|------|---------------|
| Robot | Unitree Go2 (simulation-based) |
| Environment | Isaac Gym Preview 4 |
| Task | `go2_torque` |
| Algorithm | PPO (`rsl_rl`) |
| Training | ~20 min, 3000 iterations, 4096 envs (reference HW: RTX 4090) |
| Evaluation | Velocity tracking, rough terrain, disturbance robustness, energy efficiency, stability, sim-to-real generalization |

## 8. Key References

1. SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning (RSS 2025). [arXiv:2502.12674](https://arxiv.org/abs/2502.12674)
2. RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion (RSS 2024). [Proceedings](https://www.roboticsproceedings.org/rss20/p060.html)
3. Hwangbo et al., Learning agile and dynamic motor skills for legged robots (Science Robotics, 2019). [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
4. Lee et al., Learning quadrupedal locomotion over challenging terrain (Science Robotics, 2020). [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)
5. Miki et al., Learning robust perceptive locomotion for quadrupedal robots in the wild (Science Robotics, 2022). [arXiv:2201.08117](https://arxiv.org/abs/2201.08117)
6. Chen et al., Learning Torque Control for Quadrupedal Locomotion (2022). [arXiv:2203.05194](https://arxiv.org/abs/2203.05194)
7. DecAP: Decaying Action Priors for Accelerated Imitation Learning of Torque-Based Legged Locomotion Policies (IROS 2024). [arXiv:2310.05714](https://arxiv.org/abs/2310.05714)
8. Hill, A. V., The heat of shortening and the dynamic constants of muscle (Proc. R. Soc. B, 1938). [doi:10.1098/rspb.1938.0050](https://royalsocietypublishing.org/doi/10.1098/rspb.1938.0050)
9. Muscle fatigue literature — e.g., Liu, Brown & Yue, A Dynamical Model of Muscle Activation, Fatigue, and Recovery (Biophysical Journal, 2002). [PMC1302027](https://pmc.ncbi.nlm.nih.gov/articles/PMC1302027/)
10. CPG-RL / learning-based hierarchical control literature — e.g., Bellegarda & Ijspeert, CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion (RA-L, 2022). [arXiv:2211.00458](https://arxiv.org/abs/2211.00458)

---

> **Takeaway:** SATA shows that combining torque control with bio-inspired
> adaptation yields compliant, generalizable locomotion. Rather than forcing it
> into classical control theory, we use **adaptive control's questions as a
> lens** and ask how SATA answers them differently — contrasting two
> philosophies of adaptation (analytic control vs. learning + biomechanics), and
> optionally probing a simple residual compensation.
