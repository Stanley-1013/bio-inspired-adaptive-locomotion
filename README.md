# Bio-inspired Adaptive Locomotion via Torque-based Learning: A Case Study of SATA

*A simulation-based analysis from an adaptive control perspective.*

This project reproduces and analyzes SATA:

Li et al.,
"Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning"

Official implementation: https://github.com/marmotlab/SATA

This repository is intended for educational and research purposes.

---

## Overview

This project takes SATA as its main subject. We first **reproduce** its
torque-based locomotion pipeline in simulation, then use **ablation
experiments** to observe how the biomechanical model, fatigue feedback, and
growth mechanism shape adaptive behavior. We then **discuss** its relationship
to traditional control concepts from an *adaptive / robust control*
perspective. If time allows, we further test a simple **residual compensation**
term to improve behavior in boundary cases.

> Note: SATA is a learning-based policy, not a classical adaptive controller —
> it has no Lyapunov-based update law or online parameter estimation. The
> adaptive/robust control concepts below are used as a **language for analysis
> and comparison**, not as claims of formal equivalence.

## 1. Motivation & Problem

Position control struggles on unknown and unstructured terrains because of its
rigidity and limited adaptability. SATA addresses this by learning **torque
policies** and incorporating **bio-inspired mechanisms** (muscle activation,
fatigue, growth) to achieve safe and adaptive locomotion.

## 2. Purpose

- Reproduce SATA in simulation (Isaac Gym + Unitree Go2).
- Analyze how the bio-inspired designs produce adaptive behaviors.
- Discuss and compare SATA using the language of **adaptive / robust control**.
- *(Optional, if time allows)* Explore a lightweight **residual compensation**
  term to handle boundary cases.

**Key contribution:** relate learning-based locomotion to adaptive/robust
control ideas through understanding and experiments.

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
3. **Control analogy?** How can these adaptation mechanisms be *discussed and
   compared* using adaptive/robust control language (gain scheduling, internal
   feedback, disturbance compensation, constraint adaptation) — and where do
   the analogies break down?
4. **Residual compensation?** Could a simple residual compensation term improve
   performance under extreme conditions *without* modifying the RL policy?

## 5. Methodology & Work Plan

| Phase | Focus | Key activities | Output |
|-------|-------|----------------|--------|
| **1. Reproduction** | Set up the pipeline | Set up server/container; install Isaac Gym + SATA; run `go2_torque` training; verify play in simulation | Working simulation pipeline |
| **2. Observation & Ablation** | Modify baseline and compare | Toggle fatigue model; change torque limit; modify growth schedule; test different terrains | Behavioral insights & data |
| **3. Adaptive Control Analysis** | Compare mechanisms with control concepts | See mapping table below | Analysis & discussion |
| **4. Residual Compensation** *(optional, if time allows)* | Explore a lightweight residual term | `T_final = T_SATA + T_residual` to compensate disturbances (friction, payload, terrain change) | Preliminary robustness test in boundary cases |

**Phase 3 mapping — SATA mechanism → adaptive/robust control analogy:**

| SATA mechanism | Adaptive/robust control analogy |
|----------------|---------------------------------|
| Activation | Gain scheduling |
| Fatigue feedback | Internal feedback |
| Growth limit | Constraint adaptation |
| Torque control | Compliance |

*These are analogies for discussion, not formal equivalences — SATA has no
Lyapunov-based update law or online parameter estimation.*

## 6. Expected Outcomes

- Reproduce and understand the SATA framework in simulation.
- Explain how bio-inspired mechanisms lead to adaptive locomotion.
- Relate the learning-based approach to adaptive/robust control ideas, while
  being clear about where the analogies hold and where they break down.
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

1. SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning (RSS 2025)
2. RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion
3. Hwangbo et al., Learning agile and dynamic motor skills for legged robots
4. Lee et al., Learning quadrupedal locomotion over challenging terrain
5. Miki et al., Learning robust perceptive locomotion for quadrupedal robots in the wild
6. Chen et al., Learning Torque Control for Quadrupedal Locomotion
7. DeCAP: Decaying Action Priors for Torque-Based Legged Locomotion
8. Hill, A. V., The heat of shortening and the dynamic constants of muscle
9. Muscle fatigue literature
10. CPG-RL / Learning-based hierarchical control literature

---

> **Takeaway:** SATA demonstrates that combining torque control with
> bio-inspired adaptation enables safe, adaptive, and generalizable locomotion.
> By analyzing it through an adaptive/robust control lens — and, if time allows,
> exploring a simple residual compensation — we aim to relate learning-based
> methods to classical control ideas.
