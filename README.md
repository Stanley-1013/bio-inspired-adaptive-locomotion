# Bio-inspired Adaptive Locomotion via Torque-based Learning: A Case Study of SATA

*A simulation-based analysis from an adaptive control perspective.*

This project reproduces and analyzes SATA:

Li et al.,
"Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning"

Official implementation: https://github.com/marmotlab/SATA

This repository is intended for educational and research purposes.

---

## 1. Motivation & Problem

Position control struggles on unknown and unstructured terrains because of its
rigidity and limited adaptability. SATA addresses this by learning **torque
policies** and incorporating **bio-inspired mechanisms** (muscle activation,
fatigue, growth) to achieve safe and adaptive locomotion.

## 2. Purpose

- Reproduce SATA in simulation (Isaac Gym + Unitree Go2).
- Analyze how the bio-inspired designs produce adaptive behaviors.
- Interpret SATA from an **adaptive control** perspective.
- *(Optional)* Design a simple robust compensation to handle boundary cases.

**Key contribution:** bridge learning-based locomotion and adaptive/robust
control theory through understanding and experiments.

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
3. **Control interpretation?** How can these adaptation mechanisms be
   interpreted using adaptive control concepts (gain scheduling, internal
   feedback, disturbance compensation, constraint adaptation)?
4. **Robust compensation?** Can a simple robust adaptive compensation improve
   performance under extreme conditions *without* modifying the RL policy?

## 5. Methodology & Work Plan

| Phase | Focus | Key activities | Output |
|-------|-------|----------------|--------|
| **1. Reproduction** | Set up the pipeline | Set up server/container; install Isaac Gym + SATA; run `go2_torque` training; verify play in simulation | Working simulation pipeline |
| **2. Observation & Ablation** | Modify baseline and compare | Toggle fatigue model; change torque limit; modify growth schedule; test different terrains | Behavioral insights & data |
| **3. Adaptive Control Interpretation** | Map mechanisms to theory | See mapping table below | Theoretical analysis & discussion |
| **4. Robust Compensation** *(optional)* | Add lightweight compensation | `T_final = T_SATA + T_comp` to compensate disturbances (friction, payload, terrain change) | Improved robustness in boundary cases |

**Phase 3 mapping — SATA mechanism → adaptive control view:**

| SATA mechanism | Adaptive control view |
|----------------|-----------------------|
| Activation | Gain scheduling |
| Fatigue feedback | Internal feedback |
| Growth limit | Constraint adaptation |
| Torque control | Compliance |

## 6. Expected Outcomes

- Reproduce and understand the SATA framework in simulation.
- Explain how bio-inspired mechanisms lead to adaptive locomotion.
- Establish connections between the learning-based approach and adaptive/robust
  control theory.
- Provide a baseline for future work on robust adaptive compensation and
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

1. **SATA** — Safe and Adaptive Torque-based Locomotion Policies Inspired by
   Animal Learning (RSS 2025).
2. Reinforcement learning for adaptive control of legged robots.
3. Hwangbo et al. — Learning agile and dynamic motor skills for legged robots.
4. Lee et al. — Learning quadrupedal locomotion over challenging terrain.
5. Miki et al. — Learning robust perceptive locomotion for quadrupedal robots
   in the wild.
6. Chen et al. — Learning torque control for quadrupedal locomotion.
7. Decaying action priors for torque-based legged locomotion.
8. Hill, A. V. — The heat of shortening and the dynamic constants of muscle.
9. Muscle fatigue modeling literature.
10. CPG-RL — Learning-based hierarchical control of locomotion.

---

> **Takeaway:** SATA demonstrates that combining torque control with
> bio-inspired adaptation enables safe, adaptive, and generalizable locomotion.
> By understanding it through an adaptive control lens and exploring robust
> compensation, we aim to bridge learning-based methods and classical control
> theory.
