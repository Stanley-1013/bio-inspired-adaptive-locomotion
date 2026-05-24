# Progress Report — Presentation Design Brief (3 min)

> This is a **design specification**, not slide content. It defines what the
> presentation must communicate and how it should look, so that Claude Code +
> a presentation skill can reliably generate the slides in a later step.

---

## Project

**Title:** Bio-inspired Adaptive Locomotion via Torque-based Learning: A Case Study of SATA

**Subtitle (optional):** A simulation-based study from an adaptive control perspective

---

## Overall Goal

Generate a concise **academic progress presentation** for an **Adaptive Control
Systems** course. In 3 minutes it must convince the instructor that:

1. The problem is clearly defined.
2. The topic fits the course.
3. There is an executable plan.
4. This is **not** a pure AI paper review.

The presentation should communicate:

1. A meaningful **control problem**.
2. **Why** SATA is chosen.
3. The planned **simulation methodology**.
4. How **adaptive control concepts** will be used to *interpret* — not replace —
   the learning framework.

**Tone guardrails**

- Avoid overselling novelty.
- Do **not** claim SATA is classical adaptive control.
- Focus on **understanding adaptive behavior**.

---

## Slide Count & Pacing

Target: **4 slides (~3 minutes)**

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Motivation & Problem Definition | 40 sec |
| 2 | SATA Overview | 60 sec |
| 3 | Project Plan | 50 sec |
| 4 | Expected Contribution | 30 sec |

---

## Slide 1 — Motivation & Problem Definition

**Goal:** Convince the audience this is fundamentally a **control problem**, not
an RL presentation.

**Key message:** Position-based locomotion performs well under known conditions
but struggles with **compliance, disturbance handling, unknown terrain, and
sim-to-real robustness**. This motivates **torque-based locomotion**.

**Layout**

- **LEFT:** Problem statement.
- **RIGHT:** Simple conceptual comparison.

  ```
  Position Control                 Torque Control
  command position                 command force
        ↓                                ↓
   rigid behavior                 compliant interaction
        ↓                                ↓
   poor adaptation                 adaptive response
  ```

- **BOTTOM — Research Question:**
  > How can robots achieve safer and more adaptive locomotion in unknown
  > environments?

**Background knowledge to convey briefly**

- *Position control:* policy outputs target joint angle; low-level PD converts to torque.
- *Torque control:* policy outputs torque directly.
- *Compliance:* robot yields instead of resisting.
- *Generalization:* works outside the training distribution.

**References (this slide):** SATA (main); Chen et al. (Learning Torque Control);
Lee et al. (Challenging Terrain); Miki et al. (Perceptive Locomotion in the Wild).

---

## Slide 2 — SATA Overview

**Goal:** Explain *what SATA contributes*. **Do not** explain equations.

**Key message:** SATA introduces **bio-inspired adaptation** into torque-based
locomotion.

**Layout — CENTER flow:**

```
Observation
    ↓
Torque Policy (RL)
    ↓
Biomechanical Layer
    ↓
Torque Output
    ↓
  Robot
    ↑
Fatigue Feedback  (loops back)
```

**Highlight two blocks**

- **Biomechanical Model:** activation · muscle · fatigue.
- **Growth Mechanism:** torque limit · reward · frequency.

**Background knowledge to convey**

- *Activation:* smooth actuator command.
- *Muscle:* prevent abrupt torque.
- *Fatigue:* avoid overusing joints.
- *Growth:* progressively unlock capability.
- *Zero-shot sim-to-real:* deploy without finetuning.

**References (this slide):** SATA; Hill (muscle dynamics); Liu et al. (activation,
fatigue, recovery); Bellegarda & Ijspeert (CPG-RL).

---

## Slide 3 — Project Plan

**Goal:** Demonstrate feasibility; show this is not only a literature review.

**Key message:** **Understand → Reproduce → Analyze** — not redesign RL.

**Layout — timeline:**

```
Phase 1            Phase 2            Phase 3                 (Optional)
Reproduce SATA  →  Ablation        →  Adaptive Interpretation → Residual Compensation
```

**Per-phase detail**

- **Phase 1 — Reproduce SATA**
  - Environment: CUDA server · container · venv · Isaac Gym.
  - Output: simulation running.
- **Phase 2 — Ablation**
  - Disable fatigue · change torque limit · modify growth schedule · terrain variation.
- **Phase 3 — Adaptive Interpretation**
  - Growth → gain scheduling.
  - Fatigue → feedback.
  - Torque limit → adaptive constraint.
- **Optional — Residual Compensation**
  - `τ_total = τ_SATA + τ_comp`
  - Only if feasible.

**References (this slide):** SATA; RL2AC; DecAP.

---

## Slide 4 — Expected Contribution

**Goal:** Close the story; avoid exaggerated claims.

**Key message:** This work **studies adaptive behavior** rather than proposing a
new RL algorithm.

**Layout**

- **LEFT — Expected Outputs:**
  - Simulation reproduction.
  - Behavior analysis.
  - Adaptive interpretation.
- **RIGHT — Research Questions:**
  - RQ1: Why torque control?
  - RQ2: What creates adaptation?
  - RQ3: How does it relate to adaptive control?
  - RQ4: Can lightweight compensation help?
- **BOTTOM — one-line conclusion.**

**Suggested closing sentence:**
> This project investigates how bio-inspired torque control creates adaptive
> locomotion behaviors and interprets these mechanisms through an adaptive
> control perspective.

---

## Reference Policy

- **Slides:** 5–6 references max.
- **Final report:** 10 references.
- Cite **only where a concept is introduced**.
- Do **not** fill slides with a bibliography.

---

## Visual Principles

**Prefer:** diagrams · arrows · conceptual comparison.

**Avoid:** equations · architecture screenshots · large tables · RL training curves.

**If a figure is used:** annotate the takeaway. Never show figures without
interpretation.
