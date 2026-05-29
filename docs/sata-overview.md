# SATA — what the paper contributes, and why it is worth reproducing

This is the conceptual front door to the project: what **SATA** (Safe and
Adaptive Torque-based locomotion, Li et al., MARMot Lab @ NUS, RSS 2025;
[arXiv:2502.12674](https://arxiv.org/abs/2502.12674)) actually proposes, why its
ideas are elegant, and what it honestly does not yet solve. Every section ends
with **→ in this repo**, pointing to where our experiments verify, deepen, or
probe that idea — so the rest of the repo reads as "understand & validate the
contribution", not "poke holes in it".

Companion docs: [`training-internals.md`](./training-internals.md) (the code-level
mechanism), [`control-perspective.md`](./control-perspective.md) (a control-theoretic
reading of our findings), [`concepts-primer.md`](./concepts-primer.md) (term definitions).

## The problem SATA attacks

Most RL locomotion controllers output **joint positions** (a PD controller then
turns position error into torque). Position control is easy to train but
**rigid**: it reacts to "track the commanded angle" even when the world says
otherwise, which limits compliance and dynamic behaviour and can over-react to
disturbances (a stuck leg gets forced toward its target).

> "Most existing RL-based methods rely on position control … this simplicity
> limits the policy's capacity to explore fine-grained and dynamic behaviors."

**Torque control** is the compliant alternative — command joint torque directly —
but it is hard to learn: the action space is highly non-linear and exploration
is inefficient, so prior torque-RL work was brittle and hyperparameter-sensitive.
SATA's thesis: don't brute-force it — **shape the torque action space with
biomechanics** so it is both easier to learn and physically plausible.

## The three ideas (and why each is elegant)

### 1. A biomechanical model as a "physical-intuition filter"
Between the policy's raw output and the actuator, SATA inserts muscle-inspired
layers: an **activation low-pass** (recruitment delay; no bang-bang torque) and
a **Hill force–velocity model** (max producible torque drops as the joint moves
fast in the same direction). The effect is that the learned torque is smooth and
stays in a biologically/physically plausible range.

> "The biomechanical model ensures biologically plausible and stable
> locomotion … preventing abrupt changes that could destabilize the system."

**→ in this repo:** [Phase 3](../results/phase3-bio-claims-and-robustness/)
measures exactly this. Ablating the activation layer lets peak torque jump to
**42.5 N·m** — right at the real Go2's 45 N·m actuator limit (~1.8× the sim's own
23.5 N·m clip). So the layer's measurable function is to keep the policy inside a
hardware-realisable torque envelope. The biomechanical model is, in control
terms, a **learned-in saturation/rate limit** (see control-perspective Q4).

### 2. A motor-fatigue feedback state
Each joint carries a leaky-integrator "fatigue" state (accumulates with torque,
decays over time), fed back into the observation and lightly penalised. The robot
can therefore *sense its own exertion* and learn to shed sustained high loads.

> "[fatigue] provides the neural network with a dynamic feedback signal …
> preventing certain actuators from operating under prolonged high loads."

**→ in this repo:** [Phase 2](../results/phase2-ablation/) shows the penalty is
deliberately tiny (−0.05) and removing it *raises* training reward; [Phase 3](../results/phase3-bio-claims-and-robustness/)
shows why that is the *expected* sign of a working safety constraint — without
fatigue the policy uses **2.5× the mechanical energy and 35× the action jerk**
at nominal walking. Fatigue trades a little clean-sim reward for not thrashing
the motors. (Our data also *corrects* a tempting analogy: fatigue is an
effort-regulariser, **not** a disturbance compensator — push hits every condition
alike; see control-perspective Q2.)

### 3. A growth mechanism (developmental curriculum)
The cleverest design. Borrowing from animal development, a single Gompertz curve
**progressively unlocks the robot's hardware** over training — torque ceiling
0.3→1.0, control frequency 100→200 Hz — while simultaneously reshaping the reward
weights. Early training happens in an "infant" body that is easy to control,
avoiding the local optima that trap torque-RL from a cold start.

> "…a biologically inspired growth mechanism that mimics animal development by
> progressively unlocking the robot's physical capabilities, dynamically
> adapting reward functions, and gradually increasing control frequency."

**→ in this repo:** the growth curriculum is a **training-dynamics** device, and
our experiments are careful about that. [Phase 2](../results/phase2-ablation/)
finds the *deployed* `no_growth` policy not significantly different from the
reference on reward, and [Phase 3](../results/phase3-bio-claims-and-robustness/)
finds it indistinguishable on every behavioural metric — because by deployment
the curriculum has saturated. That is consistent with growth helping *how the
policy is found*, not *what the final policy does*. (It also means the
"growth ↔ runtime gain scheduling" analogy is a time-scale mismatch — corrected
in control-perspective Q3.)

## The headline result: zero-shot sim-to-real compliance

SATA's strongest empirical claim is **zero-shot transfer** to a physical Unitree
Go2 — no hardware fine-tuning — with striking **passive compliance**: it walks on
slippery and soft terrain, absorbs pushes and weight presses, and recovers
balance even with one leg's torque artificially limited.

> "…our approach achieved zero-shot transfer without any fine-tuning and
> demonstrated highly stable operation over extended deployment periods."

**→ in this repo (scope honesty):** we **reproduce the simulation side only** —
no physical robot was available. Our entire feasibility argument (torque/energy
vs the Go2 datasheet) is the *sim-side rationale* for why such compliance and
transfer are achievable, not a hardware reproduction of them. This boundary is
stated wherever it matters and is the single biggest caveat of the project.

## The honest limitations (which become research openings)

The paper is candid about what SATA does **not** yet do — and these are precisely
the interesting next questions:

- **Weak posture under load.** High compliance means that carrying even a medium
  (~5 kg) payload makes it hard to hold body height; the calves can drop toward
  the ground.
  **→ in this repo:** [Phase 3](../results/phase3-bio-claims-and-robustness/)
  reproduces this cleanly (reference begins to fall at the in-spec 8 kg), and
  [Phase 4](../results/phase4-residual-compensation/) probes a fix — a simple
  classical height residual bolted on the frozen policy recovers some capability
  *within* the actuator envelope, but only modestly, because the frozen policy
  treats the add-on as a disturbance. That limitation motivates **co-trained**
  RL+adaptation (e.g. RL2AC, RSS 2024).
- **Single gait.** SATA learns only a basic walking gait; at high speed it just
  lengthens its stride rather than emerging trot/gallop as some position-based
  methods do.
  > "…our torque-based policy has so far only successfully learned a basic
  > walking gait."

  **→ in this repo:** not addressed; flagged as future work. A natural target for
  a follow-on project on a modern engine.

## Why reproducing SATA is worthwhile (not just "running it again")

SATA sits at the intersection of **embodied AI, RL, and classical control**: it
is a concrete, working example of replacing analytic control structure with
*learned-in, biologically-motivated* structure. Reproducing it gives a clean
platform to ask **which parts of that structure actually carry the weight**, and
on **which axes** (training reward vs hardware feasibility vs out-of-distribution
robustness). That is exactly what Phases 1–4 do, and the
[control-perspective](./control-perspective.md) synthesis ties the answers back
to the questions adaptive/robust control has always asked.
