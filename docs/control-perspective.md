# Control Perspective — SATA through the questions adaptive/robust control asks

This is the project's intellectual spine: we do **not** claim SATA *is* adaptive
control. Instead we let classical adaptive/robust control supply a set of
**questions**, and use our own experimental findings (Phase 1 reproduction,
Phase 2 ablations, Phase 3 feasibility/OOD, Phase 4 residual probe) as evidence
for **how SATA answers them differently** — with learning + biomechanics rather
than analytic control laws. Where our data contradicts a tidy analogy, we say so.

Companion: [`training-internals.md`](./training-internals.md) (what the code
does), [`concepts-primer.md`](./concepts-primer.md) (definitions). Evidence
lives in [`../results/`](../results/).

## The four questions, answered with our data

### Q1. Unknown / varying dynamics (payload) — robust vs adaptive

- **Classical adaptive control (e.g. MRAC):** estimate the unknown parameter
  (here, body mass) *online* and adjust the control law so tracking is
  maintained as the parameter drifts.
- **SATA's answer:** train one *fixed* policy that is *robust over a
  distribution* of masses (the SATA config's domain randomisation is
  `added_mass ∈ [−1, +5] kg`; the paper §IV-B states it as "up to 5 kg").
  No online estimation — a single set of weights expected to cover the set.
- **What our data shows.** This is the crisp robust-vs-adaptive distinction.
  By our fall / episode-length metric the reference holds in-distribution
  payloads and **degrades from 8 kg and beyond** (Phase 3: episode length
  3467→2994→2641 across 5/8/10 kg). This is consistent with — but measured
  differently from — SATA §VI-A, whose stated onset is a *body-height* failure
  already at 5 kg (calves contacting the ground), not necessarily a fall. A fixed robust policy works *inside* the set it was trained on and
  has no mechanism to extend beyond it — exactly the property online adaptive
  control adds. SATA answers Q1 with **robustness, not adaptation**; the §VI-A
  limitation is the signature of that choice, not a bug.

### Q2. Disturbance rejection (push, internal load) — is fatigue a disturbance compensator?

- **Classical:** a disturbance observer / explicit compensation term cancels an
  estimated external disturbance.
- **The README's first-draft analogy:** "fatigue feedback ≈ internal feedback /
  disturbance compensation."
- **What our data shows — the analogy is weak.** Under lateral push, *every*
  condition degrades alike (Phase 3: all hold at 1.5 m/s, all collapse at
  3–5 m/s); ablating fatigue does **not** worsen push rejection. Fatigue's real
  footprint is elsewhere: removing it raises sustained mechanical energy 2.5×
  and action-jerk 35× (Phase 3, p<0.001). So fatigue behaves as a
  **self-protection / effort-regulariser**, not a disturbance rejector. We
  therefore *down-grade* this analogy: fatigue rhymes with actuator
  thermal/effort limiting, not with disturbance-observer compensation.

### Q3. Operating-envelope scheduling (growth) — gain scheduling, or curriculum?

- **Classical gain scheduling:** change controller gains *at runtime* as a
  function of the current operating point.
- **The README's first-draft analogy:** "growth (torque-limit curriculum) ≈
  gain scheduling."
- **What our data shows — analogy is misleading on the time axis.** Growth
  schedules torque-limit/frequency over **training** (keyed to cumulative env
  steps, a Gompertz curve), and is fully saturated (`general_scale=1`) by
  deployment. Phase 2 (8 seeds) found `no_growth` not significantly different
  from reference on reward, and Phase 3 found the *deployed* no_growth policy
  indistinguishable from reference on every behavioural metric. So growth is a
  **training-time curriculum**, not a runtime gain schedule — the two operate on
  different time scales. Correct analogy: growth ↔ curriculum/continuation
  methods, not gain scheduling.

### Q4. Stability / actuator feasibility — analytic constraints vs learned-in constraints

- **Classical:** encode actuator saturation, rate limits, and (ideally)
  Lyapunov stability directly in the controller.
- **SATA's answer:** no formal guarantee; the Hill force–velocity model and the
  activation low-pass act as **implicit actuator-feasibility bounds baked into
  training**, so the learned policy stays inside a hardware-plausible torque
  envelope without an explicit constraint.
- **What our data shows.** Phase 3 is fairly clear here: ablating activation
  lets peak torque reach 42.5 N·m — at the real Go2's 45 N·m limit, ~1.8× the
  sim's 23.5 N·m clip (p<0.001). The bio layer is
  functionally the **saturation / rate-limit a careful classical controller
  would impose** — learned-in rather than written analytically. The cost
  (Phase 2: +20% training reward when removed) is the price of staying feasible,
  exactly as a conservatively-constrained classical controller sacrifices
  nominal performance for envelope safety.

## Q5 (output form) and the Phase 4 hybrid probe

- **Classical** emits an analytic law `τ = f(x, θ̂)`; **SATA** emits a neural
  torque policy `τ = π(o)`. Phase 4 asks the natural follow-up: can we bolt a
  **classical residual** onto the frozen neural policy — `τ_total = τ_SATA +
  τ_comp` — to recover the payload capability the fatigue constraint sacrifices,
  *within* the actuator envelope (unlike the ablation, which exceeds it)?
- **Finding (see [`../results/phase4-residual-compensation/`](../results/phase4-residual-compensation/)).**
  A simple stance-gated PD-on-body-height residual recovers payload stability
  with a few N·m of extra torque — but only with *gentle* gains; strong gains
  fail in *both* sign directions because the **frozen policy cannot observe the
  residual and treats it as an unmodelled disturbance**. The takeaway we draw:
  a bolt-on classical layer and a learned policy are not co-designed, so the
  policy fights the correction. It is precisely the gap that **RL-with-online-
  adaptation** methods close by *learning* the adaptation jointly — e.g.
  **RL2AC** (Reinforcement-Learning-based Rapid Online Adaptive Control, RSS
  2024). Our probe motivates that line without claiming to deliver it.

## Honest summary

SATA answers adaptive control's questions with **robustness + learned-in
feasibility constraints**, not online adaptation or analytic guarantees:
- Q1 unknown dynamics → fixed robust policy (works in-set, fails out-of-set: §VI-A)
- Q2 disturbances → fatigue is effort-regularisation, *not* disturbance compensation (analogy down-graded by our data)
- Q3 envelope → growth is a *training* curriculum, *not* runtime gain scheduling (analogy corrected)
- Q4 feasibility → Hill/activation read as *learned-in* saturation/rate limits — the property our Phase 3 data speaks to most directly
- Q5 output → neural law; a bolt-on classical residual only partly helps because it isn't co-designed (motivates RL+online-adaptation, e.g. RL2AC)

Two of the README's own first-draft analogies (fatigue↔disturbance
compensation, growth↔gain scheduling) seem to us not well supported by the data,
and we have revised them above — which is what we hoped taking the control lens
seriously would surface.

**Caveat (carried from Phase 3):** the feasibility argument is inferred from sim
torque/energy vs the Go2 datasheet; there is no real-robot or thermal-model
validation here. "Stays within the actuator envelope" means "within the rated
torque number", not "verified on hardware".
