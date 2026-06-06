# Phase 5 — A trainability side note

**Status: a small follow-up, not a headline, and not a comparison against the
paper's claim.** While re-reading the SATA paper
([arXiv:2502.12674](https://arxiv.org/abs/2502.12674), §V-A1), we noticed its
ablation framing is about *gait-quality trainability* — the paper reports that
"SATA w/o biomechanical model" is "completely unable to learn a coherent gait,
instead learning to shift its feet on the floor asymmetrically." Our Phase 2
ablations had only ever removed one biomechanical knob at a time, so out of
curiosity we ran the all-three-off case once. We did not build the gait-quality
measurement the paper's claim would need, so this stays a side observation.

## What we ran

`go2_torque_no_biomech` — activation, Hill, and fatigue all off at once, growth
kept on (config in
[`../phase2-ablation/configs/go2_torque_ablations.py`](../phase2-ablation/configs/go2_torque_ablations.py)),
5 seeds, same 3000 iters / 4096 envs as the reference.

**Important framing first.** The paper's claim concerns *gait quality* — a no-bio
policy "learning to shift its feet on the floor asymmetrically" rather than
walking. We did **not** evaluate gait quality here (no quantitative gait metric,
no comparison of foot-shuffle vs. true walking). So our scalar training reward is
**not comparable** to the paper's claim, and we draw **no inference** about it
either way. What we can report is narrow: with our all-off knobs the run trains to
a similar-or-higher *scalar reward* (n=5: 138.6 ± 4.7 vs. reference 103.9 ± 15.9,
sample std) rather than collapsing. Reward is not gait quality, and a degenerate
foot-shuffle can still score reward — so this number says nothing about whether
the gait is coherent.

A clip of one seed is included for completeness; we make no claim about its gait
quality from it:

![no_biomech, one seed](./videos/no_biomech_walk_s1.gif)

## Why we did not push this further

Two things make this **not** a clean comparison against the paper's result, so
we are recording it as an observation rather than a claim either way:

1. **Our knobs degenerate the pipeline to scaled raw torque.** With all three
   flags off, `_compute_torques` reduces to `torques ≈ actions · action_scale`
   (the `tanh·τ_limit` activation envelope and the Hill term cancel; the only
   residual is the 10% per-step action-hold from `loss_rate`, which we leave on
   as domain randomisation). That is a specific, benign re-parameterisation —
   not necessarily the same baseline the paper used. The paper does not give equations or code for its "w/o
   biomechanical model" variant, and the public repo
   ([marmotlab/SATA](https://github.com/marmotlab/SATA)) ships only the full
   model, so the two cannot be lined up exactly.
2. **Reset pose.** The paper describes resetting "lying flat on the ground";
   the public config resets to an upright low crouch. Starting basin matters for
   a trainability question.

Given both, more runs (raw torque with growth also off, a lying-flat reset,
etc.) would only characterise *our* variants — they could not resolve what the
paper's specific baseline does. We stopped here.

Full paper text used for this comparison is archived at
[`../../docs/reference/sata_paper_v2_extracted.txt`](../../docs/reference/sata_paper_v2_extracted.txt).

## Files

- [`run_no_biomech.sh`](./run_no_biomech.sh) — the training launcher (5 seeds, single GPU, serial)
- `logs/` — per-seed training logs
- `videos/` — recorded walking clip
