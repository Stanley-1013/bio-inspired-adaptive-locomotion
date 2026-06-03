"""
Phase 2 ablation configs for the SATA bio-inspired layer.

Each class inherits GO2TorqueCfg and overrides exactly one knob, so any
behavioural difference vs. the reference must come from that single knob.

Registered tasks (see envs/__init__.py):
  go2_torque_no_fatigue     - disables motor fatigue accumulator + observation
  go2_torque_no_hill        - disables Hill force-velocity model
  go2_torque_no_activation  - disables activation low-pass filter
  go2_torque_no_growth      - control_type='T' pins general_scale~=0.79 from start
  go2_torque_hard_terrain   - replaces slope-heavy mix with one that has stairs

Phase 5 (trainability) addition — see results/phase5-trainability/:
  go2_torque_no_biomech     - disables ALL THREE biomechanical stages at once
                              (activation + Hill + fatigue), keeping growth.
                              Reproduces the paper's "SATA w/o biomechanical
                              model" ablation (§V-A1), which the paper reports
                              is "completely unable to learn a coherent gait".

This file is maintained in the bio-inspired-adaptive-locomotion project repo
under results/phase2-ablation/configs/ and copied into the SATA tree here.
"""

from .go2_torque_config import GO2TorqueCfg


class GO2TorqueNoFatigueCfg(GO2TorqueCfg):
    class control(GO2TorqueCfg.control):
        motor_fatigue = False

    class rewards(GO2TorqueCfg.rewards):
        class scales(GO2TorqueCfg.rewards.scales):
            motor_fatigue = 0.0


class GO2TorqueNoHillCfg(GO2TorqueCfg):
    class control(GO2TorqueCfg.control):
        hill_model = False


class GO2TorqueNoActivationCfg(GO2TorqueCfg):
    class control(GO2TorqueCfg.control):
        activation_process = False


class GO2TorqueNoGrowthCfg(GO2TorqueCfg):
    class control(GO2TorqueCfg.control):
        # 'T' branch in go2_torque.py:179 pins step_count to 24*3000=72000 from
        # step 0, so the Gompertz curve sits at general_scale ~= 0.79 (NOT 1.0)
        # throughout — highly but not fully developed, no gradual curriculum.
        control_type = 'T'


class GO2TorqueHardTerrainCfg(GO2TorqueCfg):
    class terrain(GO2TorqueCfg.terrain):
        # Order: [smooth_slope, rough_slope, stairs_up, stairs_down, discrete]
        # Reference: [0.2, 0.8, 0, 0, 0.0]   -- slopes only
        # Hard:      [0.0, 0.4, 0.3, 0.3, 0.0] -- 60% stairs
        terrain_proportions = [0.0, 0.4, 0.3, 0.3, 0.0]


class GO2TorqueNoBiomechCfg(GO2TorqueCfg):
    """Phase 5: all three biomechanical stages OFF at once, growth KEPT.

    Reproduces the paper's "SATA w/o biomechanical model" (§V-A1). With all
    three flags False the torque pipeline in go2_torque.py:_compute_torques
    degenerates to `torques = actions * action_scale` — the policy's raw output
    is applied directly as joint torque, with no activation low-pass, no
    Hill force-velocity drop-off, and the fatigue observation pinned to zero.
    The paper reports this variant is "completely unable to learn a coherent
    gait, instead learning to shift its feet on the floor asymmetrically."

    growth (control_type='TG') is intentionally LEFT ON so this isolates the
    biomechanical model exactly as the paper's ablation does — no_growth is a
    separate ablation. The fatigue reward scale is also zeroed (a nonzero
    penalty on an always-zero fatigue signal would be a no-op, but we zero it
    for cleanliness / parity with no_fatigue).
    """
    class control(GO2TorqueCfg.control):
        activation_process = False
        hill_model = False
        motor_fatigue = False

    class rewards(GO2TorqueCfg.rewards):
        class scales(GO2TorqueCfg.rewards.scales):
            motor_fatigue = 0.0
