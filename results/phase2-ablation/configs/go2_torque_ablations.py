"""
Phase 2 ablation configs for the SATA bio-inspired layer.

Each class inherits GO2TorqueCfg and overrides exactly one knob, so any
behavioural difference vs. the reference must come from that single knob.

Registered tasks (see envs/__init__.py):
  go2_torque_no_fatigue     - disables motor fatigue accumulator + observation
  go2_torque_no_hill        - disables Hill force-velocity model
  go2_torque_no_activation  - disables activation low-pass filter
  go2_torque_no_growth      - control_type='T' forces general_scale=1 from start
  go2_torque_hard_terrain   - replaces slope-heavy mix with one that has stairs

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
        # 'T' branch in go2_torque.py:179 forces step_count past x0, so the
        # Gompertz curve saturates to ~1 from iteration 0 — no curriculum.
        control_type = 'T'


class GO2TorqueHardTerrainCfg(GO2TorqueCfg):
    class terrain(GO2TorqueCfg.terrain):
        # Order: [smooth_slope, rough_slope, stairs_up, stairs_down, discrete]
        # Reference: [0.2, 0.8, 0, 0, 0.0]   -- slopes only
        # Hard:      [0.0, 0.4, 0.3, 0.3, 0.0] -- 60% stairs
        terrain_proportions = [0.0, 0.4, 0.3, 0.3, 0.0]
