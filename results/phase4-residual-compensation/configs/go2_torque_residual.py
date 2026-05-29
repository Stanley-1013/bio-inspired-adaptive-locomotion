"""
Phase 4 — residual body-height compensation on top of a frozen SATA policy.

tau_total = tau_SATA + tau_comp, where tau_comp is a classical PD on body
height applied to the leg-extension (calf) joints. The residual is a separate
channel: it is added AFTER the bio-layer (activation / Hill / fatigue), so it
does NOT pass through the fatigue penalty — it is the "classical compensator
bolted onto the learned controller" of the project's Phase 4.

The RL policy is NOT retrained; this env is only used at evaluation, loading a
reference (`ref_sN`) checkpoint. Because it subclasses GO2Torque the obs/action
dimensions are identical, so a reference policy loads unchanged.

Maintained in the project repo at results/phase4-residual-compensation/configs/
and copied into the SATA tree.
"""

import torch

from .go2_torque import GO2Torque
from .go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO


class GO2TorqueResidualCfg(GO2TorqueCfg):
    class residual:
        enable = True
        target_height = 0.32      # nominal achieved base height (reference ~0.317)
        kp = 80.0                 # N·m per metre of height error
        kd = 4.0                  # N·m per (m/s) of vertical velocity
        tau_cap = 15.0            # clamp on the residual torque per joint (N·m)
        sign = -1.0               # calf-extension torque sign (calibrated empirically)
        joints = ["calf"]         # dof-name substrings the residual is applied to
        stance_gate = True        # only assist legs whose foot is in contact (else it
                                  # fights the swing legs and disrupts the gait)
        contact_thresh = 1.0      # N, foot-contact force threshold for "in stance"


class GO2TorqueResidual(GO2Torque):
    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)   # tau_SATA (post bio-layer)
        rcfg = getattr(self.cfg, "residual", None)
        if rcfg is None or not getattr(rcfg, "enable", False):
            self.residual_torque = torch.zeros_like(torques)
            return torques

        # Lazily resolve, per leg, the residual DOF index and its foot body index.
        # dof_names are per-leg [hip, thigh, calf] in leg order; feet_indices are in
        # the same leg order, so leg i -> i-th matching DOF and i-th foot.
        if not hasattr(self, "_resid_idx"):
            self._resid_idx = torch.tensor(
                [i for i, n in enumerate(self.dof_names)
                 if any(s in n for s in rcfg.joints)],
                device=self.device, dtype=torch.long)

        h = self.root_states[:, 2]        # world-frame base height
        hdot = self.root_states[:, 9]     # world-frame vertical velocity
        # Only assist when the body is BELOW target (sagging); never pull down.
        err = (rcfg.target_height - h).clamp(min=0.0)
        tau_c = (rcfg.kp * err - rcfg.kd * hdot).clamp(min=0.0, max=rcfg.tau_cap)  # [n]

        self.residual_torque = torch.zeros_like(torques)
        per_leg = rcfg.sign * tau_c.unsqueeze(1)      # [n, 1] broadcast to legs
        if getattr(rcfg, "stance_gate", True):
            # foot-contact mask per leg, aligned with self._resid_idx leg order
            contact = (self.contact_forces[:, self.feet_indices, 2].abs()
                       > rcfg.contact_thresh).float()   # [n, num_legs]
            self.residual_torque[:, self._resid_idx] = per_leg * contact
        else:
            self.residual_torque[:, self._resid_idx] = per_leg
        torques = torques + self.residual_torque
        self.torques = torques
        return torques


class GO2TorqueResidualCfgPPO(GO2TorqueCfgPPO):
    pass
