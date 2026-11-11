"""Local action terms for the P1 comparison (the official task code is not modified).

ReferenceJointAction: the action *is* the absolute arm joint reference z_t (used by DLS / QP / Isaac Lab DLS).
AccumulatedJointReferenceFFAction: PolicyIK's accumulated reference, plus optional gravity feedforward.
Both optionally add exact gravity compensation from the simulator as a joint-effort feedforward, refreshed at
every physics substep (apply_actions runs once per substep).
"""
from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from policyik_sim.tasks.actions import AccumulatedJointReferenceAction, AccumulatedJointReferenceActionCfg


def _gravity_ff(asset, joint_ids) -> torch.Tensor:
    return asset.root_physx_view.get_gravity_compensation_forces()[:, joint_ids]


class ReferenceJointAction(JointPositionAction):
    cfg: "ReferenceJointActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._int_target = self._asset.data.joint_pos[:, self._joint_ids].clone()   # z (for command_state_obs)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._int_target = actions.clone()
        self._processed_actions = self._int_target

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        if self.cfg.gravity_ff:
            self._asset.set_joint_effort_target(_gravity_ff(self._asset, self._joint_ids), joint_ids=self._joint_ids)


@configclass
class ReferenceJointActionCfg(JointPositionActionCfg):
    class_type: type = ReferenceJointAction
    gravity_ff: bool = True


class AccumulatedJointReferenceFFAction(AccumulatedJointReferenceAction):
    def apply_actions(self) -> None:
        super().apply_actions()
        if self.cfg.gravity_ff:
            self._asset.set_joint_effort_target(_gravity_ff(self._asset, self._joint_ids), joint_ids=self._joint_ids)


@configclass
class AccumulatedJointReferenceFFActionCfg(AccumulatedJointReferenceActionCfg):
    class_type: type = AccumulatedJointReferenceFFAction
    gravity_ff: bool = False
