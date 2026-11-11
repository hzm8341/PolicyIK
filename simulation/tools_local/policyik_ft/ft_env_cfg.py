"""Fine-tuning environments: the official PolicyIK training env (policyik_sim.tasks.env_cfg.PolicyIKEnvCfg) with
  * the motion-cost curriculum already at its final weights (fine-tuning continues an end-of-training policy),
  * FF:     gravity feedforward added to the PD (same action term as the P1 evaluator, ctrl_actions.py),
  * KpRand: per-episode, per-joint arm PD gain scaling, stiffness U(0.8, 1.2), damping U(0.5, 2.0).
Base applies only the first change and is the control for the other two."""
from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ctrl_actions import AccumulatedJointReferenceFFActionCfg
from policyik_sim.tasks.env_cfg import PolicyIKEnvCfg


def _final_curriculum(cfg) -> None:
    cfg.rewards.action_rate.weight = cfg.curriculum.action_rate.params["weight"]   # -0.025
    cfg.rewards.joint_vel.weight = cfg.curriculum.joint_vel.params["weight"]       # -0.005
    cfg.curriculum.action_rate = None
    cfg.curriculum.joint_vel = None


@configclass
class PolicyIKFTBaseEnvCfg(PolicyIKEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _final_curriculum(self)


@configclass
class PolicyIKFTFFEnvCfg(PolicyIKFTBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        a = self.actions.arm_action
        self.actions.arm_action = AccumulatedJointReferenceFFActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, scale=a.scale, g_min=a.g_min, d_lo=a.d_lo,
            d_hi=a.d_hi, ori_lo_deg=a.ori_lo_deg, ori_hi_deg=a.ori_hi_deg, gravity_ff=True)


@configclass
class PolicyIKFTKpRandEnvCfg(PolicyIKFTBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.randomize_arm_gains = EventTerm(
            func=il_mdp.randomize_actuator_gains, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint[1-7]"]),
                    "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.5, 2.0),
                    "operation": "scale", "distribution": "uniform"})
