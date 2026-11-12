"""Isaac Lab reach environment for the AgileX Piper right arm (6 arm joints, stock gripper), the base that
piper_env_cfg.py builds the PolicyIK tasks on -- the counterpart of OpenArmReachEnvCfg / FrankaReachEnvCfg.

Asset: the USD converted from the real right arm's URDF (fixed joints not merged, so link6 is a body). It is
read in place from the Intelligence_AOP checkout; set PIPER_USD to use another copy.

Not taken from the vendor: the PD gains. Kp 80 / Kd 4 are the PolicyIK (OpenArm) values, the starting point
agreed for the hold test of stage S1; effort 100 Nm and velocity 5 rad/s are the URDF limits.
PIPER_SELF_COLLISIONS=1 enables self-collision; it is meant for the hold run that screens start and goal
configurations. The default is off, as in the OpenArm asset the paper's experiments use.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

PIPER_USD = os.environ.get(
    "PIPER_USD", "/home/hzm/code/Intelligence_AOP/sim/assets/piper_agx/piper_right/piper_right.usd")
SELF_COLLISIONS = os.environ.get("PIPER_SELF_COLLISIONS", "0") == "1"
EE_BODY = "link6"
ARM_JOINTS = "joint[1-6]"

PIPER_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=SELF_COLLISIONS, solver_position_iteration_count=8,
            solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint[1-6]": 0.0, "gripper_joint[1-2]": 0.0}),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=[ARM_JOINTS], effort_limit_sim=100.0, velocity_limit_sim=5.0,
            stiffness=80.0, damping=4.0),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint[1-2]"], effort_limit_sim=10.0, velocity_limit_sim=3.0,
            stiffness=2e3, damping=1e2),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class PiperReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = PIPER_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [EE_BODY]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[ARM_JOINTS], scale=0.5, use_default_offset=True)
        self.commands.ee_pose.body_name = EE_BODY
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)     # unused: goals come from the pools


@configclass
class PiperReachEnvCfg_PLAY(PiperReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
