"""Importing this module after Isaac Sim starts registers the Piper tasks (kept out of __init__ so that the
kinematics and data-set code import without gymnasium / Isaac Lab)."""
import gymnasium as gym

for _id, _cfg in (("PolicyIK-Piper", "PolicyIKPiperEnvCfg"), ("PolicyIK-Piper-Play", "PolicyIKPiperEnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"policyik_piper.piper_env_cfg:{_cfg}",
            "rl_games_cfg_entry_point": "policyik_sim.tasks.agents:rl_games_ppo_cfg.yaml",
        },
    )
