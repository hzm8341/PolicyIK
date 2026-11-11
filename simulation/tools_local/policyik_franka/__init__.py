"""P2 Franka port of PolicyIK; importing this package after Isaac Sim starts registers the Franka tasks."""
import gymnasium as gym

for _id, _cfg in (("PolicyIK-Franka", "PolicyIKFrankaEnvCfg"), ("PolicyIK-Franka-Play", "PolicyIKFrankaEnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.franka_env_cfg:{_cfg}",
            "rl_games_cfg_entry_point": "policyik_sim.tasks.agents:rl_games_ppo_cfg.yaml",
        },
    )
