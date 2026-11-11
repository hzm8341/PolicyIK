"""Step-4 fine-tuning variants of the released OpenArm PolicyIK checkpoint (registration only; no Isaac imports)."""
import gymnasium as gym

for _v in ("Base", "FF", "KpRand"):
    gym.register(
        id=f"PolicyIK-OpenArm-FT-{_v}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ft_env_cfg:PolicyIKFT{_v}EnvCfg",
            "rl_games_cfg_entry_point": "policyik_sim.tasks.agents:rl_games_ppo_cfg.yaml",
        },
    )
