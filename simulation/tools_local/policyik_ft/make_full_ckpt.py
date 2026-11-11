"""Wrap a released (model-only) checkpoint into an rl_games full checkpoint (epoch 0, fresh Adam state) so that
`policyik_sim.train --checkpoint` can fine-tune from it. usage: make_full_ckpt.py <in.pth> <out.pth>"""
import sys

import torch

src = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model = src["model"]
n_params = sum(1 for k in model if not ("running_mean_std" in k or "value_mean_std" in k))
opt = {"state": {}, "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0,
                                      "amsgrad": False, "maximize": False, "foreach": None, "capturable": False,
                                      "differentiable": False, "fused": None, "params": list(range(n_params))}]}
torch.save({"model": model, "epoch": 0, "frame": 0, "optimizer": opt, "last_mean_rewards": -1e9,
            "env_state": None}, sys.argv[2])
print(f"wrote {sys.argv[2]} ({n_params} parameter tensors)")
