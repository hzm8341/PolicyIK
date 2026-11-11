"""P2 gate check: Isaac Lab Franka (USD) vs independent URDF FK on 20 random valid configurations.

Also compares joint limits (USD soft limits vs URDF) and reports the holding offset Kp^-1 tau_g(q) that an
unsaturated PD without gravity feedforward needs (paper Eq. 11), over 2,000 valid configurations.
usage: python tools_local/policyik_franka/check_fk_sim.py --headless   -> outputs/p2/fk_check.txt
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import policyik_franka  # noqa: F401,E402
from policyik_franka.urdf_fk import FrankaKinematics  # noqa: E402

N_FK, N_G = 20, 2000


def quat_angle(a, b):
    """Angle between unit quaternions (wxyz). Uses atan2(|vec|, |w|) of the relative quaternion: arccos of the
    dot product is ill-conditioned near 0 (float32 round-off of 1e-7 would read as ~0.05 deg)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    w = np.sum(a * b, -1)
    v = (a[..., :1] * b[..., 1:] - b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:]))
    return 2 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w))


def main():
    cfg = parse_env_cfg("PolicyIK-Franka-Play", num_envs=N_G)
    env = gym.make("PolicyIK-Franka-Play", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    arm, names = robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
    hand = robot.find_bodies("panda_hand")[0][0]
    kin = FrankaKinematics()
    src = np.load("outputs/p2/data/source_set.npz")
    valid = np.where(~src["self_collision"] & ~src["floor_contact"])[0]
    rows = np.random.default_rng(7).choice(valid, N_G, replace=False)
    q = torch.as_tensor(src["q"][rows], device=env.device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=arm)
    env.sim.forward()
    robot.update(env.physics_dt)
    root_p, root_q = robot.data.root_pos_w, robot.data.root_quat_w
    from isaaclab.utils.math import subtract_frame_transforms
    p_b, q_b = subtract_frame_transforms(root_p, root_q, robot.data.body_pos_w[:, hand], robot.data.body_quat_w[:, hand])
    p_b, q_b = p_b.cpu().numpy().astype(float), q_b.cpu().numpy().astype(float)
    p_u, q_u = kin.fk(src["q"][rows].astype(float))
    dp = np.linalg.norm(p_b - p_u, axis=1)
    dr = np.degrees(quat_angle(q_b, q_u))
    lim = robot.data.soft_joint_pos_limits[0, arm].cpu().numpy()
    tau = robot.root_physx_view.get_gravity_compensation_forces()[:, arm].cpu().numpy()
    kp = robot.data.joint_stiffness[0, arm].cpu().numpy()
    off = tau / kp
    eff = robot.data.joint_effort_limits[0, arm].cpu().numpy() if hasattr(robot.data, "joint_effort_limits") else None
    lines = [f"joint names (sim order): {names}",
             f"FK, first {N_FK} configs: max |dp| = {1e3 * dp[:N_FK].max():.4f} mm, max |dR| = {dr[:N_FK].max():.4f} deg",
             f"FK, all {N_G} configs: max |dp| = {1e3 * dp.max():.4f} mm, P99 = {1e3 * np.percentile(dp, 99):.4f} mm, "
             f"max |dR| = {dr.max():.4f} deg",
             "soft limits sim vs URDF (rad):"]
    lines += [f"  {n}: sim [{a:.4f}, {b:.4f}]  urdf [{c:.4f}, {d:.4f}]"
              for n, (a, b), (c, d) in zip(names, lim, kin.limits)]
    lines.append(f"max |limit diff| = {np.abs(lim - kin.limits).max():.2e} rad")
    lines.append(f"Kp per arm joint: {kp}")
    lines.append(f"effort limits: {eff}")
    lines.append("holding offset Kp^-1 tau_g over valid configs (rad): median |.| / P95 |.| / max |.| per joint")
    lines += [f"  {n}: {np.median(np.abs(off[:, j])):.4f} / {np.percentile(np.abs(off[:, j]), 95):.4f} / "
              f"{np.abs(off[:, j]).max():.4f}   (|tau_g| P95 {np.percentile(np.abs(tau[:, j]), 95):.2f} Nm)"
              for j, n in enumerate(names)]
    if eff is not None:
        lines.append(f"configs where |tau_g| exceeds the effort limit on some joint: "
                     f"{100 * (np.abs(tau) > eff).any(1).mean():.2f} %")
    ok = dp[:N_FK].max() < 1e-4 and dr[:N_FK].max() < 0.01 and np.abs(lim - kin.limits).max() < 1e-3
    lines.append(f"FK GATE: {'PASS' if ok else 'FAIL'} (thresholds 0.1 mm, 0.01 deg, 1e-3 rad limits)")
    os.makedirs("outputs/p2", exist_ok=True)
    open("outputs/p2/fk_check.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
