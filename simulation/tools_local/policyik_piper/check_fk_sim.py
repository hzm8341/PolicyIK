"""S1 gate check: the Piper USD in Isaac Lab vs the URDF kinematics of piper_kin.py on 2,000 valid
configurations of the source set (position, rotation with the atan2 metric, soft joint limits), plus the
holding offset Kp^-1 tau_g(q) an unsaturated PD without gravity feedforward needs (paper Eq. 11).
usage (from PolicyIK/simulation): python tools_local/policyik_piper/check_fk_sim.py --headless
       -> outputs/piper/s1/fk_check.txt
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
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import policyik_piper.register  # noqa: F401,E402
from policyik_piper.piper_kin import EE_LINK, PiperKinematics, quat_angle  # noqa: E402

N = 2000
OUT = "outputs/piper/s1/fk_check.txt"


def main():
    cfg = parse_env_cfg("PolicyIK-Piper-Play", num_envs=N)
    env = gym.make("PolicyIK-Piper-Play", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    arm, names = robot.find_joints(["joint[1-6]"], preserve_order=True)
    ee = robot.find_bodies(EE_LINK)[0][0]
    kin = PiperKinematics()
    src = np.load("outputs/piper/data/source_set.npz")
    valid = np.where(~src["self_collision"] & ~src["floor_contact"])[0]
    rows = np.random.default_rng(7).choice(valid, N, replace=False)
    q = torch.as_tensor(src["q"][rows], device=env.device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=arm)
    env.sim.forward()
    robot.update(env.physics_dt)
    p_b, q_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         robot.data.body_pos_w[:, ee], robot.data.body_quat_w[:, ee])
    p_b, q_b = p_b.cpu().numpy().astype(float), q_b.cpu().numpy().astype(float)
    p_u, q_u = kin.fk(src["q"][rows].astype(float))
    dp = np.linalg.norm(p_b - p_u, axis=1)
    dr = np.degrees(quat_angle(q_b, q_u))
    lim = robot.data.soft_joint_pos_limits[0, arm].cpu().numpy()
    tau = robot.root_physx_view.get_gravity_compensation_forces()[:, arm].cpu().numpy()
    kp = robot.data.joint_stiffness[0, arm].cpu().numpy()
    kd = robot.data.joint_damping[0, arm].cpu().numpy()
    eff = robot.data.joint_effort_limits[0, arm].cpu().numpy()
    off = tau / kp
    lines = [f"all joints (sim order): {robot.joint_names}",
             f"bodies: {robot.body_names}",
             f"arm joints: {names}   end-effector body: {EE_LINK}",
             f"other joints at: {robot.data.joint_pos[0].cpu().numpy()[[i for i in range(robot.num_joints) if i not in arm]]}",
             f"FK, {N} configs: max |dp| = {1e3 * dp.max():.4f} mm, P99 = {1e3 * np.percentile(dp, 99):.4f} mm, "
             f"max |dR| = {dr.max():.5f} deg",
             "soft limits sim vs URDF (rad):"]
    lines += [f"  {n}: sim [{a:.4f}, {b:.4f}]  urdf [{c:.4f}, {d:.4f}]"
              for n, (a, b), (c, d) in zip(names, lim, kin.limits)]
    lines.append(f"max |limit diff| = {np.abs(lim - kin.limits).max():.2e} rad")
    lines.append(f"Kp per arm joint: {kp}   Kd: {kd}")
    lines.append(f"effort limits: {eff}")
    lines.append("holding offset Kp^-1 tau_g over valid configs (rad): median |.| / P95 |.| / max |.| per joint")
    lines += [f"  {n}: {np.median(np.abs(off[:, j])):.4f} / {np.percentile(np.abs(off[:, j]), 95):.4f} / "
              f"{np.abs(off[:, j]).max():.4f}   (|tau_g| P95 {np.percentile(np.abs(tau[:, j]), 95):.2f} Nm, "
              f"max {np.abs(tau[:, j]).max():.2f} Nm)" for j, n in enumerate(names)]
    lines.append(f"configs where |tau_g| exceeds the effort limit on some joint: "
                 f"{100 * (np.abs(tau) > eff).any(1).mean():.2f} %")
    ok = dp.max() < 1e-4 and dr.max() < 0.01 and np.abs(lim - kin.limits).max() < 1e-3
    lines.append(f"FK GATE: {'PASS' if ok else 'FAIL'} (thresholds 0.1 mm, 0.01 deg, 1e-3 rad limits)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
