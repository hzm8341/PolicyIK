"""P1 evaluator: same task loop as policyik_sim/evaluate.py, with a selectable controller.

Controllers
  policyik     released PolicyIK checkpoint (identical to the official evaluator; --gravity_ff adds feedforward)
  dls          damped least squares, bounded (paper Appendix E/I):  k, lam=0.02, W=diag(1,1,1,.2,.2,.2)
  qp           bounded resolved-rate QP (Appendix E/I):            k, lam=0.005, same W
  dls_default  Isaac Lab differential IK, default settings: z = clamp(q + J^T (J J^T + 0.01^2 I)^-1 e, soft limits)
  mpc          kinematic MPC, 7-step horizon, DAQP (tools_local/mpc.py):  k=1.5, lam=0.005, acc=100
  hold         z = q_start (sanity check of gravity feedforward)

DLS and QP integrate z_{t+1} = z_t + dt * qd with qd from
    min ||W J qd - k W e||^2 + lam^2 ||qd||^2  s.t.  soft joint limits and |z_{t+1} - q_t| <= lead.
Without a reference-speed bound (not applied in simulation, Appendix E) the DLS rule "unconstrained step, and
a bounded least-squares solve of the same damped task when it violates a bound" equals the bounded optimum,
so DLS and QP differ in (k, lam) here, as in the paper's simulation settings.

Extra dump fields: q_t, z_t (arm joints), soft limits, controller time per step.
Run from PolicyIK/simulation:  python tools_local/evaluate_ctrl.py --headless --controller dls ...
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="PolicyIK-OpenArm-Play")
parser.add_argument("--controller", required=True, choices=["policyik", "dls", "qp", "mpc", "dls_default", "hold"])
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--task_file", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--eval_seed", type=int, default=123)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--transient_seed", type=int, default=-1)
parser.add_argument("--k", type=float, default=None, help="task gain (default: dls 2.0, qp 1.5)")
parser.add_argument("--lam", type=float, default=None, help="damping (default: dls 0.02, qp 0.005)")
parser.add_argument("--lead", type=float, default=0.15, help="command-to-measurement lead bound (rad)")
parser.add_argument("--acc", type=float, default=100.0, help="MPC reference acceleration bound (rad/s^2)")
parser.add_argument("--trust", type=float, default=0.10, help="MPC trust region on |z_i - z| (rad)")
parser.add_argument("--kp_scale", type=float, default=1.0, help="robustness test: arm PD stiffness scale")
parser.add_argument("--kd_scale", type=float, default=1.0, help="robustness test: arm PD damping scale")
parser.add_argument("--rot_w", type=float, default=0.2, help="rotation task weight in W")
ff = parser.add_mutually_exclusive_group()
ff.add_argument("--gravity_ff", dest="gravity_ff", action="store_true", default=None)
ff.add_argument("--no_gravity_ff", dest="gravity_ff", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms, compute_pose_error, quat_error_magnitude, quat_from_angle_axis, quat_mul)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import policyik_sim.tasks  # noqa: F401,E402
from boxqp import solve_box_qp  # noqa: E402
from ctrl_actions import AccumulatedJointReferenceFFActionCfg, ReferenceJointActionCfg  # noqa: E402

FREEZE = 1.0e6
ARM_EXPR = ["openarm_joint[1-7]"]
TRANSIENT_STEPS, TRANSIENT_OFFSET_M, TRANSIENT_ROT_DEG = 40, 0.30, 90.0
CTRL = args_cli.controller
DEFAULTS = {"dls": (2.0, 0.02), "qp": (1.5, 0.005), "mpc": (1.5, 0.005)}
if CTRL in DEFAULTS:
    K = args_cli.k if args_cli.k is not None else DEFAULTS[CTRL][0]
    LAM = args_cli.lam if args_cli.lam is not None else DEFAULTS[CTRL][1]
# gravity feedforward: all classical controllers receive it (Appendix E, G); PolicyIK does not
GFF = args_cli.gravity_ff if args_cli.gravity_ff is not None else CTRL in ("dls", "qp", "mpc", "dls_default", "hold")


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.eval_seed
    agent_cfg["params"]["seed"] = args_cli.eval_seed
    if args_cli.kp_scale != 1.0 or args_cli.kd_scale != 1.0:
        act = env_cfg.scene.robot.actuators["openarm_arm"]
        act.stiffness = act.stiffness * args_cli.kp_scale
        act.damping = act.damping * args_cli.kd_scale
        print(f"[evaluate_ctrl] PD perturbation: stiffness={act.stiffness} damping={act.damping}", flush=True)
    a = env_cfg.actions.arm_action
    if CTRL == "policyik":
        env_cfg.actions.arm_action = AccumulatedJointReferenceFFActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, scale=a.scale, g_min=a.g_min, d_lo=a.d_lo,
            d_hi=a.d_hi, ori_lo_deg=a.ori_lo_deg, ori_hi_deg=a.ori_hi_deg, gravity_ff=GFF)
    else:
        env_cfg.actions.arm_action = ReferenceJointActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, scale=1.0, use_default_offset=False, gravity_ff=GFF)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if CTRL == "policyik":
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate = agent_cfg["params"]["env"].get("concate_obs_groups", True)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate)
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: env)
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
    uenv = env.unwrapped

    cmd = uenv.command_manager.get_term("ee_pose")
    pos_term = uenv.reward_manager.get_term_cfg("end_effector_position_tracking")
    robot = uenv.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_EXPR, preserve_order=True)
    arm_ids = list(arm_ids)
    device = uenv.device
    act_term = uenv.action_manager.get_term("arm_action")
    assert list(act_term._joint_ids) == arm_ids or act_term._joint_ids == slice(None), act_term._joint_ids
    assert robot.is_fixed_base
    jac_body = cmd.body_idx - 1
    soft = robot.data.soft_joint_pos_limits[0, arm_ids].clone()          # (7, 2)
    lo, hi = soft[:, 0].double(), soft[:, 1].double()
    dt = env_cfg.sim.dt * env_cfg.decimation
    W2 = torch.tensor([1, 1, 1, args_cli.rot_w, args_cli.rot_w, args_cli.rot_w], device=device,
                      dtype=torch.float64) ** 2
    print(f"[evaluate_ctrl] controller={CTRL} gravity_ff={GFF} dt={dt:.5f}"
          + (f" k={K} lam={LAM} lead={args_cli.lead} rot_w={args_cli.rot_w}" if CTRL in DEFAULTS else ""), flush=True)

    if CTRL == "mpc":
        from mpc import KinematicMPC
        mpc = KinematicMPC(dt, k=K, lam=LAM, lead=args_cli.lead, rot_w=args_cli.rot_w, acc=args_cli.acc,
                           trust=args_cli.trust)
    mpc_state = {"v": None}

    def pos_err_fn():
        return pos_term.func(uenv, **pos_term.params).detach()

    def ori_err_fn():
        ee_quat = cmd.robot.data.body_quat_w[:, cmd.body_idx]
        return quat_error_magnitude(ee_quat, cmd.pose_command_w[:, 3:]).detach()

    def refresh_manual_goal_world():
        cmd.pose_command_w[:, :3], cmd.pose_command_w[:, 3:] = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, cmd.pose_command_b[:, :3], cmd.pose_command_b[:, 3:])

    def classical_step(z: torch.Tensor) -> torch.Tensor:
        q = robot.data.joint_pos[:, arm_ids].double()
        ee_pos = robot.data.body_pos_w[:, cmd.body_idx]
        ee_quat = robot.data.body_quat_w[:, cmd.body_idx]
        ep, er = compute_pose_error(ee_pos, ee_quat, cmd.pose_command_w[:, :3], cmd.pose_command_w[:, 3:],
                                    rot_error_type="axis_angle")
        e = torch.cat([ep, er], dim=1).double()
        J = robot.root_physx_view.get_jacobians()[:, jac_body, :, :][:, :, arm_ids].double()   # (n,6,7) world
        if CTRL == "hold":
            return z
        if CTRL == "dls_default":
            lam = 0.01
            JJt = J @ J.transpose(1, 2) + lam ** 2 * torch.eye(6, device=device, dtype=torch.float64)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
            return torch.clamp(q + dq, lo, hi).float()   # targets clamped at the soft limits (Appendix G)
        if CTRL == "mpc":
            zn, v = mpc.step(J.cpu().numpy(), e.cpu().numpy(), q.cpu().numpy(), z.double().cpu().numpy(),
                             mpc_state["v"], lo.cpu().numpy(), hi.cpu().numpy())
            mpc_state["v"] = v
            return torch.as_tensor(zn, device=device, dtype=torch.float32)
        JtW2 = J.transpose(1, 2) * W2                                     # J^T W^2
        H = JtW2 @ J + LAM ** 2 * torch.eye(7, device=device, dtype=torch.float64)
        g = K * (JtW2 @ e.unsqueeze(-1)).squeeze(-1)
        zd = z.double()
        zlo = torch.maximum(lo, q - args_cli.lead)
        zhi = torch.minimum(hi, q + args_cli.lead)
        qd, _ = solve_box_qp(H, g, (zlo - zd) / dt, (zhi - zd) / dt)
        return (zd + dt * qd).float()

    d = np.load(args_cli.task_file)
    q_start = d["q_start"]
    goals = np.concatenate([np.asarray(d["goal_pos"], np.float32), np.asarray(d["goal_quat"], np.float32)], 1)
    m_total = q_start.shape[0]
    transient = args_cli.transient_seed >= 0
    if transient:
        gr = np.random.default_rng(args_cli.transient_seed)
        dirs = gr.normal(size=(m_total, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        axes = gr.normal(size=(m_total, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    n, T = args_cli.num_envs, args_cli.steps
    out = {k: np.full((m_total, T), np.nan, np.float32) for k in ("pos_t", "ori_t")}
    out.update({k: np.full((m_total, T, s), np.nan, np.float32)
                for k, s in (("ee_pos_t", 3), ("ee_quat_t", 4), ("q_t", 7), ("z_t", 7))})
    out["goal_pos_w"] = np.full((m_total, 3), np.nan, np.float32)
    out["goal_quat_w"] = np.full((m_total, 4), np.nan, np.float32)
    ctrl_time = []
    with torch.inference_mode():
        obs0 = env.reset()
        if CTRL == "policyik":
            if isinstance(obs0, dict):
                obs0 = obs0["obs"]
            agent.get_batch_size(obs0, 1)
            if agent.is_rnn:
                agent.init_rnn()
        for b in range(0, m_total, n):
            sl = slice(b, min(b + n, m_total))
            m = sl.stop - sl.start
            goals_batch = np.zeros((n, 7), np.float32)
            goals_batch[:m] = goals[sl]
            if not transient:
                goals_batch[m:] = goals[sl.start]
            q_batch = np.zeros((n, 7), np.float32)
            q_batch[:m] = q_start[sl]
            q_batch[m:] = q_start[sl.start]
            goals_b = torch.from_numpy(goals_batch).to(device)
            q_b = torch.as_tensor(q_batch, device=device)
            if transient:
                g_tr = goals_batch.copy()
                g_tr[:m, :3] += (TRANSIENT_OFFSET_M * dirs[sl]).astype(np.float32)
                goals_b_tr = torch.from_numpy(g_tr).to(device)
                ang = torch.full((n,), float(np.deg2rad(TRANSIENT_ROT_DEG)), device=device)
                ax = torch.zeros((n, 3), device=device)
                ax[:m] = torch.as_tensor(axes[sl], device=device, dtype=torch.float32)
                ax[m:, 0] = 1.0
                goals_b_tr[:, 3:] = quat_mul(quat_from_angle_axis(ang, ax), goals_b_tr[:, 3:])

            def _pin(t):
                cmd.pose_command_b.copy_(goals_b_tr if (transient and t < TRANSIENT_STEPS) else goals_b)
                refresh_manual_goal_world()

            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            robot.write_joint_state_to_sim(q_b, torch.zeros_like(q_b), joint_ids=arm_ids)
            cmd.cfg.resampling_time_range = (FREEZE, FREEZE)
            cmd.time_left.fill_(FREEZE)
            _pin(0)
            uenv.sim.forward()
            if CTRL == "policyik":
                fresh = uenv.observation_manager.compute()["policy"]
                obs = torch.clamp(fresh, -clip_obs, clip_obs)
            z = q_b.clone()                                  # the reference starts at the measured joints
            if CTRL == "mpc":
                mpc_state["v"] = np.zeros((n, 7))

            for t in range(T):
                _pin(t)
                t0 = time.perf_counter()
                if CTRL == "policyik":
                    actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                    obs, _, _, _ = env.step(actions)
                else:
                    z = classical_step(z)
                    torch.cuda.synchronize()
                    ctrl_time.append(time.perf_counter() - t0)
                    uenv.step(z)
                out["pos_t"][sl, t] = pos_err_fn()[:m].cpu().numpy()
                out["ori_t"][sl, t] = ori_err_fn()[:m].cpu().numpy()
                out["ee_pos_t"][sl, t] = cmd.robot.data.body_pos_w[:, cmd.body_idx][:m].cpu().numpy()
                out["ee_quat_t"][sl, t] = cmd.robot.data.body_quat_w[:, cmd.body_idx][:m].cpu().numpy()
                out["q_t"][sl, t] = robot.data.joint_pos[:, arm_ids][:m].cpu().numpy()
                out["z_t"][sl, t] = act_term._int_target[:m].cpu().numpy()
                if t == 0:
                    out["goal_pos_w"][sl] = cmd.pose_command_w[:, :3][:m].cpu().numpy()
                    out["goal_quat_w"][sl] = cmd.pose_command_w[:, 3:][:m].cpu().numpy()
            print(f"[evaluate_ctrl] tasks {sl.start}-{sl.stop} of {m_total} done", flush=True)

    out["soft_limits"] = soft.cpu().numpy()
    if CTRL == "mpc":
        out["mpc_fallbacks"] = np.array([mpc.n_fallback, mpc.n_solves])
        print(f"[evaluate_ctrl] mpc fallbacks {mpc.n_fallback} of {mpc.n_solves} solves", flush=True)
    if ctrl_time:
        out["ctrl_time_per_step_s"] = np.array(ctrl_time, np.float64)
    out["config"] = np.array(str(dict(controller=CTRL, gravity_ff=GFF, k=K if CTRL in DEFAULTS else None,
                                      lam=LAM if CTRL in DEFAULTS else None, lead=args_cli.lead,
                                      rot_w=args_cli.rot_w, acc=args_cli.acc, trust=args_cli.trust, kp_scale=args_cli.kp_scale,
                                      kd_scale=args_cli.kd_scale, num_envs=n, steps=T,
                                      checkpoint=args_cli.checkpoint, task_file=args_cli.task_file,
                                      transient_seed=args_cli.transient_seed)))
    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    np.savez(args_cli.output, **out)
    print(f"[evaluate_ctrl] wrote {args_cli.output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
