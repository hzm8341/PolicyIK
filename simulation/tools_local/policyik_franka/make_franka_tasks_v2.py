"""P2 task sets, version 2 (narrowed tabletop distribution, simulator-validated).

Changes against make_franka_tasks.py (v1), decided after the v1 baseline study (reports/policyik_p2_prep_summary.md):
  * starts AND goals lie in a tabletop box in front of the robot (panda_hand in panda_link0):
        0.25 <= x <= 0.75 m, |y| <= 0.45 m, 0.05 <= z <= 0.65 m
    This gives a median start-goal distance of about 0.40 m (v1: 0.79 m; OpenArm paper set: 0.43 m).
  * every candidate configuration must also hold still in the simulator (60 steps with gravity feedforward,
    end-effector drift < 0.1 mm); the offline sphere screening misses hand/forearm contacts of the USD asset.
  * 1,000,000 source configurations (v1: 200,000) so the pools stay large after both filters.
stages:  source  -> outputs/p2/data_v2/{source_set,simcheck_candidates}.npz        (no simulator)
         split   -> needs outputs/p2/data_v2/simcheck_hold.npz from the simulator hold run; writes the pools,
                    eval_tasks, eval_goal_q, tune_tasks.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from urdf_fk import FrankaKinematics, SphereCollision  # noqa: E402

import policyik_sim.splits as S  # noqa: E402

N_SRC, SEED, TUNE_SEED, N_TUNE = 1_000_000, 20261113, 20261114, 1024
X, Y, Z = (0.25, 0.75), 0.45, (0.05, 0.65)
DRIFT_MAX = 1e-4
OUT = "outputs/p2/data_v2"


def in_box(p):
    return (p[:, 0] >= X[0]) & (p[:, 0] <= X[1]) & (np.abs(p[:, 1]) <= Y) & (p[:, 2] >= Z[0]) & (p[:, 2] <= Z[1])


def source():
    os.makedirs(OUT, exist_ok=True)
    kin = FrankaKinematics()
    col = SphereCollision(kin)
    q = np.random.default_rng(SEED).uniform(kin.limits[:, 0], kin.limits[:, 1], (N_SRC, 7))
    pos, quat, sc, fc = [], [], [], []
    for s in range(0, N_SRC, 20000):
        p, o = kin.fk(q[s:s + 20000])
        a, b = col.check(q[s:s + 20000])
        pos.append(p); quat.append(o); sc.append(a); fc.append(b)
    pos, quat, sc, fc = map(np.concatenate, (pos, quat, sc, fc))
    np.savez(f"{OUT}/source_set.npz", q=q.astype(np.float32), ee_pos=pos.astype(np.float32),
             ee_quat=quat.astype(np.float32), floor_contact=fc, self_collision=sc)
    cand = np.where(~sc & ~fc & in_box(pos))[0]
    np.savez(f"{OUT}/simcheck_candidates.npz", rows=cand, q_start=q[cand].astype(np.float32),
             goal_pos=pos[cand].astype(np.float32), goal_quat=quat[cand].astype(np.float32))
    print(f"offline valid {(~sc & ~fc).mean():.3f}; candidates in box {len(cand)}")


def split():
    src = dict(np.load(f"{OUT}/source_set.npz"))
    cand = np.load(f"{OUT}/simcheck_candidates.npz")["rows"]
    h = np.load(f"{OUT}/simcheck_hold.npz")
    drift = np.linalg.norm(h["ee_pos_t"][:, -1] - h["ee_pos_t"][:, 0], axis=1)
    assert len(drift) == len(cand)
    ok = np.zeros(N_SRC, bool)
    ok[cand[drift < DRIFT_MAX]] = True
    print(f"simulator-valid candidates: {ok.sum()} / {len(cand)} ({100 * ok.sum() / len(cand):.1f} %)")
    # reuse the official split()/check(): everything outside the validated box counts as invalid, so starts,
    # goals and both training pools come from the same validated set
    src["sim_contact"] = ~ok
    src["self_collision"] = src["self_collision"] | ~ok
    S.in_workspace = lambda p: in_box(p)
    goals, starts, train, train_goal = S.split(src)
    S.check(src, goals, starts, train)
    for name, m in (("train_start_pool", train), ("train_goal_pool", train_goal)):
        np.savez(f"{OUT}/{name}.npz", **{k: src[k][m] for k in ("q", "ee_pos", "ee_quat")})
    np.savez(f"{OUT}/eval_tasks.npz", q_start=src["q"][starts], goal_pos=src["ee_pos"][goals],
             goal_quat=src["ee_quat"][goals])
    np.savez(f"{OUT}/eval_goal_q.npz", q_goal=src["q"][goals])
    rng = np.random.default_rng(TUNE_SEED)
    st = np.where(train)[0][rng.choice(train.sum(), N_TUNE, replace=False)]
    gl = np.where(train_goal)[0][rng.choice(train_goal.sum(), N_TUNE, replace=False)]
    assert not (set(st) | set(gl)) & (set(goals) | set(starts))
    np.savez(f"{OUT}/tune_tasks.npz", q_start=src["q"][st], goal_pos=src["ee_pos"][gl], goal_quat=src["ee_quat"][gl],
             q_goal=src["q"][gl])
    print(f"train starts {train.sum()}, train goals {train_goal.sum()}, eval {len(goals)}, tune {N_TUNE}")
    d0 = np.linalg.norm(src["ee_pos"][starts] - src["ee_pos"][goals], axis=1)
    a0 = np.degrees(2 * np.arccos(np.abs((src["ee_quat"][starts] * src["ee_quat"][goals]).sum(1)).clip(0, 1)))
    print("eval initial error: position (m) P5/P50/P95", np.percentile(d0, [5, 50, 95]).round(3),
          "orientation (deg) P50", np.median(a0).round(1))


if __name__ == "__main__":
    {"source": source, "split": split}[sys.argv[1]]()
