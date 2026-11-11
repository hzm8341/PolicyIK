"""P2 step 1: Franka source set, training pools, frozen test set and disjoint tuning set (no simulator needed).

Mirrors the OpenArm data pipeline (policyik_sim/splits.py, whose split() and check() are reused unchanged with
Franka workspace constants patched in):
  * 200,000 joint configurations uniform within the URDF joint limits (seed 20261111),
  * FK poses of panda_hand in the panda_link0 frame (independent numpy FK from the URDF),
  * self_collision / floor_contact flags from the curobo sphere model (floor = table plane z = 0),
  * deployment workspace: cylinder r <= 0.75 m, 0.05 <= z <= 0.75 m, minus r < 0.25 m & z < 0.45 m (engineering
    choice, documented in the P2 report; the OpenArm values do not transfer),
  * 4,120 test tasks (as OpenArm, well above the >=170 gate minimum), 1,024 tuning tasks from the training pools.
Writes outputs/p2/data/{source_set,train_goal_pool,train_start_pool,eval_tasks,tune_tasks,eval_goal_q}.npz.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from urdf_fk import FrankaKinematics, SphereCollision  # noqa: E402

import policyik_sim.splits as S  # noqa: E402

N_SRC, SEED, TUNE_SEED, N_TUNE = 200_000, 20261111, 20261112, 1024
S.R_MAX, S.Z_MIN, S.Z_MAX, S.R_KEEPOUT, S.Z_KEEPOUT = 0.75, 0.05, 0.75, 0.25, 0.45
OUT = "outputs/p2/data"


def main():
    os.makedirs(OUT, exist_ok=True)
    kin = FrankaKinematics()
    col = SphereCollision(kin)
    q = np.random.default_rng(SEED).uniform(kin.limits[:, 0], kin.limits[:, 1], (N_SRC, 7))
    pos, quat = [], []
    for s in range(0, N_SRC, 20000):
        p, o = kin.fk(q[s:s + 20000])
        pos.append(p)
        quat.append(o)
    pos, quat = np.concatenate(pos), np.concatenate(quat)
    sc, fc = col.check(q)
    src = dict(q=q.astype(np.float32), ee_pos=pos.astype(np.float32), ee_quat=quat.astype(np.float32),
               floor_contact=fc, self_collision=sc)
    np.savez(f"{OUT}/source_set.npz", **src)
    goals, starts, train, train_goal = S.split(src)
    S.check(src, goals, starts, train)
    np.savez(f"{OUT}/train_start_pool.npz", **{k: src[k][train] for k in ("q", "ee_pos", "ee_quat")})
    np.savez(f"{OUT}/train_goal_pool.npz", **{k: src[k][train_goal] for k in ("q", "ee_pos", "ee_quat")})
    np.savez(f"{OUT}/eval_tasks.npz", q_start=src["q"][starts], goal_pos=src["ee_pos"][goals],
             goal_quat=src["ee_quat"][goals])
    np.savez(f"{OUT}/eval_goal_q.npz", q_goal=src["q"][goals])
    # tuning set: starts from the start pool, goals from the goal pool (both disjoint from the test set)
    rng = np.random.default_rng(TUNE_SEED)
    st = np.where(train)[0][rng.choice(train.sum(), N_TUNE, replace=False)]
    gl = np.where(train_goal)[0][rng.choice(train_goal.sum(), N_TUNE, replace=False)]
    assert not (set(st) | set(gl)) & (set(goals) | set(starts))
    np.savez(f"{OUT}/tune_tasks.npz", q_start=src["q"][st], goal_pos=src["ee_pos"][gl], goal_quat=src["ee_quat"][gl],
             q_goal=src["q"][gl])
    print(f"valid {(~sc & ~fc).mean():.3f} (self {sc.mean():.3f}, floor {fc.mean():.3f}); "
          f"train starts {train.sum()}, train goals {train_goal.sum()}, eval {len(goals)}, tune {N_TUNE}")
    d0 = np.linalg.norm(src["ee_pos"][starts] - src["ee_pos"][goals], axis=1)
    print("eval initial position error (m) P5/P50/P95:", np.percentile(d0, [5, 50, 95]).round(3))


if __name__ == "__main__":
    main()
