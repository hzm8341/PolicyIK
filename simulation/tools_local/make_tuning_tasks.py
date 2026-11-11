"""Build the P1 tuning task set (disjoint from eval_tasks) and recover the goal configurations of eval_tasks.

Tuning tasks: starts from data/train_start_pool.npz, goals from data/train_goal_pool.npz. Both pools exclude every
evaluation configuration (policyik_sim/splits.py), so the tuning set follows the test distribution (valid starts,
goals in the deployment workspace) and is disjoint from it (paper Appendix E).
Eval goal configurations: re-derived with policyik_sim.splits.split() and checked against eval_tasks.npz.
"""
import numpy as np

from policyik_sim.splits import split

N_TUNE, SEED = 1024, 20261111

st = np.load("data/train_start_pool.npz")
gp = np.load("data/train_goal_pool.npz")
rng = np.random.default_rng(SEED)
si = rng.choice(len(st["q"]), N_TUNE, replace=False)
gi = rng.choice(len(gp["q"]), N_TUNE, replace=False)
np.savez("outputs/p1/tune_tasks.npz",
         q_start=st["q"][si], goal_pos=gp["ee_pos"][gi], goal_quat=gp["ee_quat"][gi], q_goal=gp["q"][gi],
         start_rows=si, goal_rows=gi)

ev = np.load("data/eval_tasks.npz")
with np.load("data/source_set.npz") as z:
    src = {k: z[k] for k in z.files}
goals, starts, train, _ = split(src)
assert np.array_equal(src["q"][starts], ev["q_start"])
assert np.allclose(src["ee_pos"][goals], ev["goal_pos"]) and np.allclose(src["ee_quat"][goals], ev["goal_quat"])
# disjointness of the tuning set from evaluation configurations
ev_q = {r.tobytes() for r in np.concatenate([src["q"][starts], src["q"][goals]])}
assert not any(r.tobytes() in ev_q for r in np.concatenate([st["q"][si], gp["q"][gi]]))
np.savez("outputs/p1/eval_goal_q.npz", q_goal=src["q"][goals], goal_rows=goals, start_rows=starts)
print(f"tuning tasks {N_TUNE} -> outputs/p1/tune_tasks.npz; eval goal configurations verified -> outputs/p1/eval_goal_q.npz")
