"""Plan S0: Piper source set, training pools, frozen test set and disjoint tuning set (no simulator needed).

Mirrors the OpenArm data pipeline, as tools_local/policyik_franka/make_franka_tasks.py does (policyik_sim/splits.py
split() and check() reused unchanged, with the Piper workspace constants patched in):
  * 200,000 joint configurations uniform within the URDF joint limits (seed 20261111) -- the full limits, as in
    the OpenArm source set, including joint2's lower and joint3's upper limit at exactly 0;
  * poses of link6 (the flange) in the base_link frame, independent numpy FK from the URDF;
  * floor_contact: a collision-mesh hull vertex below the mounting plane z = 0 (base_link and link1 excluded);
  * self_collision: NOT screened, all False (plan section 6, choice 4); ``self_collision_screened`` says so.
    S1 measures the self-collision rate of these configurations in the simulator before the sets are used;
  * deployment workspace: the OpenArm constants do not transfer, so each one is placed at the same quantile of
    the Piper's valid set as the OpenArm value has in the OpenArm valid set (r: 0.962 and 0.092; z: 0.110, 0.808
    and 0.244), rounded to 5 mm. An engineering choice, documented in reports/piper/s0_summary.md;
  * 4,120 test tasks and 1,024 tuning tasks from the training pools (disjoint from the test set).
Writes outputs/piper/data/{source_set,train_goal_pool,train_start_pool,eval_tasks,eval_goal_q,tune_tasks}.npz and
manifest.json (sha256 of every array; npz files carry timestamps, so the arrays are what is compared).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from piper_kin import ARM_JOINTS, EE_LINK, FINGER_POS, URDF, PiperKinematics  # noqa: E402

import policyik_sim.splits as S  # noqa: E402

N_SRC, SEED, TUNE_SEED, N_TUNE = 200_000, 20261111, 20261112, 1024
WORKSPACE = dict(R_MAX=0.585, Z_MIN=0.185, Z_MAX=0.625, R_KEEPOUT=0.065, Z_KEEPOUT=0.285)
OUT = "outputs/piper/data"


def build(n_src: int = N_SRC, kin: PiperKinematics | None = None) -> dict:
    """-> {file stem: {array name: array}} for the six data files."""
    for name, value in WORKSPACE.items():
        setattr(S, name, value)
    kin = kin or PiperKinematics()
    q = np.random.default_rng(SEED).uniform(kin.limits[:, 0], kin.limits[:, 1], (n_src, len(ARM_JOINTS)))
    pos, quat = kin.fk(q)
    fc = kin.floor_contact(q)
    src = dict(q=q.astype(np.float32), ee_pos=pos.astype(np.float32), ee_quat=quat.astype(np.float32),
               floor_contact=fc, self_collision=np.zeros(n_src, bool), self_collision_screened=np.array(False))
    goals, starts, train, train_goal = S.split(src)
    S.check(src, goals, starts, train)
    rng = np.random.default_rng(TUNE_SEED)
    st = np.where(train)[0][rng.choice(train.sum(), N_TUNE, replace=False)]
    gl = np.where(train_goal)[0][rng.choice(train_goal.sum(), N_TUNE, replace=False)]
    assert not (set(st) | set(gl)) & (set(goals) | set(starts))
    pose = ("q", "ee_pos", "ee_quat")
    return {
        "source_set": src,
        "train_start_pool": {k: src[k][train] for k in pose},
        "train_goal_pool": {k: src[k][train_goal] for k in pose},
        "eval_tasks": dict(q_start=src["q"][starts], goal_pos=src["ee_pos"][goals], goal_quat=src["ee_quat"][goals]),
        "eval_goal_q": dict(q_goal=src["q"][goals]),
        "tune_tasks": dict(q_start=src["q"][st], goal_pos=src["ee_pos"][gl], goal_quat=src["ee_quat"][gl],
                           q_goal=src["q"][gl]),
    }


def array_hashes(files: dict) -> dict:
    return {stem: {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}
            for stem, arrays in files.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    files = build()
    for stem, arrays in files.items():
        np.savez(f"{OUT}/{stem}.npz", **arrays)
    src = files["source_set"]
    manifest = dict(
        urdf=URDF, urdf_sha256=hashlib.sha256(open(URDF, "rb").read()).hexdigest(), ee_link=EE_LINK,
        arm_joints=ARM_JOINTS, finger_pos=FINGER_POS, n_src=N_SRC, seed=SEED, tune_seed=TUNE_SEED,
        workspace=WORKSPACE, self_collision_screened=False,
        counts={stem: int(len(next(iter(arrays.values())))) for stem, arrays in files.items()},
        arrays_sha256=array_hashes(files))
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    valid = ~src["floor_contact"]
    print(f"valid {valid.mean():.3f} (floor {src['floor_contact'].mean():.3f}; self-collision not screened); "
          f"in workspace among valid {S.in_workspace(src['ee_pos'][valid].astype(float)).mean():.3f}; "
          + ", ".join(f"{k} {v}" for k, v in manifest["counts"].items()))
    ev = files["eval_tasks"]
    start_pos = PiperKinematics().fk(ev["q_start"].astype(float))[0]
    d0 = np.linalg.norm(start_pos - ev["goal_pos"], axis=1)
    print("eval initial position error (m) P5/P50/P95:", np.percentile(d0, [5, 50, 95]).round(3))


if __name__ == "__main__":
    main()
