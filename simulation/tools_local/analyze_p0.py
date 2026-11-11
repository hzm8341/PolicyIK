"""Local P0 analysis (not part of the official repo logic): per-task CSV + failure classes.

Reads the official evaluation dumps (single.npz, corrected.npz, transient*.npz) of each seed dir and
reuses policyik_sim.metrics for every definition, so numbers stay identical to the official metrics.

usage: python tools_local/analyze_p0.py --seed_dir outputs/eval/s1 --seed_dir outputs/eval/s2 ... \
           --task_file data/eval_tasks.npz --csv reports/policyik_errors.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from policyik_sim.metrics import (CO, CP, CORRECTED, GATE, N_TRANSIENT, SINGLE, SO, SP, TAIL, TRANSIENT,
                                  _qang, _tail_means, corrected_vs_original)

# mutually exclusive failure classes, decided only from end-effector error time series
# (joint states are not dumped by evaluate.py, so joint-limit / high-rotation classes cannot be assigned here)
CLASSES = ["success_contained", "success_mean_only", "invalid_nonfinite", "far_from_goal",
           "near_miss_never_entered", "entered_then_drifted", "oscillating"]


def classify(pos_t: np.ndarray, ori_t: np.ndarray) -> np.ndarray:
    n = pos_t.shape[0]
    out = np.empty(n, dtype=object)
    p, o = pos_t[:, -TAIL:].mean(1), ori_t[:, -TAIL:].mean(1)
    inst = (pos_t < SP) & (ori_t < SO)
    strict = (p < SP) & (o < SO)
    contained = inst[:, -TAIL:].all(1)
    entered = inst.any(1)
    finite = np.isfinite(pos_t).all(1) & np.isfinite(ori_t).all(1)
    # oscillation proxy: tail position std larger than the strict threshold while the mean is within coarse range
    tail_std = pos_t[:, -TAIL:].std(1)
    for i in range(n):
        if not finite[i]:
            out[i] = "invalid_nonfinite"
        elif strict[i]:
            out[i] = "success_contained" if contained[i] else "success_mean_only"
        elif p[i] >= CP or o[i] >= CO:
            out[i] = "far_from_goal"
        elif tail_std[i] > SP:
            out[i] = "oscillating"
        elif entered[i]:
            out[i] = "entered_then_drifted"
        else:
            out[i] = "near_miss_never_entered"
    return out


def deployment_per_task(body, corr, transients):
    """Same protocol as policyik_sim.metrics.deployment, returned per task."""
    pb, ob = _tail_means(body)
    pc, oc = corrected_vs_original(body, corr)
    pos_err = np.stack([pb, pc] + [_tail_means(w)[0] for w in transients])
    ori_err = np.stack([ob, oc] + [_tail_means(w)[1] for w in transients])
    strict = (pos_err < SP) & (ori_err < SO)
    n = pos_err.shape[1]
    executed = np.zeros_like(strict)
    covered = strict[0].copy()
    executed[0] = True
    call = (~covered) & (pb <= GATE)
    executed[1] = call
    covered |= call & strict[1]
    for k in range(2, pos_err.shape[0]):
        executed[k] = ~covered
        covered |= executed[k] & strict[k]
    first_ok = np.where((executed & strict).any(0), np.argmax(executed & strict, 0), -1)
    return covered, executed.sum(0), first_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dir", action="append", required=True)
    ap.add_argument("--task_file", default="data/eval_tasks.npz")
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    tasks = np.load(a.task_file)
    rows, per_seed = [], {}
    for sd in a.seed_dir:
        seed = os.path.basename(os.path.normpath(sd))
        body = np.load(os.path.join(sd, SINGLE))
        pos_t, ori_t = body["pos_t"].astype(np.float64), body["ori_t"].astype(np.float64)
        p, o = pos_t[:, -TAIL:].mean(1), ori_t[:, -TAIL:].mean(1)
        cls = classify(pos_t, ori_t)
        inst = (pos_t < SP) & (ori_t < SO)
        entry_t = np.where(inst.any(1), (np.argmax(inst, 1) + 1) / 30.0, np.nan)
        ee_p, ee_q = body["ee_pos_t"].astype(np.float64), body["ee_quat_t"].astype(np.float64)
        drift_p = np.linalg.norm(ee_p[:, -1] - ee_p[:, -TAIL - 1], axis=1)
        drift_o = _qang(ee_q[:, -1], ee_q[:, -TAIL - 1])
        has_deploy = os.path.exists(os.path.join(sd, CORRECTED))
        if has_deploy:
            corr = np.load(os.path.join(sd, CORRECTED))
            trs = [np.load(os.path.join(sd, TRANSIENT.format(k))) for k in range(N_TRANSIENT)]
            dep_ok, n_att, first_ok = deployment_per_task(body, corr, trs)
        # start-to-goal distance (diagnostic stratifier): FK of q_start is not dumped, use the first-step error
        init_pos, init_ori = pos_t[:, 0], ori_t[:, 0]
        per_seed[seed] = dict(cls=cls, strict=(p < SP) & (o < SO))
        for i in range(pos_t.shape[0]):
            rows.append(dict(
                seed=seed, task_id=i,
                pos_tail_mean_mm=1e3 * p[i], ori_tail_mean_deg=np.degrees(o[i]),
                pos_tail_max_mm=1e3 * pos_t[i, -TAIL:].max(), ori_tail_max_deg=np.degrees(ori_t[i, -TAIL:].max()),
                pos_final_mm=1e3 * pos_t[i, -1], ori_final_deg=np.degrees(ori_t[i, -1]),
                strict=int(p[i] < SP and o[i] < SO), coarse=int(p[i] < CP and o[i] < CO),
                containment=int(inst[i, -TAIL:].all()), entry_time_s=entry_t[i],
                drift_um=1e6 * drift_p[i], drift_deg=np.degrees(drift_o[i]),
                pos_step1_mm=1e3 * init_pos[i], ori_step1_deg=np.degrees(init_ori[i]),
                failure_class=cls[i],
                deploy_strict=int(dep_ok[i]) if has_deploy else "",
                deploy_attempts=int(n_att[i]) if has_deploy else "",
                deploy_success_attempt=("single" if first_ok[i] == 0 else "corrected" if first_ok[i] == 1 else
                                        f"transient{first_ok[i] - 2}" if first_ok[i] > 1 else "none")
                if has_deploy else "",
                q_start=" ".join(f"{v:.4f}" for v in tasks["q_start"][i]),
            ))
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {a.csv} ({len(rows)} rows)")

    print("\nFailure classes (single attempt, count per seed)")
    print(f"  {'class':28s} " + " ".join(f"{s:>6s}" for s in per_seed))
    for c in CLASSES:
        print(f"  {c:28s} " + " ".join(f"{int((v['cls'] == c).sum()):6d}" for v in per_seed.values()))
    if len(per_seed) > 1:
        S = np.stack([v["strict"] for v in per_seed.values()])
        nf = (~S).sum(0)
        print("\nCross-seed consistency of single-attempt strict failures (same task)")
        for k in range(S.shape[0] + 1):
            print(f"  tasks failing in exactly {k} seed(s): {int((nf == k).sum())}")


if __name__ == "__main__":
    main()
