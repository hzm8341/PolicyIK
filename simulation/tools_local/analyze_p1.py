"""P1 analysis: PolicyIK vs classical controllers on the same 4,120 eval tasks (single attempt, same budget).

usage: python tools_local/analyze_p1.py --run NAME=path.npz [--run ...] --ref policyik --out reports/p1
  Runs that share a NAME prefix before '@' (e.g. policyik@s1, policyik@s2) are seeds of one controller.

Reports per controller (per seed, then mean +- SD): strict / coarse / containment / entry, P50/P90/P95 terminal
errors, joint-limit stall rate (paper App. E: coarse failure and a joint within 0.01 rad of a soft limit for at
least half of the final second), reference-step statistics and controller time.
Paired comparison vs the reference controller: task-level strict-success difference with a paired bootstrap 95% CI
(tasks resampled, per-task success averaged over the seeds of each controller, so the shared tasks are not
treated as independent across seeds), plus discordant-pair counts.
Stratification (diagnostic only): goal joint-limit margin quartiles and initial position-error bins.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np

from policyik_sim.metrics import CO, CP, SO, SP, TAIL, single_attempt

STALL_TOL, STALL_FRAC = 0.01, 0.5
SOFT_FALLBACK = None


def per_task(z) -> dict:
    p, o = z["pos_t"][:, -TAIL:].mean(1), z["ori_t"][:, -TAIL:].mean(1)
    strict = (p < SP) & (o < SO)
    coarse = (p < CP) & (o < CO)
    out = dict(p=p, o=o, strict=strict, coarse=coarse)
    if "q_t" in z.files:
        lim = z["soft_limits"]
        q = z["q_t"][:, -TAIL:]
        near = (np.minimum(q - lim[:, 0], lim[:, 1] - q) < STALL_TOL)          # (n, 30, 7)
        stall_joint = near.mean(1) >= STALL_FRAC
        out["near_limit_final"] = stall_joint.any(1)
        out["stall"] = ~coarse & out["near_limit_final"]
        dz = np.diff(z["z_t"], axis=1)
        out["dz_max"] = np.abs(dz).max((1, 2))
        out["dz_rms"] = np.sqrt((dz ** 2).sum(2).mean(1))
    return out


def summary(z) -> dict:
    s = single_attempt(z)
    t = per_task(z)
    r = OrderedDict(strict=s["strict"], coarse=s["coarse"], containment=s["containment"], entry=s["entry"],
                    entry_time_s=s["entry_time_s"],
                    pos_p50_mm=1e3 * np.median(t["p"]), pos_p90_mm=1e3 * np.percentile(t["p"], 90),
                    pos_p95_mm=1e3 * np.percentile(t["p"], 95),
                    ori_p50_deg=np.degrees(np.median(t["o"])), ori_p90_deg=np.degrees(np.percentile(t["o"], 90)),
                    ori_p95_deg=np.degrees(np.percentile(t["o"], 95)),
                    drift_um=s["drift_um"])
    if "stall" in t:
        fails = ~t["strict"]
        r["stall_pct"] = 100 * t["stall"].mean()
        r["stall_share_of_strict_fail_pct"] = 100 * (t["stall"] & fails).sum() / max(fails.sum(), 1)
        r["dz_step_max_rad_p99"] = np.percentile(t["dz_max"], 99)
        r["dz_rms_rad_median"] = np.median(t["dz_rms"])
    if "ctrl_time_per_step_s" in z.files:
        r["ctrl_ms_per_step_batch"] = 1e3 * np.median(z["ctrl_time_per_step_s"])
    return r


def boot_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ref", default="policyik")
    ap.add_argument("--goal_q", default="outputs/p1/eval_goal_q.npz")
    ap.add_argument("--out", default="reports/p1")
    a = ap.parse_args()
    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in a.run:
        name, path = r.split("=", 1)
        ctrl = name.split("@")[0]
        groups.setdefault(ctrl, []).append((name, np.load(path)))
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # ---- per-controller summary
    table = OrderedDict()
    for ctrl, runs in groups.items():
        sums = [summary(z) for _, z in runs]
        table[ctrl] = sums
    keys = list(next(iter(table.values()))[0].keys())
    for v in table.values():
        for s in v:
            for k in s:
                if k not in keys:
                    keys.append(k)
    emit("## Single-attempt summary (mean +- SD over seeds where >1 run)\n")
    emit("| metric | " + " | ".join(f"{c} (n={len(v)})" for c, v in table.items()) + " |")
    emit("|---|" + "---|" * len(table))
    for k in keys:
        cells = []
        for c, v in table.items():
            vals = np.array([s.get(k, np.nan) for s in v], float)
            if np.all(np.isnan(vals)):
                cells.append("–")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{np.nanmean(vals):.3f} ± {np.nanstd(vals, ddof=1):.3f}")
        emit(f"| {k} | " + " | ".join(cells) + " |")

    # ---- paired comparison vs reference
    if a.ref in groups:
        ref_succ = np.mean([per_task(z)["strict"] for _, z in groups[a.ref]], axis=0)
        ref_coarse = np.mean([per_task(z)["coarse"] for _, z in groups[a.ref]], axis=0)
        emit(f"\n## Paired comparison vs {a.ref} (task-level, per-task success averaged over seeds; "
             f"bootstrap over tasks, 10k resamples)\n")
        emit("| controller | Δstrict (ref − ctrl) pp | 95% CI | Δcoarse pp | 95% CI | ref-only strict | ctrl-only strict |")
        emit("|---|---|---|---|---|---|---|")
        for ctrl, runs in groups.items():
            if ctrl == a.ref:
                continue
            cs = np.mean([per_task(z)["strict"] for _, z in runs], axis=0)
            cc = np.mean([per_task(z)["coarse"] for _, z in runs], axis=0)
            d = ref_succ - cs
            dc = ref_coarse - cc
            lo, hi = boot_ci(d)
            clo, chi = boot_ci(dc, seed=1)
            emit(f"| {ctrl} | {100 * d.mean():.2f} | [{100 * lo:.2f}, {100 * hi:.2f}] | {100 * dc.mean():.2f} | "
                 f"[{100 * clo:.2f}, {100 * chi:.2f}] | {int(((ref_succ >= 0.5) & (cs < 0.5)).sum())} | "
                 f"{int(((ref_succ < 0.5) & (cs >= 0.5)).sum())} |")

    # ---- stratification by goal joint-limit margin and initial distance
    if os.path.exists(a.goal_q):
        gq = np.load(a.goal_q)["q_goal"]
        any_run = next(iter(groups.values()))[0][1]
        lim = any_run["soft_limits"] if "soft_limits" in any_run.files else None
        if lim is not None and len(gq) == len(any_run["pos_t"]):
            margin = (np.minimum(gq - lim[:, 0], lim[:, 1] - gq) / (lim[:, 1] - lim[:, 0])).min(1)
            qs = np.percentile(margin, [25, 50, 75])
            bins = np.digitize(margin, qs)
            labels = [f"Q1 margin<{100 * qs[0]:.1f}%", f"Q2 <{100 * qs[1]:.1f}%", f"Q3 <{100 * qs[2]:.1f}%",
                      f"Q4 ≥{100 * qs[2]:.1f}%"]
            emit("\n## Strict success (%) by goal joint-limit margin quartile (min_j normalized distance of the goal "
                 "configuration to a soft limit)\n")
            emit("| stratum | n | " + " | ".join(groups) + " |")
            emit("|---|---|" + "---|" * len(groups))
            for b, lab in enumerate(labels):
                msk = bins == b
                cells = [f"{100 * np.mean([per_task(z)['strict'][msk].mean() for _, z in runs]):.1f}"
                         for runs in groups.values()]
                emit(f"| {lab} | {int(msk.sum())} | " + " | ".join(cells) + " |")
            d0 = any_run["pos_t"][:, 0]
            edges = [0, 0.1, 0.2, 0.3, 0.4, 10]
            emit("\n## Strict success (%) by initial position error (m)\n")
            emit("| initial error | n | " + " | ".join(groups) + " |")
            emit("|---|---|" + "---|" * len(groups))
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                msk = (d0 >= lo_) & (d0 < hi_)
                if msk.sum() == 0:
                    continue
                cells = [f"{100 * np.mean([per_task(z)['strict'][msk].mean() for _, z in runs]):.1f}"
                         for runs in groups.values()]
                emit(f"| [{lo_}, {hi_}) | {int(msk.sum())} | " + " | ".join(cells) + " |")

    with open(os.path.join(a.out, "p1_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # per-task CSV
    with open(os.path.join(a.out, "p1_per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task_id", "pos_tail_mean_mm", "ori_tail_mean_deg", "strict", "coarse", "joint_limit_stall"])
        for ctrl, runs in groups.items():
            for name, z in runs:
                t = per_task(z)
                st = t.get("stall", np.zeros_like(t["strict"]))
                for i in range(len(t["p"])):
                    w.writerow([name, i, f"{1e3 * t['p'][i]:.5g}", f"{np.degrees(t['o'][i]):.5g}", int(t["strict"][i]),
                                int(t["coarse"][i]), int(st[i])])
    print(f"\nwrote {a.out}/p1_analysis.md and p1_per_task.csv")


if __name__ == "__main__":
    main()
