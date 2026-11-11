"""Summarize the P1 tuning grid (tuning set only) and select (k, lead) per controller by strict success.

Tie-break: coarse success, then median terminal position error.
usage: python tools_local/summarize_tuning.py outputs/p1/tune  -> prints the table, writes selection.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

from policyik_sim.metrics import single_attempt

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
    name = os.path.basename(f)[:-4]
    z = np.load(f)
    s = single_attempt(z)
    m = re.match(r"(dls|qp|mpc)_k([\d.]+)_lead([\d.]+)(?:_tr([\d.]+))?$", name)
    s = {kk: float(vv) for kk, vv in s.items()}
    rows.append(dict(name=name, ctrl=m.group(1) if m else name, k=float(m.group(2)) if m else None,
                     lead=float(m.group(3)) if m else None, trust=float(m.group(4)) if m and m.group(4) else None,
                     strict=s["strict"], coarse=s["coarse"],
                     containment=s["containment"], pos_med=s["pos_median_mm"], ori_med=s["ori_median_deg"]))
print(f"{'config':26s} {'strict':>7s} {'coarse':>7s} {'contain':>7s} {'pos_med':>8s} {'ori_med':>8s}")
for r in rows:
    print(f"{r['name']:26s} {r['strict']:7.2f} {r['coarse']:7.2f} {r['containment']:7.2f} {r['pos_med']:8.3f} {r['ori_med']:8.3f}")
sel = {}
for c in ("dls", "qp", "mpc"):
    rs = [r for r in rows if r["ctrl"] == c]
    if not rs:
        continue
    best = max(rs, key=lambda r: (round(r["strict"], 6), round(r["coarse"], 6), -r["pos_med"]))
    sel[c] = best
    print(f"\nselected {c}: k={best['k']} lead={best['lead']} trust={best['trust']} strict={best['strict']:.2f} coarse={best['coarse']:.2f}")
    # strict-success grid
    if c == "mpc":
        for tr in sorted({r["trust"] for r in rs}):
            print(f"  trust={tr}: " + ", ".join(f"k{r['k']:g}/lead{r['lead']:g}={r['strict']:.2f}"
                                               for r in rs if r["trust"] == tr))
        continue
    ks = sorted({r["k"] for r in rs}); ls = sorted({r["lead"] for r in rs})
    print("  strict grid (rows k, cols lead): " + " ".join(f"{l:>6g}" for l in ls))
    for k in ks:
        vals = {r["lead"]: r["strict"] for r in rs if r["k"] == k}
        print(f"  k={k:<5g}" + " " * 25 + " ".join(f"{vals.get(l, float('nan')):6.2f}" for l in ls))
json.dump(sel, open(os.path.join(d, "selection.json"), "w"), indent=1)
