"""P1 figures: share of tasks within 1 mm & 0.2 deg at each step, and terminal-error CDFs.

usage: python tools_local/plot_p1.py --run NAME=path.npz ... --out reports/p1   (NAME@seed groups seeds)
"""
import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from policyik_sim.metrics import SO, SP, TAIL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True)
ap.add_argument("--out", default="reports/p1")
a = ap.parse_args()
groups = OrderedDict()
for r in a.run:
    name, path = r.split("=", 1)
    groups.setdefault(name.split("@")[0], []).append(np.load(path))
os.makedirs(a.out, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for ctrl, runs in groups.items():
    inside = np.mean([((z["pos_t"] < SP) & (z["ori_t"] < SO)).mean(0) for z in runs], axis=0) * 100
    t = (np.arange(inside.size) + 1) / 30.0
    ax[0].plot(t, inside, label=ctrl)
    p = np.sort(np.concatenate([z["pos_t"][:, -TAIL:].mean(1) for z in runs])) * 1e3
    o = np.sort(np.concatenate([np.degrees(z["ori_t"][:, -TAIL:].mean(1)) for z in runs]))
    y = np.linspace(0, 100, p.size)
    ax[1].plot(p, y, label=ctrl)
    ax[2].plot(o, y, label=ctrl)
ax[0].axvspan(t[-TAIL], t[-1], color="0.9")
ax[0].set(xlabel="time (s)", ylabel="tasks within 1 mm & 0.2° (%)", title="In-tolerance share per step", ylim=(0, 100))
ax[1].axvline(1.0, color="k", lw=0.8, ls="--")
ax[1].set(xscale="log", xlabel="final-second mean position error (mm)", ylabel="CDF (%)", title="Terminal position error")
ax[2].axvline(0.2, color="k", lw=0.8, ls="--")
ax[2].set(xscale="log", xlabel="final-second mean orientation error (deg)", ylabel="CDF (%)",
          title="Terminal orientation error")
for x in ax:
    x.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "p1_curves.png"), dpi=130)
print("wrote", os.path.join(a.out, "p1_curves.png"))
