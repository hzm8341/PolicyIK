"""Compare evaluator dumps pairwise, bit for bit (timing and the config string excluded).
usage: compare_npz.py a1 b1 [a2 b2 ...]; exit 1 if a pair differs."""
import sys

import numpy as np

SKIP = ("ctrl_time_per_step_s", "config")
bad = 0
for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    keys = sorted(set(x.files) | set(y.files))
    diff = [k for k in keys if k not in SKIP and not (
        k in x.files and k in y.files and x[k].shape == y[k].shape and np.array_equal(x[k], y[k], equal_nan=True))]
    worst = max((float(np.nanmax(np.abs(x[k] - y[k]))) for k in diff
                 if k in x.files and k in y.files and x[k].shape == y[k].shape), default=0.0)
    print(f"{a} vs {b}: " + ("identical" if not diff else f"DIFFERENT in {diff}, max |diff| {worst:.3e}"))
    bad += bool(diff)
sys.exit(1 if bad else 0)
