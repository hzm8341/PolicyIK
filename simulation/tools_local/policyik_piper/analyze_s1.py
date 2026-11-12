"""S1 read-out of the hold and smoke runs of run_s1.sh. usage (from PolicyIK/simulation, PYTHONPATH with the
repo root): python tools_local/policyik_piper/analyze_s1.py > outputs/piper/s1/s1_readout.txt"""
import numpy as np

from policyik_sim.metrics import single_attempt

O = "outputs/piper/s1/"
DRIFT_MAX = 1e-4   # m; the validity threshold of the Franka simulator check


def pct(x, qs=(50, 95, 100)):
    return " / ".join(f"{np.percentile(x, q):.4g}" for q in qs)


def hold(name, steps=None):
    z = np.load(O + name + ".npz", allow_pickle=True)
    t = (steps or z["ee_pos_t"].shape[1]) - 1
    ee = z["ee_pos_t"]
    assert np.isfinite(ee[:, :t + 1]).all() and np.isfinite(z["q_t"][:, :t + 1]).all(), name
    drift = np.linalg.norm(ee[:, t] - ee[:, 0], axis=1)
    dq = np.abs(z["q_t"][:, t] - z["z_t"][:, t])                      # measured joints vs held reference
    still = np.abs(z["q_t"][:, t] - z["q_t"][:, t - 10]).max(1)       # motion over the last 10 steps
    print(f"{name} (step {t + 1}, n={len(drift)}): EE displacement from step 1, mm, P50/P95/max: {pct(1e3 * drift)}")
    print(f"   |q - z| per joint, rad, max over tasks: {np.array2string(dq.max(0), precision=4)}")
    print(f"   still moving (> 1e-3 rad over the last 10 steps): {100 * (still > 1e-3).mean():.2f} %")
    return drift


d_ff = hold("hold_ff")
hold("hold_noff")
d60 = hold("hold_ff", 60)
for name in ("hold_ff_selfcol", "hold_ff_selfcol_evalstart"):
    d = hold(name)
    print(f"   pushed away (> {1e3 * DRIFT_MAX} mm) with self-collision on: {100 * (d > DRIFT_MAX).mean():.2f} %"
          f"   (same threshold, self-collision off, tuning starts at step 60: {100 * (d60 > DRIFT_MAX).mean():.2f} %)")
for name in ("smoke_dls", "smoke_qp"):
    z = np.load(O + name + ".npz", allow_pickle=True)
    finite = all(np.isfinite(z[k]).all() for k in ("pos_t", "ori_t", "q_t", "z_t", "ee_pos_t"))
    s = single_attempt(z)
    print(f"{name}: all finite {finite}; strict {s['strict']:.2f} %, coarse {s['coarse']:.2f} %, "
          f"median position {s['pos_median_mm']:.3f} mm, median rotation {s['ori_median_deg']:.3f} deg")
    q, lim = z["q_t"][:, -30:], z["soft_limits"]
    osc = (q.max(1) - q.min(1)).max(1)
    print(f"   joint swing over the last 30 steps > 0.02 rad: {100 * (osc > 0.02).mean():.2f} % of tasks")
