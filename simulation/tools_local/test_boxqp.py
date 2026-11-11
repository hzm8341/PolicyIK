"""Check the batched GPU box-QP against SciPy BVLS (the solver named in the paper, Appendix I)."""
import sys
import time

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from boxqp import solve_box_qp  # noqa: E402

rng = np.random.default_rng(0)
B, n = 2000, 7
W = np.diag([1, 1, 1, 0.2, 0.2, 0.2])
J = rng.normal(size=(B, 6, n)) * 0.3
# make some Jacobians near-singular
J[: B // 4, :, 6] = J[: B // 4, :, 5] * (1 + 1e-4)
e = rng.normal(size=(B, 6)) * rng.choice([1e-4, 1e-2, 0.3], size=(B, 1))
k = 2.0
lam = rng.choice([0.005, 0.02], size=B)
z_room = rng.uniform(0.0, 0.3, size=(B, n))
lb = -rng.uniform(0, 1, size=(B, n)) * rng.choice([0.01, 1, 10], size=(B, 1))
ub = rng.uniform(0, 1, size=(B, n)) * rng.choice([0.01, 1, 10], size=(B, 1))
# some boxes not containing zero (reference outside its lead window)
lb[: B // 10] += 0.05
ub[: B // 10] += 0.05

A = np.concatenate([W @ J, lam[:, None, None] * np.eye(n)[None]], axis=1)       # (B, 13, n)
b = np.concatenate([k * (W @ e[..., None])[..., 0], np.zeros((B, n))], axis=1)
H = np.einsum("bij,bik->bjk", A, A)
g = np.einsum("bij,bi->bj", A, b)

dev = "cuda"
t0 = time.time()
x, res = solve_box_qp(*(torch.as_tensor(v, device=dev, dtype=torch.float64) for v in (H, g, lb, ub)))
torch.cuda.synchronize()
t_gpu = time.time() - t0
x = x.cpu().numpy()

t0 = time.time()
xs = np.stack([lsq_linear(A[i], b[i], bounds=(lb[i], ub[i]), method="bvls", tol=1e-10, max_iter=100).x
               for i in range(B)])
t_cpu = time.time() - t0

def obj(v):
    r = np.einsum("bij,bj->bi", A, v) - b
    return 0.5 * (r ** 2).sum(-1)

dobj = obj(x) - obj(xs)
print(f"GPU {t_gpu*1e3:.1f} ms for {B}; BVLS {t_cpu*1e3:.1f} ms")
print(f"max |x - x_bvls| = {np.abs(x - xs).max():.3e}; max rel = {(np.abs(x - xs).max(1) / (np.abs(xs).max(1) + 1e-9)).max():.3e}")
print(f"objective gap (gpu - bvls): max {dobj.max():.3e}, min {dobj.min():.3e}")
print(f"max KKT residual {res.max().item():.3e}; bounds ok: {bool(((x >= lb - 1e-12) & (x <= ub + 1e-12)).all())}")
assert dobj.max() <= 1e-9 * max(1.0, np.abs(obj(xs)).max()), "GPU solution worse than BVLS"
print("BOXQP OK")
