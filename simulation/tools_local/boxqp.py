"""Batched box-constrained convex QP:  min_x 0.5 x^T H x - g^T x  s.t.  lb <= x <= ub.

Used for the bounded damped least-squares step of the classical controllers (paper Appendix E / I):
    min_qd ||W J qd - k W e||^2 + lam^2 ||qd||^2   s.t. qd in the feasible joint-velocity box,
i.e. H = (WJ)^T WJ + lam^2 I,  g = k (WJ)^T W e.  The paper solves it with SciPy BVLS on the hardware;
here a batched projected-Newton method (Bertsekas 1982) runs on the GPU and is checked against BVLS
(tools_local/test_boxqp.py).
"""
from __future__ import annotations

import torch


def solve_box_qp(H: torch.Tensor, g: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor,
                 max_iter: int = 60, tol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """H (B,n,n) SPD, g/lb/ub (B,n). Returns (x, kkt_residual)."""
    B, n = g.shape
    dt = H.dtype
    eye = torch.eye(n, dtype=dt, device=H.device).expand(B, n, n)
    lb = torch.minimum(lb, ub)
    x = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)          # unconstrained minimiser
    x = torch.clamp(x, lb, ub)

    def f(v):
        return 0.5 * (v.unsqueeze(1) @ H @ v.unsqueeze(-1)).squeeze(-1).squeeze(-1) - (g * v).sum(-1)

    eps_bnd = 1e-12
    for _ in range(max_iter):
        grad = (H @ x.unsqueeze(-1)).squeeze(-1) - g
        # projected-gradient KKT residual
        pg = x - torch.clamp(x - grad, lb, ub)
        res = pg.abs().amax(-1)
        if bool((res <= tol).all()):
            break
        # active set: at a bound with the gradient pushing outward
        w = torch.minimum(pg.abs().amax(-1, keepdim=True), torch.full_like(res.unsqueeze(-1), 1e-3))
        act = ((x <= lb + w + eps_bnd) & (grad > 0)) | ((x >= ub - w - eps_bnd) & (grad < 0))
        free = (~act).to(dt)
        # reduced Newton direction on the free set (identity on the active set -> gradient step)
        Hr = H * free.unsqueeze(-1) * free.unsqueeze(-2) + eye * (1.0 - free).unsqueeze(-1)
        d = -torch.linalg.solve(Hr, (grad * free).unsqueeze(-1)).squeeze(-1) - grad * (1.0 - free)
        # Armijo backtracking along the projection arc
        f0 = f(x)
        alpha = torch.ones(B, dtype=dt, device=H.device)
        x_new = torch.clamp(x + d, lb, ub)
        for _ls in range(30):
            dec = f0 - f(x_new)
            need = 1e-4 * ((grad * (x - x_new)).sum(-1))
            ok = dec >= need - 1e-15
            if bool(ok.all()):
                break
            alpha = torch.where(ok, alpha, alpha * 0.5)
            x_new = torch.clamp(x + alpha.unsqueeze(-1) * d, lb, ub)
        x = x_new
    grad = (H @ x.unsqueeze(-1)).squeeze(-1) - g
    res = (x - torch.clamp(x - grad, lb, ub)).abs().amax(-1)
    return x, res
