"""Kinematic MPC baseline (paper Appendix I, simulation settings of Appendix E), solved per env with DAQP 0.7.2.

Decision variables: reference velocities v_0..v_{N-1} (N = 7), z_i = z + dt * sum_{j<=i} v_j.
Kinematic prediction with the Jacobian frozen at the measured q: the joints reach the previous reference,
p_0 = q, p_i = z_{i-1}, and e_i = e_0 - J (p_i - q).
Objective (W = diag(1,1,1,w,w,w)):
    sum_i ||W (J v_i - k e_i)||^2 + lam^2 ||v_i||^2      (predicted resolved-rate residuals + velocity damping)
  + s * sum_i ||v_i - v_{i-1}||^2                        (velocity smoothness, s = 0.001, v_{-1} = last applied v)
  + ||W e_N||^2                                          (unit terminal pose-error weight)
Constraints: soft joint limits on every z_i; trust region |z_i - z| <= 0.10 rad; command-to-measurement lead
|z_0 - q| <= lead; reference acceleration |v_i - v_{i-1}| <= a dt (a = 100 rad/s^2 in simulation).
Not reproduced: the hardware braking envelope and the 0.20 rad/s speed bound (the latter is not applied in
simulation, Appendix E). If DAQP reports infeasibility, the step falls back to holding the reference (counted).
"""
from __future__ import annotations

import daqp
import numpy as np


class KinematicMPC:
    def __init__(self, dt: float, k: float = 1.5, lam: float = 0.005, lead: float = 0.15, rot_w: float = 0.2,
                 horizon: int = 7, smooth: float = 1e-3, trust: float = 0.10, acc: float = 100.0,
                 terminal: float = 1.0):
        self.dt, self.k, self.lam, self.lead, self.N = dt, k, lam, lead, horizon
        self.smooth, self.trust, self.acc, self.terminal = smooth, trust, acc, terminal
        self.W = np.array([1, 1, 1, rot_w, rot_w, rot_w])
        N, n = horizon, 7
        self.nv = N * n
        # selection S_i (v_i) and cumulative L_i (sum_{j<=i} v_j) as (N, n, N*n)
        S = np.zeros((N, n, N * n))
        L = np.zeros((N, n, N * n))
        for i in range(N):
            S[i, :, i * n:(i + 1) * n] = np.eye(n)
            for j in range(i + 1):
                L[i, :, j * n:(j + 1) * n] = np.eye(n)
        self.S, self.L = S, L
        D = np.zeros((N * n, N * n))      # differences v_i - v_{i-1} (row block 0 is v_0, minus v_prev in rhs)
        for i in range(N):
            D[i * n:(i + 1) * n, i * n:(i + 1) * n] = np.eye(n)
            if i > 0:
                D[i * n:(i + 1) * n, (i - 1) * n:i * n] = -np.eye(n)
        self.D = D
        # constraint rows: z_i - z = dt L_i u (N*n rows), accelerations D u (N*n rows); simple bounds on all u
        self.A = np.concatenate([dt * L.reshape(N * n, N * n), D], axis=0)
        self.sense = np.zeros(self.nv + self.A.shape[0], dtype=np.int32)
        self.n_fallback = 0
        self.n_solves = 0

    def step(self, J: np.ndarray, e0: np.ndarray, q: np.ndarray, z: np.ndarray, v_prev: np.ndarray,
             lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched inputs (B,6,7), (B,6), (B,7)x3, lo/hi (7,). Returns (z_next, v_applied)."""
        B, N, n, dt, k = J.shape[0], self.N, 7, self.dt, self.k
        W = self.W
        WJ = W[None, :, None] * J                                         # (B,6,7)
        c = e0 - np.einsum("bij,bj->bi", J, z - q)                        # error predicted at p = z
        H = np.zeros((B, self.nv, self.nv))
        f = np.zeros((B, self.nv))
        for i in range(N):
            # r_i = W J v_i - k W e_i,  e_0 = e0,  e_i = c - dt J L_{i-1} u  (i >= 1)
            Ai = np.einsum("bij,jk->bik", WJ, self.S[i])
            if i == 0:
                bi = k * W * e0
            else:
                Ai = Ai + k * dt * np.einsum("bij,jk->bik", WJ, self.L[i - 1])
                bi = k * W * c
            H += np.einsum("bki,bkj->bij", Ai, Ai)
            f -= np.einsum("bki,bk->bi", Ai, bi)
        AT = dt * np.einsum("bij,jk->bik", WJ, self.L[N - 1])             # terminal: W e_N = W c - AT u
        H += self.terminal * np.einsum("bki,bkj->bij", AT, AT)
        f -= self.terminal * np.einsum("bki,bk->bi", AT, W * c)
        H += self.lam ** 2 * np.eye(self.nv)[None] + self.smooth * (self.D.T @ self.D)[None]
        f[:, :n] -= self.smooth * v_prev                                  # -(D^T d) with d = [v_prev, 0, ...]
        # bounds
        zlo = np.maximum(lo[None] - z, -self.trust)                       # (B,7) on z_i - z
        zhi = np.minimum(hi[None] - z, self.trust)
        zlo = np.minimum(zlo, 0.0)                                        # z itself stays admissible
        zhi = np.maximum(zhi, 0.0)
        ulo = np.full((B, self.nv), -1e30)
        uhi = np.full((B, self.nv), 1e30)
        lead_lo = (q - self.lead - z) / dt
        lead_hi = (q + self.lead - z) / dt
        ulo[:, :n] = np.minimum(lead_lo, lead_hi)
        uhi[:, :n] = np.maximum(lead_hi, lead_lo)
        rlo = np.concatenate([np.tile(zlo, N), np.full((B, N * n), -self.acc * dt)], axis=1)
        rhi = np.concatenate([np.tile(zhi, N), np.full((B, N * n), self.acc * dt)], axis=1)
        rlo[:, N * n:N * n + n] += v_prev
        rhi[:, N * n:N * n + n] += v_prev
        v0 = np.zeros((B, n))
        for b in range(B):
            bu = np.concatenate([uhi[b], rhi[b]])
            bl = np.concatenate([ulo[b], rlo[b]])
            x, _, flag, _ = daqp.solve(H[b], f[b], self.A, bu, bl, self.sense.copy(),
                                       primal_tol=1e-11, dual_tol=1e-12, iter_limit=20000)
            self.n_solves += 1
            if flag < 1:
                self.n_fallback += 1
                v0[b] = 0.0
            else:
                v0[b] = x[:n]
        return z + dt * v0, v0
