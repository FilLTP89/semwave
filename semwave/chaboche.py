"""Lemaitre--Chaboche visco-plasticity for the surficial layer of the SEM model.

J2 (von Mises) visco-plasticity with Armstrong--Frederick kinematic hardening
and Voce isotropic hardening, written for plane strain and stored in the Voigt
order (xx, yy, zz, xy); the out-of-plane component is carried because
eps_zz = 0 does not make sig_zz vanish, and the deviator needs it.

    f      = J(s - X) - (sigma_y + R),        J(A) = sqrt(3/2 A:A)
    dot p  = <f / K>_+^n                       (Norton / Lemaitre viscous flow)
    dot ep = 3/2 dot p (s - X)/J(s - X)
    dot X  = 2/3 C dot ep - gamma X dot p      (Armstrong--Frederick)
    dot R  = b (Q - R) dot p                   (Voce)

The flow rule is integrated **implicitly**: the increment dp solves the scalar
consistency equation obtained by radial return,

    Phi(dp) = J_tr - 3 mu dp - dp (C - gamma X_eq)/(1 + gamma dp)
              - sigma_y - (R + b Q dp)/(1 + b dp) - K (dp/dt)^(1/n) = 0,

with X_eq the back-stress projected on the trial flow direction.  Phi is
strictly decreasing, Phi(0) = f_tr > 0 on the yielding points, and
Phi(f_tr/(3 mu)) <= 0, so the root is bracketed a priori and bisection cannot
fail; it is run vectorised over the yielding points.  This is unconditionally
stable, which matters here: an explicit update needs a cap on dp that, for a
yield stress low enough to mobilise the layer, ends up governing the answer
instead of the rheology.

This is an illustrative rheology, not a calibrated soil model: the parameters
are fixed relative to the yield stress, and the yield stress itself is set as
a fraction of the peak elastic stress reached by the same event, so that the
surficial layer yields without collapsing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def j2(a):
    """J(A) = sqrt(3/2 A:A) for a symmetric tensor in (xx, yy, zz, xy)."""
    return np.sqrt(1.5 * (a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2
                          + 2.0 * a[:, 3] ** 2))


def elastic_increment(deps, lam, mu):
    """Plane-strain Hooke increment C:deps in Voigt order, with deps_zz = 0."""
    tr = deps[:, 0] + deps[:, 1]
    out = np.empty_like(deps)
    out[:, 0] = lam * tr + 2.0 * mu * deps[:, 0]
    out[:, 1] = lam * tr + 2.0 * mu * deps[:, 1]
    out[:, 2] = lam * tr
    out[:, 3] = 2.0 * mu * deps[:, 3]
    return out


@dataclass
class Chaboche:
    """Parameters, each either a scalar or one value per quadrature point."""

    sigma_y: np.ndarray | float
    C_kin: np.ndarray | float          # kinematic hardening modulus
    gamma: float = 100.0               # its recovery rate; X saturates at C/gamma
    Q_iso: np.ndarray | float = 0.0    # isotropic hardening amplitude
    b_iso: float = 150.0               # its rate
    K_visc: np.ndarray | float = 1.0   # viscous drag stress
    n_visc: float = 5.0                # Norton exponent
    dp_max: float = 0.0                # unused: the implicit update needs no cap

    def update(self, sig, X, R, p, deps, lam, mu, dt, active, n_bisect=60):
        """Advance the state in place by one step and return the elastic
        stress increment (which the caller needs for Rayleigh damping) and the
        number of points that yielded.

        sig, X, deps: (nq, 4); R, p, lam, mu: (nq,); active: boolean (nq,).
        """
        dsig_el = elastic_increment(deps, lam, mu)
        sig += dsig_el

        # trial deviator and effective stress
        hyd = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        s_tr = sig.copy()
        s_tr[:, :3] -= hyd[:, None]
        xi = s_tr - X
        J = j2(xi)

        sy = np.broadcast_to(np.asarray(self.sigma_y, dtype=float), J.shape)
        f_tr = J - (sy + R)
        hit = active & (f_tr > 0.0) & (J > 0.0)
        n_yield = int(np.count_nonzero(hit))
        if n_yield == 0:
            return dsig_el, 0

        # restrict to the yielding points
        mu_h = np.asarray(mu, dtype=float)[hit]
        C_h = np.broadcast_to(np.asarray(self.C_kin, dtype=float), J.shape)[hit]
        Q_h = np.broadcast_to(np.asarray(self.Q_iso, dtype=float), J.shape)[hit]
        K_h = np.broadcast_to(np.asarray(self.K_visc, dtype=float), J.shape)[hit]
        sy_h, R_h, J_h, f_h = sy[hit], R[hit], J[hit], f_tr[hit]

        # unit flow direction (Frobenius) and the back-stress projected on it
        nrm = np.sqrt(xi[hit, 0] ** 2 + xi[hit, 1] ** 2 + xi[hit, 2] ** 2
                      + 2.0 * xi[hit, 3] ** 2)
        N = xi[hit] / nrm[:, None]
        x_eq = np.sqrt(1.5) * (X[hit, 0] * N[:, 0] + X[hit, 1] * N[:, 1]
                               + X[hit, 2] * N[:, 2] + 2.0 * X[hit, 3] * N[:, 3])

        def phi(dp):
            kin = dp * (C_h - self.gamma * x_eq) / (1.0 + self.gamma * dp)
            iso = (R_h + self.b_iso * Q_h * dp) / (1.0 + self.b_iso * dp)
            visc = K_h * np.power(np.maximum(dp, 0.0) / dt, 1.0 / self.n_visc)
            return J_h - 3.0 * mu_h * dp - kin - sy_h - iso - visc

        # Phi(0) = f_tr > 0 and Phi(f_tr/3mu) <= 0: the root is bracketed
        lo = np.zeros(n_yield)
        hi = f_h / (3.0 * mu_h)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            neg = phi(mid) < 0.0
            hi = np.where(neg, mid, hi)
            lo = np.where(neg, lo, mid)
        dp = 0.5 * (lo + hi)

        # state update on the yielding points
        deps_p = (np.sqrt(1.5) * dp)[:, None] * N
        sig[hit] -= 2.0 * mu_h[:, None] * deps_p
        X[hit] = (X[hit] + (2.0 / 3.0) * C_h[:, None] * deps_p) \
            / (1.0 + self.gamma * dp)[:, None]
        R[hit] = (R_h + self.b_iso * Q_h * dp) / (1.0 + self.b_iso * dp)
        p[hit] += dp
        return dsig_el, n_yield
