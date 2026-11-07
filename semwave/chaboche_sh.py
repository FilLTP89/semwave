"""Lemaitre--Chaboche visco-plasticity restricted to antiplane shear (SH).

Same rheology as chaboche.py -- J2 visco-plasticity with Armstrong--Frederick
kinematic hardening and Voce isotropic hardening, integrated by an implicit
radial return -- but written for the only stress state an SH wave can create:

    sigma = [[0, 0, s13],
             [0, 0, s23],
             [s13, s23, 0]]      carried as the pair tau = (s13, s23).

That state is already deviatoric, so the hydrostatic part never enters, and

    J(xi) = sqrt(3/2 xi:xi) = sqrt(3) |xi|,      |.| the Euclidean norm of the
                                                 pair,

which makes the consistency equation of chaboche.py hold verbatim: the radial
return still lowers J by exactly 3 mu dp, and the bracket [0, f_tr/(3 mu)] is
still valid, so the same bisection is used.  The yield stress is set the same
way as in plane strain, sigma_y = sqrt(3) mu gamma_y, so that the layer starts
to yield at the engineering shear strain gamma_y.

Run inside the course Docker image `fenicsx-jupyter` (Notebooks/Dockerfile).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def j2_sh(tau):
    """J(sigma) = sqrt(3/2 sigma:sigma) for an antiplane state (s13, s23)."""
    return np.sqrt(3.0) * np.sqrt(tau[:, 0] ** 2 + tau[:, 1] ** 2)


@dataclass
class ChabocheSH:
    """Parameters, each either a scalar or one value per quadrature point."""

    sigma_y: np.ndarray | float
    C_kin: np.ndarray | float          # kinematic hardening modulus
    gamma: float = 100.0               # its recovery rate; X saturates at C/gamma
    Q_iso: np.ndarray | float = 0.0    # isotropic hardening amplitude
    b_iso: float = 150.0               # its rate
    K_visc: np.ndarray | float = 1.0   # viscous drag stress
    n_visc: float = 5.0                # Norton exponent
    dp_max: float = 0.0                # unused: the implicit update needs no cap

    def update(self, sig, X, R, p, deps, mu, dt, active, n_bisect=60):
        """Advance the state in place by one step and return the elastic
        stress increment (needed for the Rayleigh damping) and the number of
        points that yielded.

        sig, X, deps: (nq, 2) antiplane pairs (13, 23); R, p, mu: (nq,);
        `deps` is the *tensor* strain increment (de13, de23), i.e. half the
        engineering shear.
        """
        dsig_el = 2.0 * np.asarray(mu, dtype=float)[:, None] * deps
        sig += dsig_el

        xi = sig - X                      # already deviatoric
        nrm2 = np.sqrt(xi[:, 0] ** 2 + xi[:, 1] ** 2)
        J = np.sqrt(3.0) * nrm2

        sy = np.broadcast_to(np.asarray(self.sigma_y, dtype=float), J.shape)
        f_tr = J - (sy + R)
        hit = active & (f_tr > 0.0) & (J > 0.0)
        n_yield = int(np.count_nonzero(hit))
        if n_yield == 0:
            return dsig_el, 0

        mu_h = np.asarray(mu, dtype=float)[hit]
        C_h = np.broadcast_to(np.asarray(self.C_kin, dtype=float), J.shape)[hit]
        Q_h = np.broadcast_to(np.asarray(self.Q_iso, dtype=float), J.shape)[hit]
        K_h = np.broadcast_to(np.asarray(self.K_visc, dtype=float), J.shape)[hit]
        sy_h, R_h, J_h, f_h = sy[hit], R[hit], J[hit], f_tr[hit]

        # unit flow direction (Frobenius: ||xi||_F = sqrt(2) |xi|) and the
        # back-stress projected on it, x_eq = sqrt(3/2) X:N
        N = xi[hit] / (np.sqrt(2.0) * nrm2[hit])[:, None]
        x_eq = np.sqrt(1.5) * 2.0 * (X[hit, 0] * N[:, 0] + X[hit, 1] * N[:, 1])

        def phi(dp):
            kin = dp * (C_h - self.gamma * x_eq) / (1.0 + self.gamma * dp)
            iso = (R_h + self.b_iso * Q_h * dp) / (1.0 + self.b_iso * dp)
            visc = K_h * np.power(np.maximum(dp, 0.0) / dt, 1.0 / self.n_visc)
            return J_h - 3.0 * mu_h * dp - kin - sy_h - iso - visc

        lo = np.zeros(n_yield)
        hi = f_h / (3.0 * mu_h)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            neg = phi(mid) < 0.0
            hi = np.where(neg, mid, hi)
            lo = np.where(neg, lo, mid)
        dp = 0.5 * (lo + hi)

        deps_p = (np.sqrt(1.5) * dp)[:, None] * N
        sig[hit] -= 2.0 * mu_h[:, None] * deps_p
        X[hit] = (X[hit] + (2.0 / 3.0) * C_h[:, None] * deps_p) \
            / (1.0 + self.gamma * dp)[:, None]
        R[hit] = (R_h + self.b_iso * Q_h * dp) / (1.0 + self.b_iso * dp)
        p[hit] += dp
        return dsig_el, n_yield
