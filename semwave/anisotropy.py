"""Transversely isotropic elasticity for the 2-D plane-strain solver.

The random medium of `random_field.py` is isotropic: one `lam`, one `mu` per
point.  Sediments are not.  Fine layering, aligned clay platelets and aligned
cracks all make the rock faster along the fabric than across it, and at the
wavelengths of `sem_wave.py` the stack is not resolved -- it is *felt* as an
effective transversely isotropic (TI) medium (Backus, 1962).

One symmetry axis, tilted by `theta` from the vertical, is enough to describe
it.  In the plane of the model the stiffness then has four constants, written
here in the fabric frame (1 = along the fabric, 3 = along the symmetry axis):

    sig_11 = C11 e_11 + C13 e_33,
    sig_33 = C13 e_11 + C33 e_33,
    sig_13 = 2 C55 e_13,

and they are parameterised the way seismologists measure them, by Thomsen
(1986) `epsilon` and `delta` on top of the isotropic part:

    C33 = lam + 2 mu                     (P along the symmetry axis)
    C55 = mu                             (S polarised in the plane)
    C11 = C33 (1 + 2 epsilon)            (P along the fabric)
    C13 = sqrt( 2 delta C33 (C33 - C55) + (C33 - C55)^2 ) - C55

`epsilon` is exactly the P-wave anisotropy, Vp(90 deg)/Vp(0 deg) =
sqrt(1 + 2 epsilon); `delta` controls the near-axis curvature of the
wavefront, i.e. how an-elliptic it is (`delta = epsilon` gives an ellipse).
Both vanish in the isotropic limit, where C11 = C33 = lam + 2 mu and
C13 = lam.

The formulae are written once, with `sqrt` injected, so that the same code
serves the numpy path (sampling, quality checks) and the UFL path (the form
the solver assembles).
"""
from __future__ import annotations

import numpy as np

__all__ = ["thomsen_stiffness", "phase_velocities", "check_stability"]


def thomsen_stiffness(lam, mu, epsilon=0.0, delta=0.0, sqrt=np.sqrt):
    """(C11, C13, C33, C55) of a TI medium, in its own fabric frame.

    `lam`, `mu`, `epsilon`, `delta` may be numbers, numpy arrays or UFL
    expressions; pass `sqrt=ufl.sqrt` for the latter.
    """
    c33 = lam + 2.0 * mu
    c55 = mu
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c13 = sqrt(2.0 * delta * c33 * (c33 - c55) + (c33 - c55) ** 2) - c55
    return c11, c13, c33, c55


def phase_velocities(c11, c13, c33, c55, rho, angle):
    """qP and qSV phase velocities at `angle` [rad] from the symmetry axis.

    The 2-D Christoffel matrix for a propagation direction
    n = (sin a, cos a) in the fabric frame is

        G11 = C11 n1^2 + C55 n3^2,   G33 = C55 n1^2 + C33 n3^2,
        G13 = (C13 + C55) n1 n3,

    and rho V^2 are its two eigenvalues.  Returns (v_qp, v_qsv).
    """
    n1, n3 = np.sin(angle), np.cos(angle)
    g11 = c11 * n1**2 + c55 * n3**2
    g33 = c55 * n1**2 + c33 * n3**2
    g13 = (c13 + c55) * n1 * n3
    half = 0.5 * (g11 + g33)
    disc = np.sqrt((0.5 * (g11 - g33)) ** 2 + g13**2)
    return np.sqrt((half + disc) / rho), np.sqrt(np.maximum(half - disc, 0.0) / rho)


def check_stability(c11, c13, c33, c55):
    """Strain energy positive definite: C11, C33, C55 > 0 and C11 C33 > C13^2.

    Returns the worst (smallest) margin of each condition; negative means the
    parameters are not an elastic medium, which `epsilon`/`delta` chosen
    independently of each other can easily produce.
    """
    c11, c13, c33, c55 = (np.asarray(x, dtype=float) for x in
                          (c11, c13, c33, c55))
    return {"C11": float(np.min(c11)), "C33": float(np.min(c33)),
            "C55": float(np.min(c55)),
            "C11*C33-C13^2": float(np.min(c11 * c33 - c13**2))}
