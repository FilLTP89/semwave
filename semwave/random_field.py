"""Shinozuka--Deodatis spectral representation of a 2-D random field, used
here to draw correlated Lame parameters for the SEM wave simulations.

The field is the classical double cosine sum (Shinozuka & Deodatis, 1991,
1996)

    f(x1,x2) = sqrt(2) sum_{n1,n2} sqrt(2 S(k1,k2) dk1 dk2)
               [ cos(k1 x1 + k2 x2 + phi1) + cos(k1 x1 - k2 x2 + phi2) ]

with independent phases phi ~ U[0,2pi), evaluated on a regular grid.  The
target spectral density is the 2-D von Karman (Hurst exponent nu; nu = 1/2 is
the exponential / Markov model used for crustal velocity fluctuations):

    S(k1,k2) = sigma^2 b1 b2 Gamma(nu+1)
               / ( pi Gamma(nu) (1 + b1^2 k1^2 + b2^2 k2^2)^{nu+1} ).

Run inside the course Docker image `fenicsx-jupyter` (Notebooks/Dockerfile).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn


def von_karman_psd(k1, k2, sigma, b1, b2, nu=0.5):
    """2-D von Karman power spectral density (zero-mean, variance sigma^2)."""
    pref = sigma**2 * b1 * b2 * gamma_fn(nu + 1.0) / (np.pi * gamma_fn(nu))
    return pref / (1.0 + (b1 * k1) ** 2 + (b2 * k2) ** 2) ** (nu + 1.0)


def shinozuka_deodatis_2d(x1, x2, sigma=0.2, b1=250.0, b2=100.0, nu=0.5,
                          n_wave=(48, 48), k_cut=None, rng=None):
    """Draw one realisation of a zero-mean von Karman field on the grid
    (x1, x2) given as 1-D coordinate vectors.  Returns an (n2, n1) array,
    i.e. ready for `imshow` with `origin="lower"`.

    n_wave  number of wavenumbers (N1, N2) retained per direction
    k_cut   (k1_max, k2_max); defaults to 12 / correlation length
    """
    rng = np.random.default_rng() if rng is None else rng
    n1, n2 = n_wave
    k1_max, k2_max = k_cut if k_cut is not None else (12.0 / b1, 12.0 / b2)
    dk1, dk2 = k1_max / n1, k2_max / n2
    # mid-point wavenumbers: avoids the k = 0 term, which carries no energy
    k1 = (np.arange(n1) + 0.5) * dk1
    k2 = (np.arange(n2) + 0.5) * dk2

    K1, K2 = np.meshgrid(k1, k2, indexing="ij")
    amp = np.sqrt(2.0 * von_karman_psd(K1, K2, sigma, b1, b2, nu) * dk1 * dk2)
    phi1 = rng.uniform(0.0, 2.0 * np.pi, size=(n1, n2))
    phi2 = rng.uniform(0.0, 2.0 * np.pi, size=(n1, n2))

    X1, X2 = np.meshgrid(x1, x2, indexing="xy")  # (n_x2, n_x1)
    field = np.zeros_like(X1, dtype=float)
    for i in range(n1):
        a1 = K1[i, 0] * X1
        for j in range(n2):
            a2 = K2[i, j] * X2
            field += amp[i, j] * (np.cos(a1 + a2 + phi1[i, j])
                                  + np.cos(a1 - a2 + phi2[i, j]))
    return np.sqrt(2.0) * field


# Layered-cake background: (thickness [m], Vs [m/s], b1 [m], b2 [m]) from the
# free surface downwards; the last entry is the half-space and its thickness is
# ignored.  Thicknesses are multiples of the element size, so that interfaces
# fall on element boundaries of the SEM mesh (H = 1200 m, 24 elements -> 50 m).
# The correlation lengths grow with depth: young, finely layered sediments at
# the top, smoother and longer-correlated units at depth.
DEFAULT_LAYERS = ((150.0, 600.0, 120.0, 40.0),
                  (250.0, 900.0, 250.0, 80.0),
                  (400.0, 1400.0, 500.0, 150.0),
                  (np.inf, 2000.0, 900.0, 250.0))


def layer_depths(layers=DEFAULT_LAYERS):
    """Top depth of each layer [m]."""
    tops, d = [], 0.0
    for entry in layers:
        tops.append(d)
        d += entry[0]
    return tops


def layered_profile(x2, H, layers=DEFAULT_LAYERS):
    """Background Vs(z) of a layered cake, as a function of the vertical
    coordinate x2 (x2 = H is the free surface, so depth = H - x2)."""
    depth = H - np.asarray(x2, dtype=float)
    vs0 = np.full_like(depth, layers[-1][1])
    top = 0.0
    for entry in layers[:-1]:
        thickness, value = entry[0], entry[1]
        vs0[(depth >= top) & (depth < top + thickness)] = value
        top += thickness
    return vs0


def layered_fluctuation(x1, x2, H, layers=DEFAULT_LAYERS, sigma=0.15, nu=0.5,
                        rng=None, **kw):
    """Zero-mean log-fluctuation field whose correlation lengths follow the
    layering: each unit carries its own von Karman texture (b1, b2), drawn from
    an independent stream, and the units are stitched at their interfaces.

    Layers must be given as 4-tuples (thickness, Vs, b1, b2); 2-tuples fall
    back to the correlation lengths passed in `kw` (b1, b2).
    """
    rng = np.random.default_rng() if rng is None else rng
    depth = H - np.asarray(x2, dtype=float)
    field = np.zeros((len(x2), len(x1)))
    top = 0.0
    for entry in layers:
        thickness = entry[0]
        if len(entry) >= 4:
            b1, b2 = entry[2], entry[3]
        else:
            b1, b2 = kw.get("b1", 250.0), kw.get("b2", 100.0)
        rows = (depth >= top) & (depth < top + thickness)
        if rows.any():
            f = shinozuka_deodatis_2d(x1, x2, sigma=sigma, b1=b1, b2=b2, nu=nu,
                                      rng=rng)
            field[rows, :] = f[rows, :]
        top += thickness
    return field


def lame_fields(x1, x2, vs0=1200.0, vp_vs=1.78, sigma_ln=0.15,
                b1=250.0, b2=100.0, nu=0.5, rng=None, layers=DEFAULT_LAYERS,
                **kw):
    """Lognormal Vs field on a layered-cake background, with depth-dependent
    correlation lengths, plus the Lame parameters the solver needs.

    Vs = Vs0(z) exp(f),  Vp = vp_vs Vs,  rho from Gardner's relation,
    mu = rho Vs^2,  lambda = rho (Vp^2 - 2 Vs^2).

    `layers=None` falls back to a homogeneous background `vs0` and a single
    von Karman texture (b1, b2).
    """
    if layers is None:
        f = shinozuka_deodatis_2d(x1, x2, sigma=sigma_ln, b1=b1, b2=b2, nu=nu,
                                  rng=rng, **kw)
        background = vs0
    else:
        H = float(np.max(x2))
        f = layered_fluctuation(x1, x2, H, layers=layers, sigma=sigma_ln,
                                nu=nu, rng=rng, b1=b1, b2=b2)
        background = layered_profile(x2, H, layers)[:, None]   # (n_x2, 1)
    vs = background * np.exp(f)
    vp = vp_vs * vs
    rho = 310.0 * vp ** 0.25            # Gardner, SI units
    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    return {"f": f, "vs": vs, "vp": vp, "rho": rho, "mu": mu, "lam": lam}
