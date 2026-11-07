"""2-D spectral-element elastodynamics on random media, with an optional
visco-plastic layer.

    from semwave import SEMWave2D, SEMWaveVP2D, SEMWaveSH2D, lame_fields

  * `random_field` -- Shinozuka--Deodatis von Karman fields, layered-cake
    background, Lame parameters and Gardner density;
  * `anisotropy`   -- transversely isotropic stiffness from Thomsen epsilon
    and delta, and the Christoffel phase velocities that check it;
  * `sem_wave`     -- the linear solver: GLL spectral elements, lumped mass,
    explicit central differences, Ricker source, Cerjan sponge, free surface,
    and an optional tilted TI fabric;
  * `chaboche`     -- Lemaitre--Chaboche J2 visco-plasticity, integrated
    implicitly by bracketed bisection on the consistency equation;
  * `sem_wave_vp`  -- the same solver with the stress carried at the GLL
    points, so a surficial layer can be visco-plastic;
  * `chaboche_sh` / `sem_wave_sh` -- the same rheology and the same solver
    restricted to antiplane (SH) motion, which is what a surface-to-borehole
    transfer function is read on.

Needs FEniCSx (dolfinx, basix, ufl, petsc4py, mpi4py) plus numpy and scipy;
`random_field` and `chaboche` need only numpy/scipy and can be used on their
own, without FEniCSx.
"""
from .anisotropy import check_stability, phase_velocities, thomsen_stiffness
from .chaboche import Chaboche, elastic_increment, j2
from .chaboche_sh import ChabocheSH, j2_sh
from .random_field import (DEFAULT_LAYERS, lame_fields, layer_depths,
                           layered_fluctuation, layered_profile,
                           shinozuka_deodatis_2d, von_karman_psd)

__all__ = [
    "check_stability", "phase_velocities", "thomsen_stiffness",
    "Chaboche", "elastic_increment", "j2",
    "ChabocheSH", "j2_sh",
    "DEFAULT_LAYERS", "lame_fields", "layer_depths", "layered_fluctuation",
    "layered_profile", "shinozuka_deodatis_2d", "von_karman_psd",
    "SEMWave2D", "SEMWaveVP2D", "SEMWaveSH2D", "ricker",
]

__version__ = "0.1.0"


def __getattr__(name):
    """Import the FEniCSx-dependent solvers lazily, so that the random-field
    and constitutive parts remain usable without dolfinx installed."""
    if name in ("SEMWave2D", "ricker"):
        from .sem_wave import SEMWave2D, ricker
        return {"SEMWave2D": SEMWave2D, "ricker": ricker}[name]
    if name == "SEMWaveVP2D":
        from .sem_wave_vp import SEMWaveVP2D
        return SEMWaveVP2D
    if name == "SEMWaveSH2D":
        from .sem_wave_sh import SEMWaveSH2D
        return SEMWaveSH2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
