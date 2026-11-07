#!/usr/bin/env python3
"""Theme-1 divider image: 2-D heterogeneous domain, extended fault, wave
snapshot, and the spectral-element mesh drawn over it.

Antiplane (SH) elastodynamics, rho d_tt u = div(mu grad u), on a layered
random medium (Shinozuka-Deodatis von Karman on a layered background).
The source is an extended, dipping fault ruptured from its bottom tip at
0.8 Vs, so the snapshot shows directivity rather than a point-source circle.
GLL spectral elements, degree 4, lumped mass, explicit central differences.
"""
from pathlib import Path

import numpy as np, basix, ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import RegularGridInterpolator
from semwave import lame_fields

L, H = 4000.0, 2500.0          # width, height; y = H is the free surface
NX, NY, DEG = 16, 10, 4        # 250 m elements: coarse enough that the GLL grid reads
F0, CFL, SPONGE = 0.75, 0.28, 320.0   # f_max ~1.9 Hz: ~4 GLL nodes per min. wavelength
T_SNAP = 1.70
# fault: a 60-degree dipping segment, ruptured upward from its lower tip
FAULT_A = np.array([1400.0,  600.0])   # lower tip (hypocentre)
FAULT_B = np.array([1900.0, 1500.0])   # upper tip
V_RUP_FRAC = 0.8


def ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


comm = MPI.COMM_WORLD
msh = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([L, H])],
                            [NX, NY], mesh.CellType.quadrilateral)
el = basix.ufl.element("Lagrange", "quadrilateral", DEG,
                       lagrange_variant=basix.LagrangeVariant.gll_isaac)
V = fem.functionspace(msh, el)
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_rule": "GLL",
                                             "quadrature_degree": 2 * DEG - 1})

x1 = np.linspace(0.0, L, 201); x2 = np.linspace(0.0, H, 141)
flds = lame_fields(x1, x2, rng=np.random.default_rng(7), sigma_ln=0.30,
                   b1=320.0, b2=110.0)
mu = fem.Function(V); rho = fem.Function(V)
for key, fn in (("mu", mu), ("rho", rho)):
    ip = RegularGridInterpolator((x2, x1), flds[key], bounds_error=False,
                                 fill_value=None)
    fn.interpolate(lambda x, ip=ip: ip(np.column_stack(
        (np.clip(x[1], 0, H), np.clip(x[0], 0, L)))))

v = ufl.TestFunction(V); u = fem.Function(V)
f_int = fem.form(mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
m_vec = assemble_vector(fem.form(rho * v * dx))
m_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
m = m_vec.array_r.copy()
xd = V.tabulate_dof_coordinates()[:, :2]

# nodes within 60 m of the fault segment, with a rupture-time delay
d = FAULT_B - FAULT_A; Lf = np.linalg.norm(d); t_hat = d / Lf
w = xd - FAULT_A
proj = np.clip(w @ t_hat, 0.0, Lf)
dist = np.linalg.norm(w - np.outer(proj, t_hat), axis=1)
fault = np.where(dist < 60.0)[0]
vs_f = float(np.sqrt(flds["mu"].mean() / flds["rho"].mean()))
delay = proj[fault] / (V_RUP_FRAC * vs_f)
print(f"fault nodes {fault.size}, length {Lf:.0f} m, v_rup {V_RUP_FRAC*vs_f:.0f} m/s")

dsp = np.minimum.reduce([xd[:, 0], L - xd[:, 0], xd[:, 1]])
taper = np.ones_like(dsp); ins = dsp < SPONGE
taper[ins] = np.exp(-(0.3 * (SPONGE - dsp[ins]) / SPONGE) ** 2)

dt = CFL * (min(L / NX, H / NY) / DEG) / float(flds["vs"].max())
nst = int(np.ceil(T_SNAP / dt))
print(f"dt {dt:.2e}, steps {nst}")
u_old = np.zeros(u.x.array.size); u_now = np.zeros_like(u_old)
beta, alpha = 1.0 / (150.0 * 2 * np.pi * F0), 0.5
t0 = 1.2 / F0
for n in range(nst):
    t = n * dt
    u.x.array[:] = u_now + (beta / dt) * (u_now - u_old)
    r = assemble_vector(f_int)
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    force = -r.array_r.copy()
    force[fault] += 9.0e3 * ricker(t - delay, F0, t0) * m[fault]
    c_lo = 0.5 * alpha * dt
    u_new = (2 * u_now - (1 - c_lo) * u_old + dt**2 * force / m) / (1 + c_lo)
    u_new *= taper
    u_old, u_now = u_now, u_new
snap = (u_now - u_old) / (2 * dt)

out = Path(__file__).resolve().with_name("fault_snap.npz")
np.savez_compressed(out, xdofs=xd, snap=snap,
                    x1=x1, x2=x2, vs=flds["vs"],
                    L=L, H=H, NX=NX, NY=NY, DEG=DEG,
                    fault_a=FAULT_A, fault_b=FAULT_B, sponge=SPONGE)
print("saved", out)
