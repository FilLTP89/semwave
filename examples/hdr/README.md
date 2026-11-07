# Reproducing the HDR figures

This directory contains the simulation and plotting workflows behind the
SEM-based figures in the [HDR gallery](../../docs/hdr-gallery.md).  Generated
`.npz` caches are deliberately ignored: they can be large, whereas the final
figures and every parameter needed to regenerate them are versioned.

## Environment

Install this repository in editable mode and add Matplotlib:

```bash
python -m pip install -e ".[plots]"
```

The simulation runners also require FEniCSx (`dolfinx`, `basix`, `ufl`,
`petsc4py`, and `mpi4py`).  The plotting scripts use LaTeX text rendering, so
a LaTeX installation with `amsmath` and `amssymb` is needed for exact output.

## Regeneration order

Run these commands from the repository root:

```bash
# Textured polycrystal: simulation, fabric, and ultrasonic coda
(cd examples/hdr/aniso_ndt_data && python run_aniso_ndt.py)
python examples/hdr/make_aniso_fabric.py
python examples/hdr/make_aniso_coda.py

# Heterogeneous ballistic front and scattered coda
(cd examples/hdr/ballistic_coda_data && python run_ballistic_coda.py)
python examples/hdr/make_ballistic_coda.py

# Neumann-series decomposition
(cd examples/hdr/coda_neumann_data && python run_neumann_orders.py)
python examples/hdr/make_coda_neumann.py

# Nonlinear SH soil column
python examples/hdr/soil_nonlinear_data/run_soil_sh_tf.py
python examples/hdr/make_soil_nonlinear.py

# Extended-fault divider graphic
python examples/hdr/make_theme1_fault_sim.py
python examples/hdr/make_theme1_fault_plot.py

# Analytical cost map; no simulation cache is needed
python examples/hdr/make_cost_map.py
```

All plotting scripts resolve paths relative to this checkout and overwrite the
matching final image under `docs/figures/hdr/`.
