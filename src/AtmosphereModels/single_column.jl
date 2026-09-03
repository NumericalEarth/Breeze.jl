using Oceananigans.Grids: AbstractGrid, Flat, Bounded

"""
    const SingleColumnGrid

A grid with `topology = (Flat, Flat, Bounded)` — a single vertical column.

On such a grid every horizontal finite-difference operator returns zero (via `Flat`-topology
dispatch in `Oceananigans.Operators`), so horizontal advection, diffusion, and pressure-gradient
terms vanish and no halo information is exchanged in the horizontal. When the horizontal dimensions
are given size greater than one — e.g. via `Oceananigans.Grids.ColumnEnsembleSize`, which forces the
horizontal halos to zero — the grid holds a horizontally independent "forest" of columns that can be
advanced concurrently in a single kernel launch, with no coupling between columns.

`AtmosphereModel`s built on a `SingleColumnGrid` run in "single column mode": the anelastic
dynamics omit the pressure solve and the vertical-velocity stepping (`w ≡ 0`), so vertical transport
comes from the turbulence closure and prescribed large-scale forcing (e.g. subsidence). This mirrors
`Oceananigans.HydrostaticFreeSurfaceModel`'s single-column / column-ensemble mode.
"""
const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}
