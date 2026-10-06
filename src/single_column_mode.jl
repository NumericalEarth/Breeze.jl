"""
    SingleColumnMode

Run an `AtmosphereModel` as a single vertical column, or as a *forest* of independent columns
advanced concurrently, on a grid with `topology = (Flat, Flat, Bounded)`.

On such a grid every horizontal finite-difference operator returns zero (via `Flat`-topology
dispatch in `Oceananigans.Operators`), so horizontal advection, diffusion, and pressure-gradient
terms vanish and no halo information is exchanged in the horizontal. When the horizontal dimensions
are given a size greater than one — e.g. with `Oceananigans.Grids.ColumnEnsembleSize`, which forces
the horizontal halos to zero — the grid holds a horizontally independent forest of columns that can
be stepped in a single kernel launch, with no coupling between them.

This module gathers the single-column-specific method extensions that would otherwise be scattered
across the dynamics, closure, and model modules. It mirrors `Oceananigans.HydrostaticFreeSurfaceModel`'s
single-column mode. Per-column *reference states* live with the rest of the reference-state machinery
in `Breeze.Thermodynamics`, since they build on the reference-state construction there.
"""
module SingleColumnMode

export SingleColumnGrid

using Oceananigans.Grids: AbstractGrid, Flat, Bounded
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, getclosure
using Oceananigans.Utils: with_tracers

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel
using Breeze.AnelasticEquations: AnelasticDynamics

"""
    const SingleColumnGrid

A grid with `topology = (Flat, Flat, Bounded)` — a single vertical column, or (when the horizontal
dimensions have size greater than one, e.g. via `ColumnEnsembleSize`) a forest of independent columns.
"""
const SingleColumnGrid = AbstractGrid{<:Any, <:Flat, <:Flat, <:Bounded}

# Anelastic dynamics on a single-column grid, possibly an ensemble of columns. The grid is the 5th
# `AtmosphereModel` type parameter (`Dyn, Frm, Arc, Tst, Grd, …`).
const AnelasticSingleColumnModel = AtmosphereModel{<:AnelasticDynamics, <:Any, <:Any, <:Any, <:SingleColumnGrid}

#####
##### No pressure solve, no vertical-velocity stepping (w ≡ 0)
#####
#
# On a `SingleColumnGrid` the anelastic mass constraint ∂z(ρᵣ w) = 0 with rigid top/bottom boundaries
# (w = 0 there) forces `w ≡ 0` throughout the column — there is no elliptic problem left to solve. So
# we build no pressure solver, skip the pressure correction, and hold the vertical-momentum tendency
# at zero (`ρw`, initialized to 0, never moves). Vertical transport is carried by the turbulence
# closure and prescribed large-scale forcing (e.g. subsidence).

AtmosphereModels.dynamics_pressure_solver(::AnelasticDynamics, ::SingleColumnGrid) = nothing

AtmosphereModels.compute_pressure_correction!(::AnelasticSingleColumnModel, Δt) = nothing
AtmosphereModels.make_pressure_correction!(::AnelasticSingleColumnModel, Δt) = nothing

# `compute_z_momentum_tendency!` overwrites `Gρw`, so zeroing it here — rather than launching it —
# is what omits vertical-velocity stepping.
function AtmosphereModels.compute_vertical_momentum_tendency!(::AnelasticSingleColumnModel, Gρw, w_args)
    fill!(parent(Gρw), 0)
    return nothing
end

#####
##### Per-column closures
#####
#
# The turbulence closure may be an *array* of closures — one per column — so that an ensemble of
# independent columns each use different mixing parameters (e.g. for closure calibration). Each
# flux-divergence kernel selects its column's closure with `getclosure` (a matrix is indexed `[i, j]`,
# a vector `[i]`) and forwards to the single-closure method. Breeze's flux signatures carry the
# reference density `ρ` between `grid` and `closure`, so the array sits at that same slot.

const ClosureArray = AbstractArray{<:AbstractTurbulenceClosure}

@inline AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid::SingleColumnGrid, ρ, closures::ClosureArray, args...) =
    AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, getclosure(i, j, closures), args...)

@inline AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid::SingleColumnGrid, ρ, closures::ClosureArray, args...) =
    AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, getclosure(i, j, closures), args...)

# Included for completeness; the anelastic single-column mode never launches the z-momentum tendency
# (`w ≡ 0`), so this is dormant there.
@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid::SingleColumnGrid, ρ, closures::ClosureArray, args...) =
    AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, getclosure(i, j, closures), args...)

@inline AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid::SingleColumnGrid, ρ, closures::ClosureArray, args...) =
    AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, getclosure(i, j, closures), args...)

# A per-column array of closures/rotations must live on the grid architecture so it can be indexed
# inside GPU kernels. `with_tracers` has no method for a bare array of closures, so we map it over the
# array here (mirroring Oceananigans' per-`ISSDVector` methods). See `materialize_closure`/
# `materialize_coriolis` in `AtmosphereModels`, which pass scalars through unchanged.
AtmosphereModels.materialize_closure(closures::ClosureArray, scalar_names, arch) =
    on_architecture(arch, map(closure -> with_tracers(scalar_names, closure), closures))

AtmosphereModels.materialize_coriolis(coriolis::AbstractArray, arch) = on_architecture(arch, coriolis)

end # module SingleColumnMode
