#####
##### Time discretization types for CompressibleDynamics
#####
##### These types determine how the compressible equations are time-stepped:
##### - SplitExplicitTimeDiscretization: Acoustic substepping (Wicker-Skamarock scheme)
##### - ExplicitTimeStepping: Standard explicit time-stepping (small Δt required)
#####

"""
$(TYPEDEF)

Split-explicit time discretization for compressible dynamics.

Uses acoustic substepping following [Wicker and Skamarock (2002)](@cite WickerSkamarock2002):
- Outer loop: SSP RK3 for slow tendencies (advection, Coriolis, diffusion)
- Inner loop: Acoustic substeps for fast tendencies (pressure gradient, buoyancy)

This allows using advective CFL time steps (~10-20 m/s) instead of acoustic CFL
time steps (~340 m/s), typically enabling ~6x larger time steps.

The first positional argument controls the vertical stepping strategy:
- `nothing` (default): explicit vertical step (subject to vertical acoustic CFL)
- [`VerticallyImplicit`](@ref)`(α)`: implicit vertical solve with off-centering `α`

Fields
======

- `time_discretization`: `nothing` or [`VerticallyImplicit`](@ref)
- `substeps`: Number of acoustic substeps per full time step. Default: 6
- `divergence_damping_coefficient`: Divergence damping coefficient. Default: 0.05. When using base-state pressure correction (`reference_potential_temperature` in `CompressibleDynamics`), the stability constraint `(1-κᵈ)^Ns < 0.1` must be satisfied (e.g., κᵈ=0.2 for Ns=12, κᵈ=0.1 for Ns=24)

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{VTD, N, FT}
    time_discretization :: VTD
    substeps :: N
    divergence_damping_coefficient :: FT
end

function SplitExplicitTimeDiscretization(time_discretization=nothing; substeps=6, divergence_damping_coefficient=0.05)
    return SplitExplicitTimeDiscretization(time_discretization, substeps, divergence_damping_coefficient)
end

"""
$(TYPEDEF)

Standard explicit time discretization for compressible dynamics.

All tendencies (including pressure gradient and acoustic modes) are computed
together and time-stepped explicitly. This requires small time steps limited
by the acoustic CFL condition (sound speed ~340 m/s).

Use [`SplitExplicitTimeDiscretization`](@ref) for more efficient time-stepping with larger Δt.
"""
struct ExplicitTimeStepping end

#####
##### Vertical time discretization for acoustic substeps
#####

"""
$(TYPEDEF)

Vertically implicit time discretization for acoustic substeps.

Treats the vertical coupling between ``w`` and ``ρ`` (via the vertical
pressure gradient and buoyancy) implicitly using a tridiagonal solver.
This removes the vertical CFL restriction on the acoustic substep size,
following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007)
and CM1's `sound.F`.

The off-centering parameter ``α`` controls the time discretization:
- 0.5: Crank-Nicolson (second-order, no acoustic damping)
- Greater than 0.5: Forward-weighted (damps vertically propagating acoustic modes)

Fields
======

- `implicit_weight`: Off-centering parameter (0.5 for Crank-Nicolson, typically 0.5-0.55)
"""
struct VerticallyImplicit{FT}
    implicit_weight :: FT
end
