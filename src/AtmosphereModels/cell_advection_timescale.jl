#####
##### Direction-aware advective timescale for the time-step wizard
#####
##### The `TimeStepWizard` floats Δt at a target advective CFL by calling
##### `cell_advection_timescale(model)` — a `minimum` over the grid of
#####
#####   τ = 1 / (|u|/Δx + |v|/Δy + |w|/Δz).
#####
##### Adaptive implicit vertical advection (AIVA) removes the *vertical* advective CFL as a
##### stability constraint: its explicit vertical velocity is `wᵉ = w · min(1, cfl/α)`, so the
##### explicit vertical CFL is `min(α, cfl) ≤ cfl` regardless of Δt. When every vertically-advected
##### prognostic uses AIVA, the vertical term therefore imposes no restriction and should drop out
##### of the timescale — otherwise the wizard would still clamp Δt to a transient fast updraft and
##### AIVA would buy nothing at the run level. This mirrors Oceananigans' vertically-implicit
##### diffusion, whose `cell_diffusion_timescale` returns `Inf` for the same reason.
#####
##### `cell_advection_timescale(model::AtmosphereModel)` (the wizard's default) makes this choice
##### automatically from `model.advection`; `CellAdvectionTimescale(formulation)` is the explicit
##### override for forcing or monitoring a particular direction (see its docstring) — e.g. to watch
##### the true three-dimensional CFL even while the wizard floats Δt on the horizontal one.

using Oceananigans.Advection: Advection, cell_advection_timescale
using Oceananigans.BoundaryConditions: needs_implicit_solver
using Oceananigans.Fields: ZeroField
using Oceananigans.TurbulenceClosures: HorizontalFormulation, ThreeDimensionalFormulation
using Oceananigans.Utils: sum_of_velocities

"""
$(TYPEDSIGNATURES)

A callable that returns the resolved-flow advective timescale of a `model` restricted to the
directions of `formulation`: `HorizontalFormulation()` counts only the horizontal advective CFL
(dropping the vertical term), while `ThreeDimensionalFormulation()` counts all three directions.
Pass it to the `cell_advection_timescale` keyword of `TimeStepWizard` /
`conjure_time_step_wizard!`, or as the
`timescale` argument of `CFL` (`CFL(Δt, CellAdvectionTimescale(...))`), to control or monitor
which resolved-flow directions bind the time step. The automatic `cell_advection_timescale(model)`
also includes field-specific microphysical velocities.
"""
struct CellAdvectionTimescale{F}
    formulation :: F
end

(τ::CellAdvectionTimescale)(model) = cell_advection_timescale(model, τ.formulation)

# The vertical advecting velocity is Cartesian `w` on height-coordinate grids and the contravariant
# `w̃` on terrain-following grids (see `advecting_vertical_velocity`). A `ZeroField` in the vertical
# slot makes Oceananigans' own kernel compute the horizontal-only timescale — same reduction, same
# topology/Flat handling, with the `|w|/Δz` term identically zero.
function Advection.cell_advection_timescale(model::AtmosphereModel, ::ThreeDimensionalFormulation)
    u, v, _ = model.velocities
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
    return cell_advection_timescale(model.grid, (u, v, w))
end

function Advection.cell_advection_timescale(model::AtmosphereModel, ::HorizontalFormulation)
    u, v, _ = model.velocities
    return cell_advection_timescale(model.grid, (u, v, ZeroField()))
end

# Automatic default: drop the vertical term exactly when every vertically-advected prognostic uses
# AIVA (they share the advecting `w`, so a single explicit prognostic re-imposes the vertical CFL).
function Advection.cell_advection_timescale(model::AtmosphereModel)
    resolved_timescale = if all_vertical_advection_is_implicit(model.advection)
        cell_advection_timescale(model, HorizontalFormulation())
    else
        cell_advection_timescale(model, ThreeDimensionalFormulation())
    end

    names = prognostic_field_names(model.microphysics)
    return minimum_microphysical_advection_timescale(model, names, resolved_timescale)
end

# Microphysical terminal velocities are field-specific, so they do not appear in
# `transport_velocities(model)`. Include each prognostic's full transport velocity in the
# default timescale; otherwise fast sedimentation can violate an explicit scalar scheme's CFL
# while the time-step wizard sees only the resolved flow.
@inline minimum_microphysical_advection_timescale(model, ::Tuple{}, timescale) = timescale

@inline function minimum_microphysical_advection_timescale(model, names::Tuple{Symbol, Vararg}, timescale)
    name = first(names)
    microphysical_velocity = microphysical_velocities(model.microphysics,
                                                      model.microphysical_fields,
                                                      Val(name))
    field_timescale = microphysical_advection_timescale(model, microphysical_velocity, timescale)
    return minimum_microphysical_advection_timescale(model, Base.tail(names), field_timescale)
end

@inline microphysical_advection_timescale(model, ::Nothing, timescale) = timescale

@inline function microphysical_advection_timescale(model, microphysical_velocity, timescale)
    transport_velocity = sum_of_velocities(transport_velocities(model), microphysical_velocity)
    return min(timescale, cell_advection_timescale(model.grid, transport_velocity))
end

# `nothing` schemes advect nothing (no vertical CFL); every other scheme must be AIVA.
# Note: `all` follows the three-valued logic and _may_ return `missing` in some cases.  Let's
# inform the compiler with the `::Bool` annotation that we know we only deal with booleans.
all_vertical_advection_is_implicit(advection::NamedTuple)::Bool =
    all(scheme -> scheme === nothing || needs_implicit_solver(scheme), values(advection))::Bool
