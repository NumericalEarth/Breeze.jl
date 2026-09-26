#####
##### Direction-aware advective timescale for the time-step wizard
#####
##### The `TimeStepWizard` floats Δt at a target advective CFL by calling
##### `cell_advection_timescale(model)` — a `minimum` over the grid of
#####
#####   τ = 1 / (|u|/Δx + |v|/Δy + |w|/Δz).
#####
##### The minimum also includes each sedimenting prognostic's air + fall velocity,
##### matching the transport velocity used by its advective flux.
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

A callable that returns the advective timescale of a `model` restricted to the directions of
`formulation`: `HorizontalFormulation()` counts only the horizontal advective CFL (dropping the
vertical term), `ThreeDimensionalFormulation()` counts all three directions. Pass it to the
`cell_advection_timescale` keyword of `TimeStepWizard` / `conjure_time_step_wizard!`, or as the
`timescale` argument of `CFL` (`CFL(Δt, CellAdvectionTimescale(...))`), to control or monitor
which directions bind the time step.

The timescale includes each microphysical prognostic's transport velocity (air velocity
plus sedimentation velocity), so falling precipitation also constrains explicit advection.
"""
struct CellAdvectionTimescale{F}
    formulation :: F
end

(τ::CellAdvectionTimescale)(model) = cell_advection_timescale(model, τ.formulation)

# The vertical advecting velocity is Cartesian `w` on height-coordinate grids and the contravariant
# `w̃` on terrain-following grids (see `advecting_vertical_velocity`). A `ZeroField` in the vertical
# slot makes Oceananigans' own kernel compute the horizontal-only timescale — same reduction, same
# topology/Flat handling, with the `|w|/Δz` term identically zero.
function Advection.cell_advection_timescale(model::AtmosphereModel,
                                          formulation::Union{ThreeDimensionalFormulation, HorizontalFormulation})
    u, v, _ = model.velocities
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
    velocities = (; u, v, w)
    timescale = directional_advection_timescale(model.grid, velocities, formulation)

    # Use the same air + sedimentation velocity as the prognostic's advective flux.
    # The air velocity alone can vanish while precipitation still crosses cells.
    for name in prognostic_field_names(model.microphysics)
        microphysical_velocity = microphysical_velocities(model.microphysics, model.microphysical_fields, Val(name))
        if microphysical_velocity !== nothing && model.advection[name] !== nothing
            transport_velocity = sum_of_velocities(velocities, microphysical_velocity)
            timescale = min(timescale, directional_advection_timescale(model.grid, transport_velocity, formulation))
        end
    end

    return timescale
end

directional_advection_timescale(grid, velocities, ::ThreeDimensionalFormulation) =
    cell_advection_timescale(grid, velocities)

directional_advection_timescale(grid, velocities, ::HorizontalFormulation) =
    cell_advection_timescale(grid, (velocities.u, velocities.v, ZeroField()))

# Automatic default: drop the vertical term exactly when every vertically-advected prognostic uses
# AIVA. A single explicit prognostic retains the conservative three-dimensional CFL constraint.
function Advection.cell_advection_timescale(model::AtmosphereModel)
    if all_vertical_advection_is_implicit(model.advection)
        return cell_advection_timescale(model, HorizontalFormulation())
    else
        return cell_advection_timescale(model, ThreeDimensionalFormulation())
    end
end

# `nothing` schemes advect nothing (no vertical CFL); every other scheme must be AIVA.
# Note: `all` follows the three-valued logic and _may_ return `missing` in some cases.  Let's
# inform the compiler with the `::Bool` annotation that we know we only deal with booleans.
all_vertical_advection_is_implicit(advection::NamedTuple)::Bool =
    all(scheme -> scheme === nothing || needs_implicit_solver(scheme), values(advection))::Bool
