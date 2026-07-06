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
##### automatically from `model.advection`. `CellAdvectionTimescale(formulation)` is the explicit
##### override — pass it to the wizard's `cell_advection_timescale` keyword to force a horizontal
##### or three-dimensional timescale, or use it in an `AdvectiveCFL` diagnostic to monitor the true
##### three-dimensional CFL even while the wizard floats on the horizontal one.

using Oceananigans.Advection: Advection, cell_advection_timescale, needs_implicit_solver
using Oceananigans.Fields: ZeroField
using Oceananigans.TurbulenceClosures: HorizontalFormulation, ThreeDimensionalFormulation

"""
$(TYPEDSIGNATURES)

A callable that returns the advective timescale of a `model` restricted to the directions of
`formulation`: `HorizontalFormulation()` counts only the horizontal advective CFL (dropping the
vertical term), `ThreeDimensionalFormulation()` counts all three directions. Pass it to the
`cell_advection_timescale` keyword of `TimeStepWizard` / `conjure_time_step_wizard!`, or to
`AdvectiveCFL`, to control or monitor which directions bind the time step.
"""
struct CellAdvectionTimescale{F}
    formulation :: F
end

CellAdvectionTimescale() = CellAdvectionTimescale(ThreeDimensionalFormulation())

@inline (τ::CellAdvectionTimescale)(model) = cell_advection_timescale(model, τ.formulation)

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
    if all_vertical_advection_is_implicit(model.advection)
        return cell_advection_timescale(model, HorizontalFormulation())
    else
        return cell_advection_timescale(model, ThreeDimensionalFormulation())
    end
end

# `nothing` schemes advect nothing (no vertical CFL); every other scheme must be AIVA.
all_vertical_advection_is_implicit(advection::NamedTuple) =
    all(scheme -> scheme === nothing || needs_implicit_solver(scheme), values(advection))
