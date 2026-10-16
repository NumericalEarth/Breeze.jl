#####
##### Direction-aware advective timescale for the time-step wizard
#####
##### The `TimeStepWizard` floats Δt at a target advective CFL by calling
##### `cell_advection_timescale(model)` — a `minimum` over the grid of
#####
#####   τ = 1 / (|u|/Δx + |v|/Δy + |w|/Δz).
#####
##### Sedimenting prognostics are advected by the air velocity plus their fall velocity,
##### so `|w|` is replaced by the largest vertical transport speed among the air and every
##### explicitly advected sedimenting prognostic. This is evaluated pointwise inside the
##### single reduction, so the cost does not grow with the number of sedimenting species.
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

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: Advection, cell_advection_timescale
using Oceananigans.BoundaryConditions: needs_implicit_solver
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: Center, Face
using Oceananigans.TurbulenceClosures: HorizontalFormulation, ThreeDimensionalFormulation

"""
$(TYPEDSIGNATURES)

A callable that returns the advective timescale of a `model` restricted to the directions of
`formulation`: `HorizontalFormulation()` counts only the horizontal advective CFL (dropping the
vertical term), `ThreeDimensionalFormulation()` counts all three directions. Pass it to the
`cell_advection_timescale` keyword of `TimeStepWizard` / `conjure_time_step_wizard!`, or as the
`timescale` argument of `CFL` (`CFL(Δt, CellAdvectionTimescale(...))`), to control or monitor
which directions bind the time step.

The vertical term uses the largest transport speed among the air and every explicitly
advected sedimenting prognostic (air velocity plus fall velocity), so falling precipitation
also constrains explicit advection.
"""
struct CellAdvectionTimescale{F}
    formulation :: F
end

(τ::CellAdvectionTimescale)(model) = cell_advection_timescale(model, τ.formulation)

# The vertical advecting velocity is Cartesian `w` on height-coordinate grids and the contravariant
# `w̃` on terrain-following grids (see `advecting_vertical_velocity`). Its magnitude is replaced by
# the largest vertical transport speed, so Oceananigans' own kernel computes the sedimentation-aware
# timescale in one reduction with the same topology/Flat handling.
function Advection.cell_advection_timescale(model::AtmosphereModel, ::ThreeDimensionalFormulation)
    u, v, _ = model.velocities
    w = advecting_vertical_velocity(model.dynamics, model.velocities)
    wᵗ = maximum_vertical_transport_speed(model, w)
    return cell_advection_timescale(model.grid, (u, v, wᵗ))
end

# A `ZeroField` in the vertical slot makes the `|w|/Δz` term identically zero.
function Advection.cell_advection_timescale(model::AtmosphereModel, ::HorizontalFormulation)
    u, v, _ = model.velocities
    return cell_advection_timescale(model.grid, (u, v, ZeroField()))
end

# Largest vertical transport speed among the air and the sedimenting prognostics, as a lazy
# field at the vertical faces where `w` and the fall velocities live. With no sedimentation
# the air velocity itself is returned, so the timescale reduces to Oceananigans' own.
function maximum_vertical_transport_speed(model, w)
    fall_velocities = sedimentation_velocities(model)
    isempty(fall_velocities) && return w
    return KernelFunctionOperation{Center, Center, Face}(vertical_transport_speedᶜᶜᶠ, model.grid, w, fall_velocities)
end

@inline vertical_transport_speedᶜᶜᶠ(i, j, k, grid, w, fall_velocities) =
    vertical_transport_speed(i, j, k, @inbounds(w[i, j, k]), fall_velocities)

# Compile-time recursion over the fall velocities: the air's own speed is the base case, so
# prognostics without sedimentation retain their CFL limit as well.
@inline vertical_transport_speed(i, j, k, w, ::Tuple{}) = abs(w)

@inline function vertical_transport_speed(i, j, k, w, fall_velocities::Tuple)
    wᶠ = @inbounds first(fall_velocities)[i, j, k]
    return max(abs(w + wᶠ), vertical_transport_speed(i, j, k, w, Base.tail(fall_velocities)))
end

# Fall velocities of the sedimenting prognostics, i.e. the vertical component of
# `microphysical_velocities` (which carries no horizontal part). A prognostic whose advection
# scheme is `nothing` has no advective flux at all, sedimentation included (`div_Uc` returns
# zero), so its fall velocity imposes no CFL limit and is skipped.
function sedimentation_velocities(model)
    names = prognostic_field_names(model.microphysics)
    velocities = map(names) do name
        transport = microphysical_velocities(model.microphysics, model.microphysical_fields, Val(name))
        advected = model.advection[name] !== nothing
        return advected && transport !== nothing ? transport.w : nothing
    end
    return filter(!isnothing, velocities)
end

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
