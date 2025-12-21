#####
##### Runge-Kutta 3rd order time-stepping for AtmosphereModel
#####
#
# This implementation follows the HydrostaticFreeSurfaceModel pattern of stepping
# momentum and scalar fields separately. This is essential for compatibility with
# TKE-based closures (CATKE, TKEDissipation) which require proper tracer indexing
# for the implicit vertical diffusion solver.
#
# Key difference from NonhydrostaticModel:
# - Momentum fields are stepped with `tracer_index = nothing` (uses viscosity)
# - Scalar fields are stepped with `tracer_index = Val(i)` where i starts at 1
#   (uses tracer diffusivity with proper indexing into closure coefficient tuples)
#
# Note: TKE-based closures with VerticallyImplicitTimeDiscretization are currently
# best used with QuasiAdamsBashforth2 timestepper, as the TKE equation stepping in
# Oceananigans assumes AB2-style parameters (χ). For RK3 with TKE closures,
# consider using ExplicitTimeDiscretization.

using Oceananigans: fields
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: rk3_substep_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, FlavorOfTD

import Oceananigans.TimeSteppers: rk3_substep!

#####
##### Main RK3 substep function
#####

function rk3_substep!(model::AtmosphereModel, Δt, γⁿ, ζⁿ)
    grid = model.grid
    FT = eltype(grid)
    Δt = convert(FT, Δt)

    # Step momentum and scalars separately for proper tracer indexing
    rk3_substep_momentum!(model, Δt, γⁿ, ζⁿ)
    rk3_substep_scalars!(model, Δt, γⁿ, ζⁿ)

    return nothing
end

#####
##### Step momentum fields
#####

function rk3_substep_momentum!(model, Δt, γⁿ, ζⁿ)
    arch = architecture(model.grid)
    grid = model.grid
    FT = eltype(grid)
    Δt = convert(FT, Δt)

    for name in (:ρu, :ρv, :ρw)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        momentum_field = model.momentum[name]

        launch!(arch, grid, :xyz, rk3_substep_field!, momentum_field, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)

        # Momentum uses viscosity, not tracer diffusivity - pass nothing for tracer_index
        implicit_step!(momentum_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       nothing,
                       model.clock,
                       fields(model),
                       Δt)
    end

    return nothing
end

#####
##### Step scalar fields
#####

function rk3_substep_scalars!(model, Δt, γⁿ, ζⁿ)
    arch = architecture(model.grid)
    grid = model.grid
    FT = eltype(grid)
    Δt = convert(FT, Δt)
    closure = model.closure

    catke_in_closures = hasclosure(closure, FlavorOfCATKE)
    td_in_closures = hasclosure(closure, FlavorOfTD)

    scalars = scalar_fields(model)

    for (scalar_index, scalar_name) in enumerate(propertynames(scalars))

        # TKE closure tracers are stepped by their own time-stepping routines
        if catke_in_closures && scalar_name == :e
            @debug "Skipping RK3 step for e (handled by CATKE)"
        elseif td_in_closures && scalar_name == :e
            @debug "Skipping RK3 step for e (handled by TKE-Dissipation)"
        elseif td_in_closures && scalar_name == :ϵ
            @debug "Skipping RK3 step for ϵ (handled by TKE-Dissipation)"
        else
            Gⁿ = model.timestepper.Gⁿ[scalar_name]
            G⁻ = model.timestepper.G⁻[scalar_name]
            scalar_field = scalars[scalar_name]

            launch!(arch, grid, :xyz, rk3_substep_field!, scalar_field, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)

            # Scalars use tracer diffusivity with proper indexing starting at 1
            implicit_step!(scalar_field,
                           model.timestepper.implicit_solver,
                           model.closure,
                           model.closure_fields,
                           Val(scalar_index),
                           model.clock,
                           fields(model),
                           Δt)
        end
    end

    return nothing
end

