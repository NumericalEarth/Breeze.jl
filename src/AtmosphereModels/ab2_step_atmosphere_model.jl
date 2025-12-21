#####
##### Adams-Bashforth 2nd order time-stepping for AtmosphereModel
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

using Oceananigans: fields
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, FlavorOfTD

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Main AB2 step function
#####

function ab2_step!(model::AtmosphereModel, Δt)
    grid = model.grid
    FT = eltype(grid)
    χ = convert(FT, model.timestepper.χ)
    Δt = convert(FT, Δt)

    # Step momentum and scalars separately for proper tracer indexing
    ab2_step_momentum!(model, Δt, χ)
    ab2_step_scalars!(model, Δt, χ)

    return nothing
end

#####
##### Step momentum fields
#####

function ab2_step_momentum!(model, Δt, χ)
    arch = architecture(model.grid)
    grid = model.grid

    for name in (:ρu, :ρv, :ρw)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        momentum_field = model.momentum[name]

        launch!(arch, grid, :xyz, ab2_step_field!, momentum_field, Δt, χ, Gⁿ, G⁻)

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

"""
    scalar_fields(model::AtmosphereModel)

Return a NamedTuple of all scalar (non-momentum) prognostic fields in the order
they appear in closure coefficient tuples.

The order is: thermodynamic density, moisture density, microphysical fields, tracers.
This matches the order used in `build_closure_fields` via `scalar_names`.
"""
function scalar_fields(model::AtmosphereModel)
    # Thermodynamic density field (ρθ or ρe)
    thermo_fields = prognostic_fields(model.formulation)
    
    # Moisture density
    moisture_fields = (; ρqᵗ = model.moisture_density)
    
    # Microphysical prognostic fields
    μ_names = prognostic_field_names(model.microphysics)
    μ_fields = NamedTuple{μ_names}(model.microphysical_fields[name] for name in μ_names)
    
    # User tracers (includes closure-required tracers like e, ϵ)
    tracer_fields = model.tracers
    
    return merge(thermo_fields, moisture_fields, μ_fields, tracer_fields)
end

# Helper to check for TKE closures
hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

function ab2_step_scalars!(model, Δt, χ)
    arch = architecture(model.grid)
    grid = model.grid
    closure = model.closure

    catke_in_closures = hasclosure(closure, FlavorOfCATKE)
    td_in_closures = hasclosure(closure, FlavorOfTD)

    scalars = scalar_fields(model)

    for (scalar_index, scalar_name) in enumerate(propertynames(scalars))

        # TKE closure tracers are stepped by their own time-stepping routines
        if catke_in_closures && scalar_name == :e
            @debug "Skipping AB2 step for e (handled by CATKE)"
        elseif td_in_closures && scalar_name == :e
            @debug "Skipping AB2 step for e (handled by TKE-Dissipation)"
        elseif td_in_closures && scalar_name == :ϵ
            @debug "Skipping AB2 step for ϵ (handled by TKE-Dissipation)"
        else
            Gⁿ = model.timestepper.Gⁿ[scalar_name]
            G⁻ = model.timestepper.G⁻[scalar_name]
            scalar_field = scalars[scalar_name]

            launch!(arch, grid, :xyz, ab2_step_field!, scalar_field, Δt, χ, Gⁿ, G⁻)

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

