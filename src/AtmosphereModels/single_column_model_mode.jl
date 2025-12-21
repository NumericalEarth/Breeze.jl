#####
##### Implements a "single column model mode" for AtmosphereModel
#####
#
# In single column mode:
# - Horizontal dimensions are Flat (no horizontal gradients)
# - No pressure solve is needed
# - No vertical velocity tendency needs to be computed
# - Vertical velocity remains zero (or prescribed)

using Oceananigans: Oceananigans, fields
using Oceananigans.Grids: AbstractGrid, Flat, Bounded
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

# AnelasticFormulation is brought into scope by the parent module via
# using .AnelasticFormulations: AnelasticFormulation

# Note: validate_momentum_advection and validate_tracer_advection for SingleColumnGrid
# are already defined in Oceananigans.Models.HydrostaticFreeSurfaceModels, so we reuse those.

#####
##### SingleColumnGrid type alias
#####

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}
const SingleColumnAnelasticModel = AtmosphereModel{<:AnelasticFormulation, <:Any, <:Any, <:SingleColumnGrid}

#####
##### Model constructor utilities
#####

# No pressure solver needed for single column mode
formulation_pressure_solver(::AnelasticFormulation, ::SingleColumnGrid) = nothing

#####
##### Time-stepping: no pressure correction needed
#####

compute_pressure_correction!(model::SingleColumnAnelasticModel, Δt) = nothing
make_pressure_correction!(model::SingleColumnAnelasticModel, Δt) = nothing

#####
##### Tendency computation: skip vertical momentum tendency
#####

function compute_tendencies!(model::SingleColumnAnelasticModel)
    grid = model.grid
    arch = grid.architecture
    Gρu = model.timestepper.Gⁿ.ρu
    Gρv = model.timestepper.Gⁿ.ρv
    Gρw = model.timestepper.Gⁿ.ρw

    model_fields = fields(model)

    #####
    ##### Momentum tendencies (only horizontal)
    #####

    momentum_args = (
        model.formulation.reference_state.density,
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        model.momentum,
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu)
    v_args = tuple(momentum_args..., model.forcing.ρv)

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gρv, grid, v_args)

    # Zero vertical momentum tendency (no pressure solve, no buoyancy)
    fill!(Gρw, 0)

    # Arguments common to energy density, moisture density, and tracer density tendencies:
    common_args = (
        model.formulation,
        model.thermodynamic_constants,
        model.specific_moisture,
        model.velocities,
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        model_fields)

    #####
    ##### Thermodynamic density tendency (dispatches on thermodynamics type)
    #####

    compute_thermodynamic_tendency!(model, common_args)

    #####
    ##### Moisture density tendency
    #####

    ρq_args = (
        model.specific_moisture,
        Val(2),
        Val(:ρqᵗ),
        model.forcing.ρqᵗ,
        model.advection.ρqᵗ,
        common_args...)

    Gρqᵗ = model.timestepper.Gⁿ.ρqᵗ
    launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρqᵗ, grid, ρq_args)

    #####
    ##### Tracer density tendencies
    #####

    for (i, name) in enumerate(keys(model.tracers))
        ρc = model.tracers[name]

        scalar_args = (
            ρc,
            Val(i + 2),
            Val(name),
            model.forcing[name],
            model.advection[name],
            common_args...)

        Gρc = getproperty(model.timestepper.Gⁿ, name)
        launch!(arch, grid, :xyz, compute_scalar_tendency!, Gρc, grid, scalar_args)
    end

    return nothing
end

