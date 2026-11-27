using Oceananigans.Grids: AbstractGrid, Flat, Bounded, topology

using Breeze.AnelasticEquations: AnelasticDynamics

const FlatHorizontalGrid = AbstractGrid{<:Any, <:Flat, <:Flat, <:Bounded}

const SingleColumnMultirateModel =
    AtmosphereModel{<:AnelasticDynamics, <:Any, <:Any, <:MultirateInfinitesimalStep, <:FlatHorizontalGrid}

AtmosphereModels.closure_prognostics_solved_last(::SingleColumnMultirateModel) = true

function multirate_substepper(grid, ::AnelasticDynamics, prognostic_fields; kw...)
    grid isa FlatHorizontalGrid ||
        throw(ArgumentError("MultirateInfinitesimalStep supports AnelasticDynamics only on a single-column grid, " *
                            "topology = (Flat, Flat, Bounded), where the dynamics have no fast tendency; " *
                            "got topology = $(topology(grid))."))
    return nothing
end

step_start_storage(grid, ::Nothing, tableau, relinearize_stages) = nothing

multirate_post_loop_scalar_tendencies(::Nothing, post_loop_scalar_tendencies) = false

transport_density_accumulator_storage(::Nothing, storage, grid, U⁰, fold, s) = ()

fast_tendency_storage(::AccumulatorStorage, ::PreLoopScalarTendencies, grid, prognostic_fields, dynamics, ::Nothing,
                      moisture_forcing_names) = nothing

AtmosphereModels.slow_dynamics(::MultirateInfinitesimalStep, dynamics::AnelasticDynamics) = dynamics

const SubstepperFreeMultirateStep = MultirateInfinitesimalStep{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}

Oceananigans.prognostic_state(::SubstepperFreeMultirateStep) = nothing

function assemble_single_column_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt)
    for name in keys(prognostic_fields(model))
        assemble_stage_forcing!(ts, model, Δt, name, SlowTendencySplit())
    end
    return nothing
end

function single_column_multirate_stage!(model, Δt, dᵢ)
    ts = model.timestepper
    grid = model.grid
    arch = grid.architecture
    Δt_stage = dᵢ * Δt
    kernel_Δt = kernel_time_step(arch, grid, Δt_stage)

    compute_flux_bc_tendencies!(model)
    compute_closure_tendencies!(model)
    closure_before_stage_recombination!(ts, model, kernel_Δt)
    assemble_single_column_stage_forcing!(ts, model, Δt)

    U⁰ = stage_start_state(ts, model)
    Gⁿ = ts.Gⁿ
    prognostic = prognostic_fields(model)
    thermodynamic_name = thermodynamic_density_name(model.formulation)

    rounding = stage_rounding(ts, ts.stage[])

    for (field_number, (name, u, u⁰, G)) in enumerate(zip(keys(prognostic), prognostic, U⁰, Gⁿ))
        key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
        launch_rk3_substep!(rounding, arch, grid, u, u⁰, G, kernel_Δt, key)

        skip_vertical_diffusion(model, name) && continue
        name === thermodynamic_name && continue
        AtmosphereModels.solved_after_thermodynamic_variable(model.closure, name) && continue

        closure_before_implicit_solve!(model.closure_fields, model.closure, model, name)
        implicit_stage_step!(model, name, u, Δt_stage)
    end

    isnothing(ts.implicit_solver) || implicit_sedimentation_step!(model, Δt_stage, model.velocities)
    closure_before_thermodynamic_solve!(model.closure_fields, model.closure, model, Δt_stage)
    implicit_stage_step!(model, thermodynamic_name, prognostic[thermodynamic_name], Δt_stage)
    for (name, u) in zip(keys(prognostic), prognostic)
        AtmosphereModels.solved_after_thermodynamic_variable(model.closure, name) || continue
        closure_before_implicit_solve!(model.closure_fields, model.closure, model, name)
        implicit_stage_step!(model, name, u, Δt_stage)
    end
    closure_after_implicit_solves!(model.closure_fields, model.closure, model, Δt_stage)

    compute_pressure_correction!(model, Δt_stage)
    make_pressure_correction!(model, Δt_stage)

    return nothing
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Step forward a single-column `model` with anelastic dynamics one time step `Δt` with the slow
method of a multirate infinitesimal step (see `single_column_multirate_stage!`).
"""
function OceananigansTimeSteppers.time_step!(model::SingleColumnMultirateModel, Δt; callbacks=[])
    ts = model.timestepper
    ts.current_Δt[] = Δt
    ts.next_stage[] = 1
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    (; c, d) = ts.tableau
    s = length(d)
    tⁿ⁺¹ = next_time(model.clock, Δt)

    store_outer_step_state!(ts, model)

    for i in 1:s
        ts.stage[] = i
        single_column_multirate_stage!(model, Δt, d[i])

        if i < s
            ts.next_stage[] = i + 1
            Δc = i == 1 ? c[1] : c[i] - c[i-1]
            tick_stage!(model.clock, Δc * Δt)
            update_stage_state!(model, callbacks, i)
            store_stage_increment!(ts, model, i, Δt)
        end
    end

    ts.next_stage[] = 1

    corrected_Δt = time_difference_in_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
    AtmosphereModels.update_state_between_radiation_solves!(model, callbacks)

    microphysics_model_update!(model.microphysics, model)
    step_lagrangian_particles!(model, Δt)

    return nothing
end
