#####
##### MultirateInfinitesimalStep for anelastic dynamics on a single-column grid
#####
##### On a grid with `topology = (Flat, Flat, Bounded)` the anelastic dynamics have no fast
##### tendency: every horizontal difference vanishes, the mass constraint with rigid boundaries
##### gives w ≡ 0 (see `single_column_mode.jl`), and so there is no pressure gradient, no buoyancy
##### acting on a prognostic, and no flux divergence of the thermodynamic variable by the resolved
##### flow. The multirate infinitesimal step of stage i,
#####
#####     V(0) = Uⁿ + Σⱼ αᵢⱼ ΔYⱼ,    ∂τ V = f_fast(V) + (1/dᵢ) [Σⱼ γᵢⱼ ΔYⱼ / Δt + Σⱼ βᵢⱼ Sⱼ],    0 ≤ τ ≤ dᵢ Δt,
#####
##### then integrates a constant and is exact in one explicit update over dᵢ Δt,
#####
#####     Vᵢ = Uⁿ + Σⱼ (αᵢⱼ + γᵢⱼ) ΔYⱼ + Δt Σⱼ βᵢⱼ Sⱼ,
#####
##### followed by the vertically implicit solves over dᵢ Δt that give Yᵢ, as the stage with
##### acoustic substepping applies them (`scalar_substep!`, `implicit_substep!`). No substepper is
##### built and no acoustic loop runs. What a turbulence closure sees is what it sees in three
##### dimensions: the same start states, slow-tendency and increment weights, stage intervals,
##### solves and stage hooks, a tendency evaluation per stage, and one `step_closure_prognostics!`
##### per step. A step costs s tendency evaluations and s sets of solves (SSP-RK3: three).
#####

using Oceananigans.Grids: AbstractGrid, Flat, Bounded, topology

using Breeze.AnelasticEquations: AnelasticDynamics

# The alias of `single_column_mode.jl`, which is loaded after the time steppers.
const FlatHorizontalGrid = AbstractGrid{<:Any, <:Flat, <:Flat, <:Bounded}

# Anelastic dynamics stepped by a multirate infinitesimal step, on a single column or a forest of
# independent columns. The grid is the fifth `AtmosphereModel` type parameter.
const SingleColumnMultirateModel =
    AtmosphereModel{<:AnelasticDynamics, <:Any, <:Any, <:MultirateInfinitesimalStep, <:FlatHorizontalGrid}

function multirate_substepper(grid, ::AnelasticDynamics, prognostic_fields; kw...)
    grid isa FlatHorizontalGrid ||
        throw(ArgumentError("MultirateInfinitesimalStep supports AnelasticDynamics only on a single-column grid, " *
                            "topology = (Flat, Flat, Bounded), where the dynamics have no fast tendency; " *
                            "got topology = $(topology(grid))."))
    return nothing
end

# No stage-entry pressure gradient or buoyancy whose remainder would be recombined.
step_start_storage(grid, ::Nothing, tableau, relinearize_stages) = nothing

# Without an acoustic loop there is no "after the loop": `update_state!` builds every tendency
# whole, before the stage, and every tendency is slow, so no fast part needs scratch.
multirate_post_loop_scalar_tendencies(::Nothing, post_loop_scalar_tendencies) = false

# No acoustic loop transports the dry density, so there is no transport density tendency to combine.
transport_density_accumulator_storage(::Nothing, storage, grid, U⁰, fold, s) = ()

fast_tendency_storage(::AccumulatorStorage, ::PreLoopScalarTendencies, grid, prognostic_fields, dynamics, ::Nothing,
                      moisture_forcing_names) = nothing

# The tendencies `update_state!` builds are the whole tendency: nothing is integrated apart.
AtmosphereModels.slow_dynamics(::MultirateInfinitesimalStep, dynamics::AnelasticDynamics) = dynamics

const SubstepperFreeMultirateStep = MultirateInfinitesimalStep{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}

Oceananigans.prognostic_state(::SubstepperFreeMultirateStep) = nothing

# Every prognostic's tendency is slow.
function assemble_single_column_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt)
    for name in keys(prognostic_fields(model))
        assemble_stage_forcing!(ts, model, Δt, name, SlowTendencySplit())
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Run stage `ts.stage[]` of a multirate infinitesimal step with no fast tendency: complete the
tendencies `update_state!` built at ``Yᵢ₋₁``, recombine them and the earlier stages' increments
into the stage forcing and start state, update every prognostic over ``dᵢ Δt``, and apply the
vertically implicit solves over ``dᵢ Δt``, in the order `ssp_rk3_substep!` applies them (the
thermodynamic variable last) and with the same closure hooks around them.
"""
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

    # Every prognostic is a slow variable here, so each takes the round-off treatment that the
    # scalars take in the stage with acoustic substepping (see `stage_rounding`).
    rounding = stage_rounding(ts, ts.stage[])

    for (field_number, (name, u, u⁰, G)) in enumerate(zip(keys(prognostic), prognostic, U⁰, Gⁿ))
        key = rounding_key(rounding, model.clock.iteration, model.clock.stage, field_number)
        launch_rk3_substep!(rounding, arch, grid, u, u⁰, G, kernel_Δt, key)

        skip_vertical_diffusion(model, name) && continue
        name === thermodynamic_name && continue

        closure_before_implicit_solve!(model.closure_fields, model.closure, model, name)
        implicit_stage_step!(model, name, u, Δt_stage)
    end

    isnothing(ts.implicit_solver) || implicit_sedimentation_step!(model, Δt_stage, model.velocities)
    closure_before_thermodynamic_solve!(model.closure_fields, model.closure, model, Δt_stage)
    implicit_stage_step!(model, thermodynamic_name, prognostic[thermodynamic_name], Δt_stage)
    closure_after_implicit_solves!(model.closure_fields, model.closure, model, Δt_stage)

    # No-ops on a single column, where w ≡ 0; kept so the stage is that of the anelastic dynamics.
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
    tⁿ⁺¹ = model.clock.time + Δt

    store_outer_step_state!(ts, model)

    for i in 1:s
        ts.stage[] = i
        single_column_multirate_stage!(model, Δt, d[i])

        if i < s
            ts.next_stage[] = i + 1
            Δc = i == 1 ? c[1] : c[i] - c[i-1]
            tick_stage!(model.clock, Δc * Δt)
            update_state!(model, callbacks; compute_tendencies = true)
            # After `update_state!`, as with acoustic substepping: the increments are those of the
            # state it leaves, with its negative-moisture correction (see `stage_increment_source`).
            store_stage_increment!(ts, model, i, Δt)
        end
    end

    ts.next_stage[] = 1

    corrected_Δt = time_difference_in_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    # The operator-split microphysics update, once per step, and the particles (see `SSPRungeKutta3`).
    microphysics_model_update!(model.microphysics, model)
    step_lagrangian_particles!(model, Δt)

    return nothing
end
