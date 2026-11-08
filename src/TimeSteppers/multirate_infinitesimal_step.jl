using Breeze.AtmosphereModels: thermodynamic_density_name
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!

using Oceananigans: prognostic_fields, CenterField

"""
$(TYPEDEF)

Multirate infinitesimal step (MIS) outer time stepper
([Wensch et al. (2009)](@cite WenschKnothGalant2009); [Knoth and Wensch (2014)](@cite KnothWensch2014))
for fully compressible dynamics, driving the same linearized acoustic substep loop as
[`AcousticRungeKutta3`](@ref). Stage ``i`` of an ``s``-stage method starts the acoustic
integration from ``Uⁿ + Σⱼ αᵢⱼ ΔYⱼ`` (``ΔYⱼ = Yⱼ - Uⁿ``) and integrates over ``dᵢ Δt`` with the
frozen slow forcing ``(1/dᵢ)[Σⱼ γᵢⱼ ΔYⱼ / Δt + Σⱼ βᵢⱼ Sⱼ]``, where ``Sⱼ`` is the slow tendency
evaluated at ``Yⱼ₋₁``. The fast tendency is linearized about the latest stage ``Yᵢ₋₁``.

The slow tendency excludes everything the acoustic loop integrates: momentum tendencies never
contain the pressure gradient or buoyancy; the continuity tendency ``-∇·(ρu)`` is entirely fast;
and the thermodynamic slow tendency is the full tendency minus ``-∇·(θ (ρu))`` in the loop's
discretization (see [`compute_linearized_thermodynamic_flux_divergence!`](@ref)). With
the `:WickerSkamarock3` coefficients the scheme reduces to `AcousticRungeKutta3`.

Fields
======

- `tableau`: the [`MultirateInfinitesimalStepTableau`](@ref).
- `U⁰`: start state of the current stage's acoustic integration.
- `Gⁿ`: tendency storage; holds the combined slow forcing during each acoustic loop.
- `implicit_solver`, `substepper`: as for `AcousticRungeKutta3`.
- `Uⁿ`: the outer-step-start state.
- `stage_increments`: ``ΔYⱼ = Yⱼ - Uⁿ`` for ``j = 1, …, s - 1``.
- `slow_tendencies`: ``Sⱼ`` for ``j = 1, …, s``.
- `fast_thermodynamic_tendency`: scratch for the stage-entry fast thermodynamic tendency.
- `stage`: the stage being integrated.
"""
struct MultirateInfinitesimalStep{TB, U0, TG, TI, AS, UN, DY, ST, FC} <: AcousticTimeStepper
    tableau :: TB
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    Uⁿ :: UN
    stage_increments :: DY
    slow_tendencies :: ST
    fast_thermodynamic_tendency :: FC
    stage :: Base.RefValue{Int}
end

"""
$(TYPEDSIGNATURES)

Construct a [`MultirateInfinitesimalStep`](@ref) time stepper. `coefficients` is a `Symbol`
naming a set in [`multirate_infinitesimal_step_tableau`](@ref) (default `:MIS2`) or a
[`MultirateInfinitesimalStepTableau`](@ref).

Select it with `AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep,
timestepper_kwargs = (; coefficients = :MIS2))`.
"""
function MultirateInfinitesimalStep(grid, prognostic_fields;
                                    dynamics,
                                    coefficients = :MIS2,
                                    implicit_solver = nothing,
                                    cache_advecting_state = false)

    FT = eltype(grid)
    tableau = multirate_infinitesimal_step_tableau(coefficients, FT)
    s = number_of_stages(tableau)

    substepper = AcousticSubstepper(grid, dynamics.time_discretization; cache_advecting_state)

    U⁰ = map(similar, prognostic_fields)
    Gⁿ = map(similar, prognostic_fields)
    Uⁿ = map(similar, prognostic_fields)
    stage_increments = ntuple(_ -> map(similar, prognostic_fields), s - 1)
    slow_tendencies = ntuple(_ -> map(similar, prognostic_fields), s)
    fast_thermodynamic_tendency = CenterField(grid)

    return MultirateInfinitesimalStep(tableau, U⁰, Gⁿ, implicit_solver, substepper,
                                      Uⁿ, stage_increments, slow_tendencies,
                                      fast_thermodynamic_tendency, Ref(1))
end

const MultirateAcousticModel = AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:MultirateInfinitesimalStep}

@inline function final_stage_increment(ts::MultirateInfinitesimalStep)
    c = ts.tableau.c
    s = length(c)
    return s == 1 ? c[1] : c[s] - c[s-1]
end

#####
##### Stage assembly
#####

"""
$(TYPEDSIGNATURES)

Assemble the start state `U⁰` and slow forcing `Gⁿ` of MIS stage ``i = `` `ts.stage[]`. On
entry `Gⁿ` holds the full tendency at ``Yᵢ₋₁``; its slow part is stored as ``Sᵢ`` first.
The continuity forcing keeps the current (fast) mass-flux divergence, and the thermodynamic
forcing keeps the current fast flux divergence at weight one, so only slow parts are
recombined across stages.
"""
function assemble_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt)
    i = ts.stage[]
    (; α, β, γ, d) = ts.tableau
    names = keys(ts.Gⁿ)
    θ_name = thermodynamic_density_name(model.formulation)
    C = parent(ts.fast_thermodynamic_tendency)

    compute_linearized_thermodynamic_flux_divergence!(ts.fast_thermodynamic_tendency, ts.substepper, model)

    Sᵢ = ts.slow_tendencies[i]
    dᵢ = d[i]

    for name in names
        G = parent(ts.Gⁿ[name])
        S = parent(Sᵢ[name])
        U⁰ = parent(ts.U⁰[name])
        Uⁿ = parent(ts.Uⁿ[name])

        if name === :ρᵈ
            # Continuity is entirely fast: keep the stage-entry divergence at weight one.
            fill!(S, 0)
        elseif name === θ_name
            S .= G .- C
            G .= C .+ (β[i, i] / dᵢ) .* S
        else
            S .= G
            G .= (β[i, i] / dᵢ) .* S
        end

        U⁰ .= Uⁿ

        for j in 1:i-1
            ΔY = parent(ts.stage_increments[j][name])
            Sⱼ = parent(ts.slow_tendencies[j][name])
            wβ = β[i, j] / dᵢ
            wγ = γ[i, j] / (dᵢ * Δt)
            G .+= wβ .* Sⱼ .+ wγ .* ΔY
            U⁰ .+= α[i, j] .* ΔY
        end
    end

    return nothing
end

function store_stage_increment!(ts::MultirateInfinitesimalStep, model, i)
    for (ΔY, u, uⁿ) in zip(ts.stage_increments[i], prognostic_fields(model), ts.Uⁿ)
        parent(ΔY) .= parent(u) .- parent(uⁿ)
    end
    return nothing
end

function store_outer_step_state!(ts::MultirateInfinitesimalStep, model)
    for (uⁿ, u) in zip(ts.Uⁿ, prognostic_fields(model))
        parent(uⁿ) .= parent(u)
    end
    return nothing
end

#####
##### Time stepping
#####

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with a multirate infinitesimal step method and
linearized acoustic substepping.
"""
function OceananigansTimeSteppers.time_step!(model::MultirateAcousticModel, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    ts = model.timestepper
    (; c, d) = ts.tableau
    s = length(d)
    tⁿ⁺¹ = model.clock.time + Δt

    store_outer_step_state!(ts, model)
    freeze_linearization_state!(ts.substepper, model)

    for i in 1:s
        ts.stage[] = i
        acoustic_rk3_substep!(model, Δt, d[i])

        if i < s
            store_stage_increment!(ts, model, i)
            Δc = i == 1 ? c[1] : c[i] - c[i-1]
            tick_stage!(model.clock, Δc * Δt)
            update_state!(model, callbacks; compute_tendencies = true)
            cache_transport_velocity!(model)
        end
    end

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    cache_transport_velocity!(model)
    microphysics_model_update!(model.microphysics, model)
    step_lagrangian_particles!(model, Δt)

    return nothing
end
