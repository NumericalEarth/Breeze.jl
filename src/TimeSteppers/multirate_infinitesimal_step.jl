using Breeze.AtmosphereModels: thermodynamic_density_name
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!

using Oceananigans: prognostic_fields, CenterField, architecture
using Oceananigans.Utils: launch!, KernelParameters
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

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

    substepper = build_acoustic_substepper(grid, dynamics, prognostic_fields; cache_advecting_state)

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

# How a field's full tendency splits into the part the acoustic loop integrates (kept at
# weight one, from the stage-entry state) and the slow part MIS recombines across stages.
struct SlowTendencySplit end                  # momentum, moisture, tracers: all slow
struct ContinuityTendencySplit end            # continuity: all fast
struct ThermodynamicTendencySplit{C}          # thermodynamic: fast part is C = -∇·(θᴸ (ρu)ᴸ)
    fast_tendency :: C
end

Adapt.adapt_structure(to, s::ThermodynamicTendencySplit) = ThermodynamicTendencySplit(adapt(to, s.fast_tendency))

@inline slow_part(::SlowTendencySplit, G, I) = @inbounds G[I]
@inline slow_part(::ContinuityTendencySplit, G, I) = zero(eltype(G))
@inline slow_part(s::ThermodynamicTendencySplit, G, I) = @inbounds G[I] - s.fast_tendency[I]

tendency_split(name, θ_name, C) = name === :ρᵈ ? ContinuityTendencySplit() :
                                  name === θ_name ? ThermodynamicTendencySplit(C) : SlowTendencySplit()

# One pass per field: store Sᵢ, then form the stage forcing
#   G = (fast part at Yᵢ₋₁) + (1/dᵢ)[βᵢᵢ Sᵢ + Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt)]
#     = Gᵢ + (βᵢᵢ/dᵢ - 1) Sᵢ + (1/dᵢ) Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt),
# written in the second form so a diagonal tableau (Wicker–Skamarock) leaves Gᵢ bitwise intact,
# and the stage start state U⁰ = Uⁿ + Σⱼ αᵢⱼ ΔYⱼ.
@kernel function _assemble_multirate_stage!(G, U⁰, S, ΔY, Uⁿ, split, i, wβ, wγ, wα)
    I = @index(Global, Cartesian)
    @inbounds begin
        Sᵢ = slow_part(split, G, I)
        Gᵢ = G[I] + (wβ[i] - 1) * Sᵢ
        uᵢ = Uⁿ[I]
        for j in 1:i-1
            Gᵢ += wβ[j] * S[j][I] + wγ[j] * ΔY[j][I]
            uᵢ += wα[j] * ΔY[j][I]
        end
        S[i][I] = Sᵢ
        G[I] = Gᵢ
        U⁰[I] = uᵢ
    end
end

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
    s = length(d)
    θ_name = thermodynamic_density_name(model.formulation)
    grid = model.grid
    arch = architecture(grid)

    compute_linearized_thermodynamic_flux_divergence!(ts.fast_thermodynamic_tendency, ts.substepper, model)
    C = parent(ts.fast_thermodynamic_tendency)

    dᵢ = d[i]
    wβ = ntuple(j -> β[i, j] / dᵢ, s)
    wγ = ntuple(j -> γ[i, j] / (dᵢ * Δt), s)
    wα = ntuple(j -> α[i, j], s)

    for name in keys(ts.Gⁿ)
        G = parent(ts.Gⁿ[name])
        S = map(Sⱼ -> parent(Sⱼ[name]), ts.slow_tendencies)
        ΔY = map(ΔYⱼ -> parent(ΔYⱼ[name]), ts.stage_increments)
        split = tendency_split(name, θ_name, C)
        launch!(arch, grid, KernelParameters(size(G), (0, 0, 0)), _assemble_multirate_stage!,
                G, parent(ts.U⁰[name]), S, ΔY, parent(ts.Uⁿ[name]), split, i, wβ, wγ, wα)
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

    complete_acoustic_time_step!(model, Δt, tⁿ⁺¹, callbacks)

    return nothing
end
