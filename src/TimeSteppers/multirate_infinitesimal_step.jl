using Breeze.AtmosphereModels: thermodynamic_density_name
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!

using Oceananigans: prognostic_fields, CenterField, architecture
using Oceananigans.Utils: launch!
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
evaluated at ``Yⱼ₋₁``. The fast tendency is linearized once per step, about ``Uⁿ``.

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
- `accumulators`: for each stage ``k = 3, …, s``, the parts of its forcing and start state known so far,
  `forcing` ``= (1/dₖ) Σⱼ (βₖⱼ Sⱼ + γₖⱼ ΔYⱼ / Δt)`` and `offset` ``= Σⱼ αₖⱼ ΔYⱼ``, accumulated as each
  ``Sⱼ`` and ``ΔYⱼ = Yⱼ - Uⁿ`` becomes available, so no ``Sⱼ`` or ``ΔYⱼ`` is stored. Until stage 2
  consumes them, stage 3's pair holds the raw ``S₁`` and ``ΔY₁``.
- `fast_thermodynamic_tendency`: scratch for the stage-entry fast thermodynamic tendency.
- `stage`: the stage being integrated.
- `next_stage`, `current_Δt`: the stage the next tendency evaluation feeds and the current outer step, which
  size the adaptive-implicit vertical-advection split (see `adaptive_advection_timestep`).
- `relinearize_stages`: re-linearize the fast operator at every stage entry (as `AcousticRungeKutta3` does)
  instead of once per step. Off by default: it makes the fast operator stage-dependent, which costs an
  order of accuracy; it exists to diagnose robustness in strongly nonlinear flows.
"""
struct MultirateInfinitesimalStep{TB, U0, TG, TI, AS, UN, AC, FC} <: AcousticTimeStepper
    relinearize_stages :: Bool
    tableau :: TB
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    Uⁿ :: UN
    accumulators :: AC
    fast_thermodynamic_tendency :: FC
    stage :: Base.RefValue{Int}
    next_stage :: Base.RefValue{Int}
    current_Δt :: Base.RefValue{Float64}
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
                                    cache_advecting_state = false,
                                    relinearize_stages = false)

    FT = eltype(grid)
    tableau = multirate_infinitesimal_step_tableau(coefficients, FT)
    s = number_of_stages(tableau)
    s ≥ 3 || throw(ArgumentError("MultirateInfinitesimalStep needs at least three stages; got $s."))

    substepper = build_acoustic_substepper(grid, dynamics, prognostic_fields; cache_advecting_state)

    U⁰ = map(similar, prognostic_fields)
    Gⁿ = map(similar, prognostic_fields)
    Uⁿ = map(similar, prognostic_fields)
    accumulators = ntuple(_ -> (forcing = map(similar, prognostic_fields),
                                offset = map(similar, prognostic_fields)), s - 2)
    fast_thermodynamic_tendency = CenterField(grid)

    return MultirateInfinitesimalStep(relinearize_stages, tableau, U⁰, Gⁿ, implicit_solver, substepper,
                                      Uⁿ, accumulators,
                                      fast_thermodynamic_tendency, Ref(1), Ref(1), Ref(NaN))
end

# MIS assumes one fast operator for the whole step: the linearization coefficients (θᴸ, the
# pressure coefficients) stay at the values `freeze_linearization_state!` computed from Uⁿ.
# Re-linearizing per stage makes the fast operator stage-dependent, which costs an order of
# accuracy because MIS stages start away from the stage-entry state.
refresh_stage_linearization!(ts::MultirateInfinitesimalStep, substepper, model) =
    ts.relinearize_stages ? prepare_acoustic_cache!(substepper, model) : nothing

"""
$(TYPEDSIGNATURES)

The adaptive-implicit split time step for the MIS stage that the tendencies about to be built will
drive: its fast interval ``dᵢ Δt``, so the explicit fraction frozen into those tendencies pairs with the
implicit remainder that stage applies (`implicit_substep!` over ``dᵢ Δt``). Tendencies built at the end of
a step feed stage 1 of the next, whose ``Δt`` is not yet known; the current one is used, as
`AcousticRungeKutta3` does. Stage ``i``'s forcing also recombines earlier stages' explicit tendencies
with weights ``βᵢⱼ/dᵢ``, so the explicit/implicit pairing is exact only for the diagonal term.
"""
@inline function Oceananigans.Advection.adaptive_advection_timestep(ts::MultirateInfinitesimalStep, clock)
    Δt = ifelse(isfinite(ts.current_Δt[]), ts.current_Δt[], clock.last_Δt)
    return ts.tableau.d[ts.next_stage[]] * Δt
end

"""
$(TYPEDSIGNATURES)

Whether the tendencies the next `update_state!` builds feed the first stage of an outer step, so that
state an advection scheme freezes for the whole step (see `Breeze.Advection.FrozenWeightsWENO`) is
to be refreshed from the current fields. `true` for every time stepper but MIS, which raises
`next_stage` past 1 while its stages run.
"""
tendencies_feed_first_stage(timestepper) = true
tendencies_feed_first_stage(ts::MultirateInfinitesimalStep) = ts.next_stage[] == 1

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

# Pointwise updates of tuples of arrays whose length is known at compile time, so they unroll.
@inline accumulate!(n, ::Tuple{}, ::Tuple{}, x) = nothing
@inline function accumulate!(n, A::Tuple, w::Tuple, x)
    @inbounds A[1][n] += w[1] * x
    return accumulate!(n, Base.tail(A), Base.tail(w), x)
end

@inline combine!(n, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, a, b, c) = nothing
@inline function combine!(n, A::Tuple, wa::Tuple, wb::Tuple, wc::Tuple, a, b, c)
    @inbounds A[1][n] = wa[1] * a + wb[1] * b + wc[1] * c
    return combine!(n, Base.tail(A), Base.tail(wa), Base.tail(wb), Base.tail(wc), a, b, c)
end

@inline assign!(n, ::Tuple{}, ::Tuple{}, x) = nothing
@inline function assign!(n, A::Tuple, w::Tuple, x)
    @inbounds A[1][n] = w[1] * x
    return assign!(n, Base.tail(A), Base.tail(w), x)
end

# The stage forcing is
#   G = (fast part at Yᵢ₋₁) + (1/dᵢ)[βᵢᵢ Sᵢ + Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt)] = Gᵢ + (βᵢᵢ/dᵢ - 1) Sᵢ + Fᵢ,
# written so a diagonal tableau (Wicker–Skamarock) leaves Gᵢ bitwise intact, and the start state
# is U⁰ = Uⁿ + Zᵢ with Zᵢ = Σⱼ αᵢⱼ ΔYⱼ. Each kernel makes one pass over a field's parent array.

# Stage 1: F₁ = Z₁ = 0. Keep S₁ (in stage 3's forcing accumulator) for later stages.
@kernel function _first_multirate_stage!(G, U⁰, Uⁿ, split, wᵢ, S₁)
    n = @index(Global, Linear)
    @inbounds begin
        slow = slow_part(split, G, n)
        S₁[n] = slow
        G[n] = G[n] + wᵢ * slow
        U⁰[n] = Uⁿ[n]
    end
end

# Stage 2 reads the raw S₁, ΔY₁ and turns them, with S₂, into the accumulators of stages k ≥ 3
# (whose first pair is where S₁, ΔY₁ were kept: they are read before being overwritten).
@kernel function _second_multirate_stage!(G, U⁰, Uⁿ, split, wᵢ, wβ₁, wγ₁, α₁, F, Z, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, α₁ᵏ)
    n = @index(Global, Linear)
    @inbounds begin
        S₁ = F[1][n]
        ΔY₁ = Z[1][n]
        S₂ = slow_part(split, G, n)
        G[n] = G[n] + wᵢ * S₂ + wβ₁ * S₁ + wγ₁ * ΔY₁
        U⁰[n] = Uⁿ[n] + α₁ * ΔY₁
        combine!(n, F, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, S₁, ΔY₁, S₂)
        assign!(n, Z, α₁ᵏ, ΔY₁)
    end
end

# Stage i ≥ 3 uses its completed accumulators and adds Sᵢ to those of later stages.
@kernel function _later_multirate_stage!(G, U⁰, Uⁿ, split, wᵢ, Fᵢ, Zᵢ, F, wβᵏ)
    n = @index(Global, Linear)
    @inbounds begin
        slow = slow_part(split, G, n)
        G[n] = G[n] + wᵢ * slow + Fᵢ[n]
        U⁰[n] = Uⁿ[n] + Zᵢ[n]
        accumulate!(n, F, wβᵏ, slow)
    end
end

# After stage i: ΔYᵢ = Yᵢ - Uⁿ, kept raw after stage 1 and accumulated after later stages.
@kernel function _store_first_increment!(ΔY₁, u, Uⁿ)
    n = @index(Global, Linear)
    @inbounds ΔY₁[n] = u[n] - Uⁿ[n]
end

@kernel function _accumulate_increment!(F, Z, u, Uⁿ, wγᵏ, αᵏ)
    n = @index(Global, Linear)
    @inbounds begin
        ΔY = u[n] - Uⁿ[n]
        accumulate!(n, F, wγᵏ, ΔY)
        accumulate!(n, Z, αᵏ, ΔY)
    end
end

# Weights of stage j's contributions to the stages k = i+1, …, s that still need them.
later_stages(i, s) = ntuple(m -> i + m, s - i)
forcing_weights(β, d, j, ks) = map(k -> β[k, j] / d[k], ks)
increment_weights(γ, d, Δt, j, ks) = map(k -> γ[k, j] / (d[k] * Δt), ks)
offset_weights(α, j, ks) = map(k -> α[k, j], ks)

# Accumulators of stages ks (k ≥ 3), as parent arrays of field `name`.
accumulator_arrays(ts, name, ks, part) = map(k -> parent(getproperty(ts.accumulators[k - 2], part)[name]), ks)

"""
$(TYPEDSIGNATURES)

Assemble the start state `U⁰` and slow forcing `Gⁿ` of MIS stage ``i = `` `ts.stage[]`. On
entry `Gⁿ` holds the full tendency at ``Yᵢ₋₁``. The continuity forcing keeps the current (fast)
mass-flux divergence, and the thermodynamic forcing keeps the current fast flux divergence at
weight one, so only slow parts are recombined across stages; the slow part ``Sᵢ`` is folded into
the accumulators of the later stages.
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
    wᵢ = β[i, i] / d[i] - 1
    ks = later_stages(max(i, 2), s)

    for name in keys(ts.Gⁿ)
        G = parent(ts.Gⁿ[name])
        U⁰ = parent(ts.U⁰[name])
        Uⁿ = parent(ts.Uⁿ[name])
        split = tendency_split(name, θ_name, C)
        F = accumulator_arrays(ts, name, ks, :forcing)
        size_G = (length(G),)

        if i == 1
            launch!(arch, grid, size_G, _first_multirate_stage!, G, U⁰, Uⁿ, split, wᵢ,
                    parent(ts.accumulators[1].forcing[name]))
        elseif i == 2
            Z = accumulator_arrays(ts, name, ks, :offset)
            launch!(arch, grid, size_G, _second_multirate_stage!, G, U⁰, Uⁿ, split, wᵢ,
                    β[2, 1] / d[2], γ[2, 1] / (d[2] * Δt), α[2, 1], F, Z,
                    forcing_weights(β, d, 1, ks), increment_weights(γ, d, Δt, 1, ks),
                    forcing_weights(β, d, 2, ks), offset_weights(α, 1, ks))
        else
            Fᵢ = parent(ts.accumulators[i - 2].forcing[name])
            Zᵢ = parent(ts.accumulators[i - 2].offset[name])
            launch!(arch, grid, size_G, _later_multirate_stage!, G, U⁰, Uⁿ, split, wᵢ,
                    Fᵢ, Zᵢ, F, forcing_weights(β, d, i, ks))
        end
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Fold the increment ``ΔYᵢ = Yᵢ - Uⁿ`` of the stage just completed into the accumulators of the
stages after it (after stage 1, keep it raw for stage 2).
"""
function store_stage_increment!(ts::MultirateInfinitesimalStep, model, i, Δt)
    (; α, γ, d) = ts.tableau
    s = length(d)
    grid = model.grid
    arch = architecture(grid)
    ks = later_stages(max(i, 2), s)

    for (name, u) in pairs(prognostic_fields(model))
        Uⁿ = parent(ts.Uⁿ[name])
        size_u = (length(Uⁿ),)
        if i == 1
            launch!(arch, grid, size_u, _store_first_increment!,
                    parent(ts.accumulators[1].offset[name]), parent(u), Uⁿ)
        else
            launch!(arch, grid, size_u, _accumulate_increment!,
                    accumulator_arrays(ts, name, ks, :forcing), accumulator_arrays(ts, name, ks, :offset),
                    parent(u), Uⁿ, increment_weights(γ, d, Δt, i, ks), offset_weights(α, i, ks))
        end
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

    ts = model.timestepper
    ts.current_Δt[] = Δt
    ts.next_stage[] = 1
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    (; c, d) = ts.tableau
    s = length(d)
    tⁿ⁺¹ = model.clock.time + Δt

    store_outer_step_state!(ts, model)
    freeze_linearization_state!(ts.substepper, model)

    for i in 1:s
        ts.stage[] = i
        acoustic_rk3_substep!(model, Δt, d[i])

        if i < s
            store_stage_increment!(ts, model, i, Δt)
            ts.next_stage[] = i + 1
            Δc = i == 1 ? c[1] : c[i] - c[i-1]
            tick_stage!(model.clock, Δc * Δt)
            update_state!(model, callbacks; compute_tendencies = true)
            cache_transport_velocity!(model)
        end
    end

    ts.next_stage[] = 1
    complete_acoustic_time_step!(model, Δt, tⁿ⁺¹, callbacks)

    return nothing
end
