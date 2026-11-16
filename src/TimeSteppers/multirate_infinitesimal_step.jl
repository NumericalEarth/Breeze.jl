using Breeze.AtmosphereModels: thermodynamic_density_name, dynamics_prognostic_fields, SplitScalarTendencies
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!,
                                    StageEntryForceRemainder, stage_entry_force_remainder,
                                    step_start_force_state, store_step_start_force_state!,
                                    compute_pressure_remainder!,
                                    CompressibleEquations, ForwardBackwardSubsteps, SymmetricSubsteps,
                                    with_residual
using Breeze.RoundingTreatments: rounding_error

using Oceananigans: prognostic_fields, CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Grids: topology, Periodic, Flat
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
evaluated at ``Yⱼ₋₁``. The fast tendency is linearized once per step, about ``Uⁿ``.

The slow tendency excludes everything the acoustic loop integrates: momentum tendencies never
contain the pressure gradient or buoyancy; the continuity tendency ``-∇·(ρu)`` is entirely fast;
and the thermodynamic slow tendency is the full tendency minus ``-∇·(θ (ρu))`` in the loop's
discretization (see `compute_linearized_thermodynamic_flux_divergence!`). With
the `:WickerSkamarock3` coefficients the scheme reduces to `AcousticRungeKutta3`.

Fields
======

- `tableau`: the [`MultirateInfinitesimalStepTableau`](@ref).
- `U⁰`: what the start state of stages 2 and 3 is formed from: the increment ``ΔY₁`` during stage 2 and
  the offset ``Σⱼ α₃ⱼ ΔYⱼ`` during stage 3. No start state is stored: stage 1 starts from `Uⁿ` itself
  and a later stage from `Uⁿ` plus its offset, summed where it is read (see `stage_start_state`).
- `Gⁿ`: tendency storage; holds the combined slow forcing during each acoustic loop.
- `implicit_solver`, `substepper`: as for `AcousticRungeKutta3`.
- `rounding`: Round-off treatment of the state updates that reach ``Uⁿ⁺¹``, [`RoundToNearest`](@ref)
  or [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding).
- `Uⁿ`: the outer-step-start state.
- `accumulators`: for each stage ``k = 3, …, s``, the parts of its forcing and start state known so far,
  `forcing` ``= (1/dₖ) Σⱼ (βₖⱼ Sⱼ + γₖⱼ ΔYⱼ / Δt)`` and `offset` ``= Σⱼ αₖⱼ ΔYⱼ``, accumulated as each
  ``Sⱼ`` and ``ΔYⱼ = Yⱼ - Uⁿ`` becomes available, so no ``Sⱼ`` and only ``ΔY₁`` is stored. Stage 3's
  forcing holds the raw ``S₁`` until stage 2 consumes it, and its offset, which is `U⁰`, holds the
  raw ``ΔY₁`` through stage 2, so the offsets of later stages are first written with ``ΔY₂``. Stage
  4's offset then shares the storage of stage 3's forcing for the fields that form their increments
  in the assembly (see `stage_accumulators`).
- `fast_tendency`: scratch for the fast parts of the tendencies: with `PreLoopScalarTendencies`
  one field, for the stage-entry fast thermodynamic tendency, sharing storage with the substepper; with `PostLoopScalarTendencies`
  one per prognostic the acoustic loop does not advance by continuity or momentum — the
  thermodynamic variable's, used as before the loop, and each scalar's, which the post-loop build
  fills with its advective tendency (see `fast_tendency_scratch`).
- `stage`: the stage being integrated.
- `next_stage`, `current_Δt`: the stage the next tendency evaluation feeds and the current outer step, which
  size the adaptive-implicit vertical-advection split (see `adaptive_advection_timestep`).
- `relinearize_stages`: re-linearize the fast operator at every stage entry (as `AcousticRungeKutta3` does)
  instead of once per step. Off by default: the slow tendencies of earlier stages were split against
  a different fast operator, so with a non-diagonal tableau the step is first-order accurate (the
  constructor warns). With the `:WickerSkamarock3` coefficients it reproduces `AcousticRungeKutta3`
  bitwise; it exists to diagnose robustness in strongly nonlinear flows.
- `step_start`: the outer-step-start pressure and water density that the
  slow remainder of the stage-entry pressure gradient and buoyancy is measured from, or `nothing`
  where that remainder is not recombined (see `step_start_storage`).
- `residual`: what the rounded stage states of the acoustic prognostics lack of the exact sums,
  kept from the end of a stage to the assembly of the next, in the substepper's perturbation fields
  where it can be (see `stage_residual_storage`), or `nothing` for a tableau that recombines no
  increments. What the rounded start states lack is evaluated where it is read
  (see `start_state_residual` in the acoustic substepper).
- `fold_stage_increments`: whether the acoustic prognostics form their increments in the assembly
  of the next stage (see `stage_increment_source`).
- `scalar_tendencies`: when the moisture and tracer tendencies are built,
  `PreLoopScalarTendencies` (between stages, the default) or
  `PostLoopScalarTendencies` (after each stage's own acoustic loop). With the latter, the
  scalars' slow tendency ``Sᵢ`` stored for stage ``i`` is the post-loop one, and their stage forcing
  and start state are assembled after the loop.
"""
struct MultirateInfinitesimalStep{TB, U0, TG, TI, AS, R, UN, AC, FC, ST, SS, TA, RS} <: AcousticTimeStepper
    relinearize_stages :: Bool
    fold_stage_increments :: Bool
    tableau :: TB
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
    rounding :: R
    Uⁿ :: UN
    accumulators :: AC
    fast_tendency :: FC
    scalar_tendencies :: ST
    step_start :: SS
    transport_density_accumulators :: TA # stages 2 to s: the parts known so far of the transport density tendency's stage combination; stage 2's shares storage with `U⁰.ρᵈ`
    residual :: RS # what the rounded stage states of the acoustic prognostics lack of the exact ones; `nothing` when no stage recombines increments
    stage :: Base.RefValue{Int}
    next_stage :: Base.RefValue{Int}
    current_Δt :: Base.RefValue{Float64}
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`MultirateInfinitesimalStep`](@ref) time stepper. `coefficients` is a `Symbol`
naming a set in [`multirate_infinitesimal_step_tableau`](@ref) (default `:SSPMIS43`) or a
[`MultirateInfinitesimalStepTableau`](@ref).

Select it with `AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep,
timestepper_kwargs = (; coefficients = :SSPMIS43))`.

The default, `:SSPMIS43`, is chosen for positivity: its slow method is strong-stability preserving
with coefficient ``r = 1.79``, so the moisture and tracer stages are convex combinations of
forward-Euler steps for an advective Courant number up to ``1.79`` times the forward-Euler bound of
the advection operator. Of the other sets, `:TVDMISB` has ``r = 0.75``, and `:MIS4a` and
`:WickerSkamarock3` have ``r = 0``: no such bound at any Courant number. The bound is on the time stepping alone. Positive moisture also needs an
advection operator that is positive under forward Euler, which the clamp of bounded WENO
reconstructions does not provide by itself; a flux limiter would (Oceananigans.jl#6475).

`:SSPMIS43` has four stages and is second-order accurate with the fast and slow parts coupled. Its
first stage covers 0.56 of the step, which takes at least two acoustic substeps. It needs the
moisture's pressure effect in the acoustic loop to stay robust in moist convection: with the
stage-entry pressure remainder recombined across stages, the part of that remainder the loop does
not carry is integrated explicitly. `:MIS4a` is third order, without the positivity bound.

The stages of every set but `:WickerSkamarock3` start from and are forced with the increments of
earlier stages, so their acoustic substeps are symmetric in the horizontal momentum and the
pressure (see `SymmetricSubsteps`). Without divergence damping, `:TVDMISB` and `:MIS4a` then leave
the acoustic modes neutral at the default `acoustic_cfl = 0.5`, and `:SSPMIS43` does except for
horizontal acoustic Courant numbers ``c Δt / Δx`` between 1.5 and 1.8 (amplified by up to 1.1 per
step) and between 3.3 and 3.6 (1.02), which `acoustic_cfl = 0.4` or the default damping removes.

`post_loop_scalar_tendencies = true` rebuilds the moisture and tracer tendencies after each
stage's own acoustic loop, with that loop's time-averaged transport velocity, instead of
between stages with the preceding loop's (see `PostLoopScalarTendencies`).

`Gⁿ` and `U⁰` are the tendency storage and the storage the first start states are formed from,
allocated by default. Passing them shares the storage of another time stepper, as the
adiabatic-balance twin does (see `adiabatic_balance_twin`).

`rounding` sets the round-off treatment of the state updates that reach ``Uⁿ⁺¹`` (default:
[`RoundToNearest`](@ref)). For the acoustic prognostics that is the last stage's recovery of the
state from its stage-entry value plus the accumulated perturbation, alone: the earlier stages'
recoveries and the start states ``U⁰ = Uⁿ + Σⱼ αᵢⱼ ΔYⱼ`` round to nearest and keep their rounding
errors, which are added back to the increments ``ΔYⱼ`` and to the perturbation the next loop
starts from, so that no intermediate sum loses anything or perturbs the step. For the moisture,
microphysical and tracer fields, [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding)
rounds the update ``u⁰ + Δt G`` and the start state stochastically in the last stage and in every
stage whose increment a later stage reuses: what round to nearest loses from such an increment
is lost from ``Uⁿ⁺¹`` as well. Stages that enter later ones through their tendency alone (the
first two of `:WickerSkamarock3`), the perturbations themselves, and the vertically-implicit
solves stay round-to-nearest.
"""
function MultirateInfinitesimalStep(grid, prognostic_fields;
                                    dynamics,
                                    coefficients = :SSPMIS43,
                                    implicit_solver = nothing,
                                    cache_advecting_state = false,
                                    relinearize_stages = false,
                                    post_loop_scalar_tendencies = false,
                                    Gⁿ = map(similar, prognostic_fields),
                                    U⁰ = map(similar, prognostic_fields),
                                    rounding = RoundToNearest())

    FT = eltype(grid)
    rounding = materialize_rounding(rounding, grid.architecture)
    tableau = multirate_infinitesimal_step_tableau(coefficients, FT)
    s = number_of_stages(tableau)
    s ≥ 3 || throw(ArgumentError("MultirateInfinitesimalStep needs at least three stages; got $s."))

    substepper = build_acoustic_substepper(grid, dynamics, prognostic_fields; cache_advecting_state)

    Uⁿ = map(similar, prognostic_fields)
    fold = fold_stage_increments[]
    folded_names = fold ? (keys(dynamics_prognostic_fields(dynamics))..., :ρu, :ρv, :ρw) : ()
    accumulators = stage_accumulators(prognostic_fields, U⁰, folded_names, s)
    scalar_tendencies = scalar_tendency_timing(post_loop_scalar_tendencies)
    fast_tendency = fast_tendency_storage(scalar_tendencies, grid, prognostic_fields, dynamics, substepper)

    step_start = step_start_storage(grid, substepper, tableau, relinearize_stages)

    if relinearize_stages && !is_diagonal(tableau.β)
        @warn "relinearize_stages = true with a non-diagonal tableau is first-order accurate in time: " *
              "the slow tendencies of earlier stages were split against a different fast operator."
    end

    residual = recombines_stage_increments(tableau) ? stage_residual_storage(grid, dynamics, substepper) : nothing

    return MultirateInfinitesimalStep(relinearize_stages, fold, tableau, U⁰, Gⁿ, implicit_solver, substepper,
                                      rounding, Uⁿ, accumulators, fast_tendency, scalar_tendencies, step_start,
                                      transport_density_accumulator_storage(grid, U⁰, fold, s), residual,
                                      Ref(1), Ref(1), Ref(NaN))
end

# The accumulators of stages 3 to s. Stage 3's offset is `U⁰`. A field that forms its increments
# in the assembly (`folded_names`) reads stage 3's forcing for the last time in the kernel that
# first writes stage 4's offset, so the two share storage. The thermodynamic density also forms its
# increments in the assembly, but it is not told apart from the scalars here, where only the
# dynamics is known, so its stage-4 offset has storage of its own.
function stage_accumulators(prognostic_fields, U⁰, folded_names, s)
    names = keys(prognostic_fields)
    forcing = ntuple(_ -> map(similar, prognostic_fields), s - 2)

    return ntuple(s - 2) do m
        offset = if m == 1
            U⁰
        elseif m == 2
            NamedTuple{names}(map(name -> name ∈ folded_names ? forcing[1][name] : similar(prognostic_fields[name]), names))
        else
            map(similar, prognostic_fields)
        end
        (; forcing = forcing[m], offset)
    end
end

# The accumulators of the transport density tendency of stages 2 to s. Stage 2's is written when
# stage 1 is assembled and read when stage 2 is, just before the dry density's assembly keeps ΔY₁
# in `U⁰`, which nothing uses until then when the dry density forms its increments in the
# assembly. So the two share storage.
function transport_density_accumulator_storage(grid, U⁰, fold, s)
    second = fold ? U⁰.ρᵈ : CenterField(grid)
    return (second, ntuple(_ -> CenterField(grid), s - 2)...)
end

# One field per acoustic prognostic for the rounding errors of its stage states (see
# `start_state_residual` in the acoustic substepper), with default boundary conditions: the
# errors have no boundary values of their own. The thermodynamic density is a cell-centered field.
#
# The errors are written by the kernel that recovers the state from the loop's perturbations,
# point by point after it has read them, and are read when the next stage is assembled, before the
# next loop rewrites the perturbations. So they are kept in the perturbation fields themselves,
# where that kernel covers every point the assembly reads and reads no perturbation off its own
# point (`residuals_share_perturbation_storage`).
function stage_residual_storage(grid, dynamics, substepper)
    shared = residuals_share_perturbation_storage(grid, dynamics)
    storage(perturbation, field) = shared && eltype(perturbation) === eltype(grid) ? perturbation : field(grid)
    momentum = substepper.momentum_perturbation

    return (ρ = storage(substepper.density_perturbation, CenterField),
            ρᵡ = storage(substepper.density_potential_temperature_perturbation, CenterField),
            ρu = storage(momentum.u, XFaceField),
            ρv = storage(momentum.v, YFaceField),
            ρw = storage(momentum.w, ZFaceField))
end

# The recovery covers the cells of the grid. In a bounded horizontal direction the assembly also
# reads the face on the far boundary, where the momentum perturbation of an open boundary is not
# a rounding error; and over terrain the recovery of the vertical momentum reads the horizontal
# momentum perturbations of neighboring points, and the assembly reads the halos. The top face of
# the vertical momentum perturbation is held at zero, as a residual never written is.
residuals_share_perturbation_storage(grid, dynamics) =
    all(T -> T === Periodic || T === Flat, topology(grid)[1:2])

residuals_share_perturbation_storage(grid, ::TerrainCompressibleDynamics) = false

# The residual field of prognostic `name`, or `nothing` for a field that carries none: the moisture,
# microphysical and tracer fields, and every field of a tableau that recombines no increments.
stage_residual(::Nothing, model, name) = nothing

function stage_residual(residual, model, name)
    name === :ρᵈ && return residual.ρ
    name === thermodynamic_density_name(model.formulation) && return residual.ρᵡ
    return get(residual, name, nothing)
end

# Stage 1 starts from Uⁿ itself, and nothing recombines the last stage's result within the step.
# The rounding error of a later start state Uⁿ + zᵢ is evaluated from the two terms where the
# loop reads it.
function CompressibleEquations.start_state_residual(ts::MultirateInfinitesimalStep, model)
    (ts.stage[] == 1 || isnothing(ts.residual)) && return nothing
    offsets, weight = stage_offsets(ts)
    θ_name = thermodynamic_density_name(model.formulation)
    names = (:ρᵈ, θ_name, :ρu, :ρv, :ρw)
    residuals = map(name -> StartStateResidual(ts.Uⁿ[name], offsets[name], weight), names)
    return NamedTuple{(:ρ, :ρᵡ, :ρu, :ρv, :ρw)}(residuals)
end

CompressibleEquations.recovered_state_residual(ts::MultirateInfinitesimalStep) =
    ts.stage[] == number_of_stages(ts.tableau) ? nothing : ts.residual

# Scratch for the fast parts of the tendencies (see the `fast_tendency` field). Post-loop
# scalars each need their own: their advective tendencies are all in hand when the
# microphysical sources and the `TendencyCallsite` callbacks complete the slow parts, before
# any scalar is assembled. The thermodynamic variable is not told apart from the scalars here,
# where only the dynamics is known, so it also gets one, which the stage uses before the loop.
#
# Pre-loop, the one scratch is written and read while a stage is assembled, so it shares the
# storage of the substepper's stage-end moisture content, which each acoustic loop writes when it
# starts and reads for the last time when it ends.
fast_tendency_storage(::PreLoopScalarTendencies, grid, prognostic_fields, dynamics, substepper) =
    substepper.stage_end_moisture_content

function fast_tendency_storage(::PostLoopScalarTendencies, grid, prognostic_fields, dynamics, substepper)
    excluded = (keys(dynamics_prognostic_fields(dynamics))..., :ρu, :ρv, :ρw)
    names = filter(name -> name ∉ excluded, keys(prognostic_fields))
    return NamedTuple{names}(ntuple(_ -> CenterField(grid), length(names)))
end

# The scratch for the fast part of `name`'s tendency.
fast_tendency_scratch(ts::MultirateInfinitesimalStep, name) = fast_tendency_scratch(ts.fast_tendency, name)
fast_tendency_scratch(scratch::NamedTuple, name) = scratch[name]
fast_tendency_scratch(scratch, name) = scratch

# MIS assumes one fast operator for the whole step: the linearization coefficients (θᴸ, the
# pressure coefficients) stay at the values `freeze_linearization_state!` computed from Uⁿ.
# Re-linearizing per stage makes the fast operator stage-dependent: the slow tendencies recombined
# from earlier stages were split against a different operator, which is first-order accurate
# unless the tableau is diagonal.
refresh_stage_linearization!(ts::MultirateInfinitesimalStep, substepper, model) =
    ts.relinearize_stages ? prepare_acoustic_cache!(substepper, model) : nothing

# The fast interval dᵢ Δt of the stage that the tendencies about to be built will drive. Tendencies
# built at the end of a step feed stage 1 of the next, whose Δt is not yet known; the current one is used.
@inline function Oceananigans.Advection.adaptive_advection_timestep(ts::MultirateInfinitesimalStep, clock)
    Δt = ifelse(isfinite(ts.current_Δt[]), ts.current_Δt[], clock.last_Δt)
    return ts.tableau.d[ts.next_stage[]] * Δt
end

# Whether a stage starts from, or is forced with, the increments of earlier stages.
recombines_stage_increments(tableau) = any(!iszero, tableau.α) || any(!iszero, tableau.γ)

# Stages that recombine earlier increments need the acoustic substeps to return the state they
# were given without a skew between momentum and pressure (see `SymmetricSubsteps`). The others
# (`:WickerSkamarock3`) keep the forward–backward substeps of `AcousticRungeKutta3`.
CompressibleEquations.horizontal_substep_form(ts::MultirateInfinitesimalStep) =
    recombines_stage_increments(ts.tableau) ? SymmetricSubsteps() : ForwardBackwardSubsteps()

# Whether the next `update_state!` builds the tendencies of the first stage of an outer step.
tendencies_feed_first_stage(timestepper) = true
tendencies_feed_first_stage(ts::MultirateInfinitesimalStep) = ts.next_stage[] == 1

# With the production model's coefficients (see the `AcousticRungeKutta3` method).
AtmosphereModels.adiabatic_twin_timestepper(ts::MultirateInfinitesimalStep, ::SplitExplicitCompressibleDynamics) =
    :MultirateInfinitesimalStep, (; coefficients = ts.tableau)

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
# weight one, from the stage-entry state) and the slow part MIS recombines across stages,
# given what `G` holds on entry to the assembly.
struct SlowTendencySplit end                  # momentum, pre-loop scalars: G is all slow
struct ContinuityTendencySplit end            # continuity: G is all fast
struct FastSlowTendencySplit{C}               # thermodynamic: G is the full tendency, whose fast
    fast_tendency :: C                        # part C = -∇·(θᴸ (ρu)ᴸ) is subtracted to get the slow
end
struct PartitionedTendencySplit{C}            # post-loop scalars: G holds the sources and C the advection
    advective_tendency :: C                   # -∇·(ρ ⟨𝐔⟩ c), built apart; their sum is the slow tendency
end
struct StageEntryForceSplit{R}                # momentum: G is slow, and so is the remainder R of the
    remainder :: R                            # stage-entry force that the loop adds at weight one
end

Adapt.adapt_structure(to, s::FastSlowTendencySplit) = FastSlowTendencySplit(adapt(to, s.fast_tendency))
Adapt.adapt_structure(to, s::PartitionedTendencySplit) = PartitionedTendencySplit(adapt(to, s.advective_tendency))
Adapt.adapt_structure(to, s::StageEntryForceSplit) = StageEntryForceSplit(adapt(to, s.remainder))

@inline slow_part(i, j, k, grid, ::SlowTendencySplit, G) = @inbounds G[i, j, k]
@inline slow_part(i, j, k, grid, ::ContinuityTendencySplit, G) = zero(eltype(G))
@inline slow_part(i, j, k, grid, s::FastSlowTendencySplit, G) = @inbounds G[i, j, k] - s.fast_tendency[i, j, k]
@inline slow_part(i, j, k, grid, s::PartitionedTendencySplit, G) = @inbounds G[i, j, k] + s.advective_tendency[i, j, k]
@inline slow_part(i, j, k, grid, s::StageEntryForceSplit, G) = @inbounds G[i, j, k] + stage_entry_force_remainder(i, j, k, grid, s.remainder)

@inline full_tendency(i, j, k, grid, split, G) = @inbounds G[i, j, k]
@inline full_tendency(i, j, k, grid, s::PartitionedTendencySplit, G) = @inbounds G[i, j, k] + s.advective_tendency[i, j, k]

tendency_split(name, θ_name, C) = name === :ρᵈ ? ContinuityTendencySplit() :
                                  name === θ_name ? FastSlowTendencySplit(C) : SlowTendencySplit()

# The momentum's slow part includes the stage-entry force remainder when the stepper holds the
# step-start state it is measured from (`ts.step_start`). Otherwise the momentum tendency alone is slow.
function stage_tendency_split(ts, model, name, θ_name, C)
    direction = name === :ρu ? Val(:x) : name === :ρv ? Val(:y) : name === :ρw ? Val(:z) : nothing
    isnothing(direction) && return tendency_split(name, θ_name, C)
    return momentum_tendency_split(ts.step_start, ts.substepper, model, direction)
end

momentum_tendency_split(::Nothing, substepper, model, direction) = SlowTendencySplit()
momentum_tendency_split(step_start, substepper, model, direction) =
    StageEntryForceSplit(StageEntryForceRemainder(direction, substepper, model, step_start))

# Whether only the diagonal of `β` is nonzero: every stage then applies its own slow tendency alone.
is_diagonal(β) = all(i == j || iszero(β[i, j]) for i in axes(β, 1), j in axes(β, 2))

# `nothing` when the stage-entry force remainder is not recombined: with a diagonal `β` or with
# `relinearize_stages`.
function step_start_storage(grid, substepper, tableau, relinearize_stages)
    recombined = !is_diagonal(tableau.β) && !relinearize_stages
    return recombined ? step_start_force_state(grid, substepper) : nothing
end

store_step_start_state!(::Nothing, model) = nothing
store_step_start_state!(step_start, model) = store_step_start_force_state!(step_start, model)

# Pointwise updates of tuples of arrays whose length is known at compile time, so they unroll.
@inline accumulate!(i, j, k, ::Tuple{}, ::Tuple{}, x) = nothing
@inline function accumulate!(i, j, k, A::Tuple, w::Tuple, x)
    @inbounds A[1][i, j, k] += w[1] * x
    return accumulate!(i, j, k, Base.tail(A), Base.tail(w), x)
end

# Two terms in one update of each array, added in the order given.
@inline accumulate!(i, j, k, ::Tuple{}, ::Tuple{}, x, ::Tuple{}, y) = nothing
@inline function accumulate!(i, j, k, A::Tuple, wx::Tuple, x, wy::Tuple, y)
    @inbounds A[1][i, j, k] = A[1][i, j, k] + wx[1] * x + wy[1] * y
    return accumulate!(i, j, k, Base.tail(A), Base.tail(wx), x, Base.tail(wy), y)
end

@inline combine!(i, j, k, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, a, b, c) = nothing
@inline function combine!(i, j, k, A::Tuple, wa::Tuple, wb::Tuple, wc::Tuple, a, b, c)
    @inbounds A[1][i, j, k] = wa[1] * a + wb[1] * b + wc[1] * c
    return combine!(i, j, k, Base.tail(A), Base.tail(wa), Base.tail(wb), Base.tail(wc), a, b, c)
end

@inline combine!(i, j, k, ::Tuple{}, ::Tuple{}, ::Tuple{}, a, b) = nothing
@inline function combine!(i, j, k, A::Tuple, wa::Tuple, wb::Tuple, a, b)
    @inbounds A[1][i, j, k] = wa[1] * a + wb[1] * b
    return combine!(i, j, k, Base.tail(A), Base.tail(wa), Base.tail(wb), a, b)
end

@inline assign!(i, j, k, ::Tuple{}, ::Tuple{}, x) = nothing
@inline function assign!(i, j, k, A::Tuple, w::Tuple, x)
    @inbounds A[1][i, j, k] = w[1] * x
    return assign!(i, j, k, Base.tail(A), Base.tail(w), x)
end

# The stage forcing is
#   G = (fast part at Yᵢ₋₁) + (1/dᵢ)[βᵢᵢ Sᵢ + Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt)] = Gᵢ + (βᵢᵢ/dᵢ - 1) Sᵢ + Fᵢ,
# with Gᵢ the full tendency (`full_tendency`), written so a diagonal tableau (Wicker–Skamarock)
# leaves Gᵢ bitwise intact, and the start state is U⁰ = Uⁿ + Zᵢ with Zᵢ = Σⱼ αᵢⱼ ΔYⱼ, which is not
# stored: the assembly leaves Zᵢ and the readers of the start state add it to Uⁿ (`StageStartState`). Each
# kernel makes one pass over a field. These kernels do no arithmetic to speak of: their cost is
# the memory they stream, so they cover only the points that are read afterwards
# (`stage_assembly_parameters`) and touch each array once.

# Stage 1: F₁ = Z₁ = 0, so the start state is Uⁿ itself (`stage_start_state`). Keep S₁ (in stage
# 3's forcing accumulator) for later stages.
@kernel function _first_multirate_stage!(G, grid, split, wᵢ, S₁)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        slow = slow_part(i, j, k, grid, split, G)
        S₁[i, j, k] = slow
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * slow
    end
end

# Where a stage's assembly finds the increment ΔYᵢ₋₁ = Yᵢ₋₁ - Uⁿ of the stage before
# (`stage_increment_source`): formed on the spot from the prognostic field `u`, which the assembly
# of the acoustic prognostics can do because nothing changes them between the end of a stage and
# the next assembly, or already stored in the accumulators by `store_stage_increment!`.
struct IncrementFromState{U}
    u :: U
end

struct StoredIncrement end

Adapt.adapt_structure(to, source::IncrementFromState) = IncrementFromState(adapt(to, source.u))

# The increment ΔYᵢ₋₁ = Yᵢ₋₁ - Uⁿ formed from the state, with the rounding error of the recovery
# of Yᵢ₋₁ where it was kept (`residual`): the difference of the two states is exact, so the
# increment is the exact one to its own precision. A stored increment is read where it is needed.
@inline state_increment(source::IncrementFromState, residual, uⁿ, i, j, k) =
    with_residual(@inbounds(source.u[i, j, k] - uⁿ), residual, i, j, k)

@inline state_increment(::StoredIncrement, residual, uⁿ, i, j, k) = zero(uⁿ)

# ΔY₁, for stage 2: `store_stage_increment!` has kept it in `ΔY₁`, where an increment formed
# from the state is kept as well, for the start state of stage 2 and the offsets of later stages.
@inline function first_increment!(::IncrementFromState, ΔY₁, ΔY, i, j, k)
    @inbounds ΔY₁[i, j, k] = ΔY
    return ΔY
end

@inline first_increment!(::StoredIncrement, ΔY₁, ΔY, i, j, k) = @inbounds ΔY₁[i, j, k]

# Stage 3: the forcing accumulated so far, completed with ΔY₂ in registers when it has not been
# folded in already.
@inline completed_forcing(::IncrementFromState, Fᵢ, wγᵢ, ΔY, i, j, k) = @inbounds Fᵢ[i, j, k] + wγᵢ * ΔY
@inline completed_forcing(::StoredIncrement, Fᵢ, wγᵢ, ΔY, i, j, k) = @inbounds Fᵢ[i, j, k]

# Stage 3, for a field that forms its increments in the assembly: turn the raw ΔY₁ in `Z₃` into
# the offset of stage 3, and write the offsets of later stages for the first time. A field whose
# increments are stored did both when stage 2 ended (`_accumulate_second_increment!`).
@inline function write_offsets!(::IncrementFromState, Z₃, Z, α₃₁, α₃₂, α₁ᵏ, α₂ᵏ, ΔY₂, i, j, k)
    @inbounds begin
        ΔY₁ = Z₃[i, j, k]
        Z₃[i, j, k] = α₃₁ * ΔY₁ + α₃₂ * ΔY₂
    end
    combine!(i, j, k, Z, α₁ᵏ, α₂ᵏ, ΔY₁, ΔY₂)
    return nothing
end

@inline write_offsets!(::StoredIncrement, Z₃, Z, α₃₁, α₃₂, α₁ᵏ, α₂ᵏ, ΔY₂, i, j, k) = nothing

# Stage i ≥ 4: its own accumulators, completed with ΔYᵢ₋₁ when it has not been folded in already.
# The completed offset is written back, for the start state to be formed from.
@inline function completed_accumulators!(::IncrementFromState, Fᵢ, Zᵢ, wγᵢ, αᵢ, ΔY, i, j, k)
    @inbounds begin
        forcing = Fᵢ[i, j, k] + wγᵢ * ΔY
        Zᵢ[i, j, k] = Zᵢ[i, j, k] + αᵢ * ΔY
    end
    return forcing
end

@inline completed_accumulators!(::StoredIncrement, Fᵢ, Zᵢ, wγᵢ, αᵢ, ΔY, i, j, k) = @inbounds Fᵢ[i, j, k]

# Add Sᵢ, and ΔYᵢ₋₁ when it has not been folded in already, to the accumulators of later stages.
@inline function accumulate_later_stages!(::IncrementFromState, F, Z, wγᵏ, wβᵏ, αᵏ, slow, ΔY, i, j, k)
    accumulate!(i, j, k, F, wγᵏ, ΔY, wβᵏ, slow)
    accumulate!(i, j, k, Z, αᵏ, ΔY)
    return nothing
end

@inline accumulate_later_stages!(::StoredIncrement, F, Z, wγᵏ, wβᵏ, αᵏ, slow, ΔY, i, j, k) =
    accumulate!(i, j, k, F, wβᵏ, slow)

# Stage 2 reads the raw S₁ and turns it, with ΔY₁ and S₂, into the forcing accumulators of stages
# k ≥ 3 (whose first is where S₁ was kept: it is read before being overwritten). ΔY₁ stays as it
# is, in the storage of stage 3's offset: stage 2 starts from Uⁿ + α₂₁ ΔY₁, formed where it is read.
@kernel function _second_multirate_stage!(G, grid, Uⁿ, ΔY₁, source, split, wᵢ, wβ₁, wγ₁, F, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        S₁ = F[1][i, j, k]
        uⁿ = Uⁿ[i, j, k]
        ΔY = first_increment!(source, ΔY₁, state_increment(source, residual, uⁿ, i, j, k), i, j, k)
        S₂ = slow_part(i, j, k, grid, split, G)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * S₂ + wβ₁ * S₁ + wγ₁ * ΔY
        combine!(i, j, k, F, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, S₁, ΔY, S₂)
    end
end

# Stage 3 uses its completed forcing and adds S₃ to those of later stages. `Z₃` holds ΔY₁ on entry
# and the offset of stage 3 on exit, and the offsets `Z` of later stages are written for the first
# time, unless the field's increments are stored. `Z[1]` may be the array `F₃`, which is read first.
@kernel function _third_multirate_stage!(G, grid, Uⁿ, source, split, wᵢ, F₃, Z₃, wγ₃, α₃₁, α₃₂, F, Z, wγᵏ, wβᵏ, α₁ᵏ, α₂ᵏ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        uⁿ = Uⁿ[i, j, k]
        ΔY₂ = state_increment(source, residual, uⁿ, i, j, k)
        slow = slow_part(i, j, k, grid, split, G)
        forcing = completed_forcing(source, F₃, wγ₃, ΔY₂, i, j, k)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * slow + forcing
        write_offsets!(source, Z₃, Z, α₃₁, α₃₂, α₁ᵏ, α₂ᵏ, ΔY₂, i, j, k)
        accumulate_forcing!(source, F, wγᵏ, wβᵏ, slow, ΔY₂, i, j, k)
    end
end

@inline accumulate_forcing!(::IncrementFromState, F, wγᵏ, wβᵏ, slow, ΔY, i, j, k) = accumulate!(i, j, k, F, wγᵏ, ΔY, wβᵏ, slow)
@inline accumulate_forcing!(::StoredIncrement, F, wγᵏ, wβᵏ, slow, ΔY, i, j, k) = accumulate!(i, j, k, F, wβᵏ, slow)

# Stage i ≥ 4 uses its completed accumulators and adds Sᵢ to those of later stages.
@kernel function _later_multirate_stage!(G, grid, Uⁿ, source, split, wᵢ, Fᵢ, Zᵢ, wγᵢ, αᵢ, F, Z, wγᵏ, wβᵏ, αᵏ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        uⁿ = Uⁿ[i, j, k]
        ΔY = state_increment(source, residual, uⁿ, i, j, k)
        slow = slow_part(i, j, k, grid, split, G)
        forcing = completed_accumulators!(source, Fᵢ, Zᵢ, wγᵢ, αᵢ, ΔY, i, j, k)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * slow + forcing
        accumulate_later_stages!(source, F, Z, wγᵏ, wβᵏ, αᵏ, slow, ΔY, i, j, k)
    end
end

# After stage i, for the fields whose increment is stored: ΔYᵢ = Yᵢ - Uⁿ, kept raw after stage 1,
# combined with ΔY₁ into the offsets after stage 2, and accumulated after later stages.
@kernel function _store_first_increment!(ΔY₁, u, Uⁿ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔY₁[i, j, k] = with_residual(u[i, j, k] - Uⁿ[i, j, k], residual, i, j, k)
end

@kernel function _accumulate_second_increment!(F, Z₃, Z, u, Uⁿ, wγᵏ, α₃₁, α₃₂, α₁ᵏ, α₂ᵏ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ΔY₁ = Z₃[i, j, k]
        ΔY₂ = with_residual(u[i, j, k] - Uⁿ[i, j, k], residual, i, j, k)
        accumulate!(i, j, k, F, wγᵏ, ΔY₂)
        Z₃[i, j, k] = α₃₁ * ΔY₁ + α₃₂ * ΔY₂
        combine!(i, j, k, Z, α₁ᵏ, α₂ᵏ, ΔY₁, ΔY₂)
    end
end

@kernel function _accumulate_increment!(F, Z, u, Uⁿ, wγᵏ, αᵏ, residual)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ΔY = with_residual(u[i, j, k] - Uⁿ[i, j, k], residual, i, j, k)
        accumulate!(i, j, k, F, wγᵏ, ΔY)
        accumulate!(i, j, k, Z, αᵏ, ΔY)
    end
end

@kernel function _store_outer_step_state!(Uⁿ, u)
    i, j, k = @index(Global, NTuple)
    @inbounds Uⁿ[i, j, k] = u[i, j, k]
end

# The interior, boundary faces included: no kernel reads the halos of the fields assembled. The
# terrain-following acoustic loop does read them, so the whole array is covered there.
stage_assembly_parameters(model, field) = KernelParameters(size(field), (0, 0, 0))

stage_assembly_parameters(::TerrainCompressibleAcousticModel, field) =
    KernelParameters(axes(field.data)...)

# The start state U⁰ = Uⁿ + w Z of one field in stages 2 to s, summed where it is read rather than
# stored: it is read a few times per stage, when the acoustic loop starts and ends and by the
# scalar update (and on open boundaries in each substep). `Z` and `w` are the stage's offset and
# its weight (`stage_offsets`). The sum adds the small `w Z` to the state, as a stage update does,
# so it takes a round-off treatment (see `start_state_rounding`); the random numbers of a
# stochastic one are a function of the indices and the key, so every read returns the same value.
struct StageStartState{T, U, Z, R} <: AbstractArray{T, 3}
    Uⁿ :: U
    offset :: Z
    weight :: T
    rounding :: R
    key :: UInt32
end

Adapt.adapt_structure(to, s::StageStartState) =
    StageStartState(adapt(to, s.Uⁿ), adapt(to, s.offset), s.weight, s.rounding, s.key)

Base.size(s::StageStartState) = size(s.Uⁿ)

Base.@propagate_inbounds Base.getindex(s::StageStartState, i::Int, j::Int, k::Int) =
    rounded_add(s.rounding, s.Uⁿ[i, j, k], s.weight * s.offset[i, j, k], i, j, k, s.key)

# What the round-to-nearest start state Uⁿ + w Z lacks of the exact sum, where the acoustic loop
# reads it (see `start_state_residual`).
struct StartStateResidual{T, U, Z} <: AbstractArray{T, 3}
    Uⁿ :: U
    offset :: Z
    weight :: T
end

Adapt.adapt_structure(to, r::StartStateResidual) =
    StartStateResidual(adapt(to, r.Uⁿ), adapt(to, r.offset), r.weight)

Base.size(r::StartStateResidual) = size(r.Uⁿ)

Base.@propagate_inbounds function Base.getindex(r::StartStateResidual, i::Int, j::Int, k::Int)
    uⁿ = r.Uⁿ[i, j, k]
    offset = r.weight * r.offset[i, j, k]
    return rounding_error(uⁿ, offset, uⁿ + offset)
end

# Where the offset Zᵢ = Σⱼ αᵢⱼ ΔYⱼ of the current stage i ≥ 2 is, and the weight it takes: stage 2
# starts from Uⁿ + α₂₁ ΔY₁, with the raw ΔY₁ in the storage of stage 3's offset, and stage i ≥ 3
# from Uⁿ + Zᵢ.
function stage_offsets(ts::MultirateInfinitesimalStep)
    i = ts.stage[]
    weight = i == 2 ? ts.tableau.α[2, 1] : one(eltype(ts.tableau.d))
    return ts.accumulators[max(i - 2, 1)].offset, weight
end

# A field that carries a residual rounds its start state to nearest, and the loop adds the rounding
# error to the perturbation it starts from. The others round theirs with the time stepper's
# round-off treatment.
start_state_rounding(ts, ::Nothing) = ts.rounding
start_state_rounding(ts, residual) = RoundToNearest()

# `Uⁿ` in stage 1 and `U⁰ = Uⁿ + Zᵢ` after that, formed where it is read. Both are read-only
# during a stage.
function stage_start_state(ts::MultirateInfinitesimalStep, model)
    ts.stage[] == 1 && return ts.Uⁿ
    offsets, weight = stage_offsets(ts)
    names = keys(ts.Uⁿ)

    start_states = map(names) do name
        rounding = start_state_rounding(ts, stage_residual(ts.residual, model, name))
        key = start_state_rounding_key(ts, model, name)
        StageStartState(ts.Uⁿ[name], offsets[name], weight, rounding, key)
    end

    return NamedTuple{names}(start_states)
end

# Weights of stage j's contributions to the stages k = i+1, …, s that still need them.
later_stages(i, s) = ntuple(m -> i + m, s - i)
forcing_weights(β, d, j, ks) = map(k -> β[k, j] / d[k], ks)
increment_weights(γ, d, Δt, j, ks) = map(k -> γ[k, j] / (d[k] * Δt), ks)
offset_weights(α, j, ks) = map(k -> α[k, j], ks)

# Accumulators of stages ks (k ≥ 3) of field `name`.
accumulator_fields(ts, name, ks, part) = map(k -> getproperty(ts.accumulators[k - 2], part)[name], ks)

# Assemble the start state and slow forcing of the current stage. On entry `Gⁿ` holds the full
# tendency at Yᵢ₋₁. With `PostLoopScalarTendencies` the scalars are assembled after the loop instead.
function assemble_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt)
    # The stage weights divide by Δt, so Δt takes the grid's float type: a Float64 weight in a
    # Float32 kernel fails to compile on GPUs without Float64 (Metal).
    Δt = convert(eltype(model.grid), Δt)
    # The fast thermodynamic tendency is linearized about the stage-entry state, so it must be
    # evaluated here, before the loop.
    θ_name = thermodynamic_density_name(model.formulation)
    C = fast_tendency_scratch(ts, θ_name)
    compute_linearized_thermodynamic_flux_divergence!(C, ts.substepper, model)
    compute_pressure_remainder!(ts.step_start, ts.substepper, model, ts.Uⁿ)
    assemble_transport_density_tendency!(ts, model, Δt)
    for name in pre_loop_assembly_names(ts.scalar_tendencies, model)
        assemble_stage_forcing!(ts, model, Δt, name, stage_tendency_split(ts, model, name, θ_name, C))
    end
    return nothing
end

# Give the transport density tendency the stage combination that the moisture forcing has:
# (1/dᵢ) [Σⱼ βᵢⱼ Gᵗⱼ + Σⱼ γᵢⱼ ΔYⱼ / Δt], with ΔYⱼ the increments of the dry density.
function assemble_transport_density_tendency!(ts::MultirateInfinitesimalStep, model, Δt)
    i = ts.stage[]
    (; β, γ, d) = ts.tableau
    s = length(d)
    grid = model.grid
    ks = later_stages(i, s)
    T = ts.transport_density_accumulators
    Tᵢ = i == 1 ? nothing : T[i - 1]
    wγᵢ = i == 1 ? zero(eltype(d)) : γ[i, i - 1] / (d[i] * Δt)
    wγᵏ = i == 1 ? map(k -> zero(eltype(d)), ks) : increment_weights(γ, d, Δt, i - 1, ks)
    launch!(architecture(grid), grid, :xyz, _assemble_transport_density_tendency!,
            ts.substepper.transport_density_tendency, Tᵢ, map(k -> T[k - 1], ks),
            model.dynamics.dry_density, ts.Uⁿ.ρᵈ, β[i, i] / d[i], wγᵢ, wγᵏ, forcing_weights(β, d, i, ks))
    return nothing
end

@inline stage_accumulation(::Nothing, i, j, k) = false
@inline stage_accumulation(Tᵢ, i, j, k) = @inbounds Tᵢ[i, j, k]

@inline start_accumulators!(i, j, k, ::Nothing, T, wγᵏ, ΔY, wβᵏ, G) = assign!(i, j, k, T, wβᵏ, G)
@inline start_accumulators!(i, j, k, Tᵢ, T, wγᵏ, ΔY, wβᵏ, G) = accumulate!(i, j, k, T, wγᵏ, ΔY, wβᵏ, G)

@kernel function _assemble_transport_density_tendency!(Gᵗ, Tᵢ, T, ρᵈ, ρᵈⁿ, wβᵢ, wγᵢ, wγᵏ, wβᵏ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        G = Gᵗ[i, j, k]
        ΔY = ρᵈ[i, j, k] - ρᵈⁿ[i, j, k]
        Gᵗ[i, j, k] = stage_accumulation(Tᵢ, i, j, k) + wγᵢ * ΔY + wβᵢ * G
        start_accumulators!(i, j, k, Tᵢ, T, wγᵏ, ΔY, wβᵏ, G)
    end
end

# The start state U⁰ = Uⁿ + Zᵢ adds the small Zᵢ to the state, as a stage update does, so it takes
# the time stepper's round-off treatment. Its field numbers follow those of the stage updates
# (`scalar_substep!`), so the two draw different random numbers in the same stage.
function start_state_rounding_key(ts::MultirateInfinitesimalStep, model, name)
    names = keys(prognostic_fields(model))
    field_number = length(names) + findfirst(==(name), names)
    return rounding_key(ts.rounding, model.clock.iteration, model.clock.stage, field_number)
end

# Whether a later stage reuses the increment ΔYᵢ = Yᵢ - Uⁿ of stage i.
reuses_stage_increment(tableau, i) =
    any(!iszero(tableau.α[k, i]) || !iszero(tableau.γ[k, i]) for k in i+1:number_of_stages(tableau))

# The round-off treatment of stage i's updates of the moisture, microphysical and tracer fields:
# the time stepper's for the last stage, whose result is Uⁿ⁺¹, and for every stage whose increment
# a later stage reuses, since what round to nearest loses from that increment is lost from Uⁿ⁺¹
# too. A stage that later stages use only through its tendency (every stage but the last of
# Wicker–Skamarock RK3) rounds to nearest.
function stage_rounding(ts::MultirateInfinitesimalStep, i)
    s = number_of_stages(ts.tableau)
    rounded = i == s || reuses_stage_increment(ts.tableau, i)
    return rounded ? ts.rounding : RoundToNearest()
end

# The acoustic prognostics keep the rounding errors of their intermediate stage states instead
# (`residual`), so the loop rounds to nearest in every stage but the last.
acoustic_loop_rounding(ts::MultirateInfinitesimalStep, rounding) =
    ts.stage[] == number_of_stages(ts.tableau) ? ts.rounding : RoundToNearest()

pre_loop_assembly_names(::PreLoopScalarTendencies, model) = keys(prognostic_fields(model))
pre_loop_assembly_names(::PostLoopScalarTendencies, model) = acoustic_prognostic_names(model)

# Assemble the scalars' stage forcing after the loop. Their whole tendency is slow: the
# non-advective part in `Gⁿ` plus the advective part in the scalar's `fast_tendency` scratch.
assemble_post_loop_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt) =
    assemble_post_loop_stage_forcing!(ts.scalar_tendencies, ts, model, convert(eltype(model.grid), Δt))

assemble_post_loop_stage_forcing!(::PreLoopScalarTendencies, ts, model, Δt) = nothing

function assemble_post_loop_stage_forcing!(::PostLoopScalarTendencies, ts, model, Δt)
    for name in scalar_prognostic_names(model)
        split = PartitionedTendencySplit(fast_tendency_scratch(ts, name))
        assemble_stage_forcing!(ts, model, Δt, name, split)
    end
    return nothing
end

# The post-loop build adds the non-advective part of each scalar's tendency to `Gⁿ` and writes the
# advective part to the scalar's `fast_tendency` scratch (see `assemble_post_loop_stage_forcing!`).
post_loop_scalar_tendency_kernel(ts::MultirateInfinitesimalStep) = SplitScalarTendencies(ts.fast_tendency)

# Assemble one field's start state and forcing, its tendency split into fast and slow parts by `split`.
function assemble_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt, name, split)
    i = ts.stage[]
    (; α, β, γ, d) = ts.tableau
    s = length(d)
    grid = model.grid
    arch = architecture(grid)

    wᵢ = β[i, i] / d[i] - 1
    ks = later_stages(max(i, 2), s)

    G = ts.Gⁿ[name]
    Uⁿ = ts.Uⁿ[name]
    source = stage_increment_source(model, name)
    F = accumulator_fields(ts, name, ks, :forcing)
    parameters = stage_assembly_parameters(model, G)
    residual = stage_residual(ts.residual, model, name)

    if i == 1
        launch!(arch, grid, parameters, _first_multirate_stage!, G, grid, split, wᵢ,
                ts.accumulators[1].forcing[name])
    elseif i == 2
        ΔY₁ = ts.accumulators[1].offset[name]
        launch!(arch, grid, parameters, _second_multirate_stage!, G, grid, Uⁿ, ΔY₁, source, split, wᵢ,
                β[2, 1] / d[2], γ[2, 1] / (d[2] * Δt), F,
                forcing_weights(β, d, 1, ks), increment_weights(γ, d, Δt, 1, ks),
                forcing_weights(β, d, 2, ks), residual)
    elseif i == 3
        F₃ = ts.accumulators[1].forcing[name]
        Z₃ = ts.accumulators[1].offset[name]
        Z = accumulator_fields(ts, name, ks, :offset)
        launch!(arch, grid, parameters, _third_multirate_stage!, G, grid, Uⁿ, source, split, wᵢ,
                F₃, Z₃, γ[3, 2] / (d[3] * Δt), α[3, 1], α[3, 2], F, Z,
                increment_weights(γ, d, Δt, 2, ks), forcing_weights(β, d, 3, ks),
                offset_weights(α, 1, ks), offset_weights(α, 2, ks), residual)
    else
        Fᵢ = ts.accumulators[i - 2].forcing[name]
        Zᵢ = ts.accumulators[i - 2].offset[name]
        Z = accumulator_fields(ts, name, ks, :offset)
        launch!(arch, grid, parameters, _later_multirate_stage!, G, grid, Uⁿ, source, split, wᵢ,
                Fᵢ, Zᵢ, γ[i, i - 1] / (d[i] * Δt), α[i, i - 1], F, Z,
                increment_weights(γ, d, Δt, i - 1, ks), forcing_weights(β, d, i, ks),
                offset_weights(α, i - 1, ks), residual)
    end

    return nothing
end

# Form the increments of the acoustic prognostics in the next stage's assembly, which reads `Uⁿ`
# anyway, instead of storing them when a stage completes. Off, every field's increment is stored.
# A time stepper reads the switch when it is constructed: its storage depends on it
# (`stage_accumulators`).
const fold_stage_increments = Ref(true)

# The acoustic prognostics still hold Yᵢ₋₁ when the next stage is assembled, so the assembly forms
# their increment. `update_state!` changes the other scalars in between, so theirs is stored.
function stage_increment_source(model, name)
    folded = model.timestepper.fold_stage_increments && name ∈ acoustic_prognostic_names(model)
    return folded ? IncrementFromState(prognostic_fields(model)[name]) : StoredIncrement()
end

# Fold the increment ΔYᵢ = Yᵢ - Uⁿ of the stage just completed into the later stages' accumulators.
function store_stage_increment!(ts::MultirateInfinitesimalStep, model, i, Δt)
    (; α, γ, d) = ts.tableau
    Δt = convert(eltype(model.grid), Δt)
    s = length(d)
    grid = model.grid
    arch = architecture(grid)
    ks = later_stages(max(i, 2), s)

    for (name, u) in pairs(prognostic_fields(model))
        stage_increment_source(model, name) isa StoredIncrement || continue
        Uⁿ = ts.Uⁿ[name]
        parameters = stage_assembly_parameters(model, u)
        residual = stage_residual(ts.residual, model, name)
        if i == 1
            launch!(arch, grid, parameters, _store_first_increment!,
                    ts.accumulators[1].offset[name], u, Uⁿ, residual)
        elseif i == 2
            # Stage 3's offset holds the raw ΔY₁, and the offsets of stages 4 to s are first written here.
            later = later_stages(3, s)
            launch!(arch, grid, parameters, _accumulate_second_increment!,
                    accumulator_fields(ts, name, ks, :forcing), ts.accumulators[1].offset[name],
                    accumulator_fields(ts, name, later, :offset), u, Uⁿ,
                    increment_weights(γ, d, Δt, 2, ks), α[3, 1], α[3, 2],
                    offset_weights(α, 1, later), offset_weights(α, 2, later), residual)
        else
            launch!(arch, grid, parameters, _accumulate_increment!,
                    accumulator_fields(ts, name, ks, :forcing), accumulator_fields(ts, name, ks, :offset),
                    u, Uⁿ, increment_weights(γ, d, Δt, i, ks), offset_weights(α, i, ks), residual)
        end
    end

    return nothing
end

# One kernel pass over the points the stages read, rather than a broadcast over the parent arrays.
function store_outer_step_state!(ts::MultirateInfinitesimalStep, model)
    grid = model.grid
    arch = architecture(grid)
    for (uⁿ, u) in zip(ts.Uⁿ, prognostic_fields(model))
        launch!(arch, grid, stage_assembly_parameters(model, u), _store_outer_step_state!, uⁿ, u)
    end
    return nothing
end

#####
##### Time stepping
#####

function OceananigansTimeSteppers.time_step!(model::MultirateAcousticModel, Δt; callbacks=[])

    ts = model.timestepper
    ts.current_Δt[] = Δt
    ts.next_stage[] = 1
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    (; c, d) = ts.tableau
    s = length(d)
    tⁿ⁺¹ = model.clock.time + Δt

    store_outer_step_state!(ts, model)
    store_step_start_state!(ts.step_start, model)
    freeze_linearization_state!(ts.substepper, model)

    for i in 1:s
        ts.stage[] = i
        acoustic_stage!(model, Δt, d[i], stage_rounding(ts, i), callbacks)

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
