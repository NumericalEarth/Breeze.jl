using Breeze.AtmosphereModels: thermodynamic_density_name, dynamics_prognostic_fields, SplitScalarTendencies,
                               moisture_prognostic_name, microphysical_velocities, nonadvective_scalar_tendency,
                               div_ρUc, advective_tendency_storage, launch_split_scalar_tendency!,
                               scalar_advective_tendency, specific_scalar_fields
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!,
                                    StageEntryForceRemainder, stage_entry_force_remainder,
                                    step_start_force_state, store_step_start_force_state!,
                                    compute_pressure_remainder!,
                                    CompressibleEquations, ForwardBackwardSubsteps, SymmetricSubsteps,
                                    with_residual
using Breeze.RoundingTreatments: rounding_error

using Oceananigans: prognostic_fields, CenterField, XFaceField, YFaceField, ZFaceField, architecture
using Oceananigans.Advection: explicit_velocity_scaleᶜᶜᶠ
using Oceananigans.Grids: topology, Periodic, Flat
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ
using Oceananigans.Utils: launch!, KernelParameters, sum_of_velocities, AdaptiveVerticallyImplicitDiscretization
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
- `storage`: the storage layout, chosen from the structure of the coefficients (see `stage_storage`):
  `TwoRegisterStorage` for a diagonal `β` with `α` and `γ` on the first subdiagonal alone
  (`:WickerSkamarock3`, `:LowStorageMIS52`), which keeps `Uⁿ` and `Gⁿ` and nothing else per
  prognostic, as `AcousticRungeKutta3` does; `IncrementStorage` for any other diagonal `β` with up to
  five stages (`:DiagonalMIS42`, `:DiagonalMIS52`), which adds two stage increments (three for the
  moisture, microphysical and tracer fields of a five-stage set); and `AccumulatorStorage` otherwise.
- `U⁰`: with `AccumulatorStorage`, what the start state of stages 2 and 3 is formed from: the increment
  ``ΔY₁`` during stage 2 and the offset ``Σⱼ α₃ⱼ ΔYⱼ`` during stage 3. With `IncrementStorage` it is the
  first increment register, and with `TwoRegisterStorage` it is `Uⁿ`. No start state is stored: stage 1
  starts from `Uⁿ` itself and a later stage from `Uⁿ` plus its offset, summed where it is read (see
  `stage_start_state`).
- `Gⁿ`: tendency storage; holds the combined slow forcing during each acoustic loop.
- `implicit_solver`, `substepper`: as for `AcousticRungeKutta3`.
- `rounding`: Round-off treatment of the state updates that reach ``Uⁿ⁺¹``, [`RoundToNearest`](@ref)
  or [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding).
- `Uⁿ`: the outer-step-start state.
- `accumulators`: with `TwoRegisterStorage`, empty. With `IncrementStorage`, the increment registers
  (see `increment_registers`). With `AccumulatorStorage`, for each stage ``k = 3, …, s``, the parts of
  its forcing and start state known so far,
  `forcing` ``= (1/dₖ) Σⱼ (βₖⱼ Sⱼ + γₖⱼ ΔYⱼ / Δt)`` and `offset` ``= Σⱼ αₖⱼ ΔYⱼ``, accumulated as each
  ``Sⱼ`` and ``ΔYⱼ = Yⱼ - Uⁿ`` becomes available, so no ``Sⱼ`` and only ``ΔY₁`` is stored. Stage 3's
  forcing holds the raw ``S₁`` until stage 2 consumes it, and its offset, which is `U⁰`, holds the
  raw ``ΔY₁`` through stage 2, so the offsets of later stages are first written with ``ΔY₂``. Stage
  4's offset then shares the storage of stage 3's forcing for the fields that form their increments
  in the assembly (see `stage_accumulators`).
- `fast_tendency`: scratch for the fast parts of the tendencies, or `nothing` for a coefficient set with a
  diagonal `β`, which applies every tendency whole: with `PreLoopScalarTendencies`
  one field, for the stage-entry fast thermodynamic tendency, sharing storage with the substepper; with `PostLoopScalarTendencies`
  one per moisture and condensate density, which the post-loop build fills with its advective
  tendency and the assembly leaves holding its slow tendency, for the next stage's loop (see
  `PredictedStageForcing`). The thermodynamic variable and the other scalars, whose tendencies no
  loop reads, use the substepper's field in turn (see `fast_tendency_scratch`).
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
- `transport_mass_fluxes`: the weights and mass fluxes of the correction that makes the
  recombined post-loop scalar fluxes carry each stage's own mass flux, or `nothing` where the
  stages' fluxes need no correction (see `ScalarMassFluxCorrection`).
- `scalar_tendencies`: when the moisture and tracer tendencies are built,
  `PostLoopScalarTendencies` (after each stage's own acoustic loop, the default) or
  `PreLoopScalarTendencies` (between stages). With the former, the
  scalars' slow tendency ``Sᵢ`` stored for stage ``i`` is the post-loop one, and their stage forcing
  and start state are assembled after the loop. For a coefficient set with a diagonal `β`, which has no
  `fast_tendency` scratch, it holds the fields in which the moisture and condensate tendencies
  are kept for the next stage's loop (see `previous_scalar_tendency`).
"""
struct MultirateInfinitesimalStep{TB, SL, U0, TG, TI, AS, R, UN, AC, FC, ST, SS, TA, RS, MF} <: AcousticTimeStepper
    relinearize_stages :: Bool
    fold_stage_increments :: Bool
    tableau :: TB
    storage :: SL # the storage layout: `TwoRegisterStorage`, `IncrementStorage` or `AccumulatorStorage`
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
    transport_density_accumulators :: TA # with `AccumulatorStorage`, stages 2 to s: the parts known so far of the transport density tendency's stage combination; stage 2's shares storage with `U⁰.ρᵈ`. Empty otherwise
    residual :: RS # what the rounded stage states of the acoustic prognostics lack of the exact ones; `nothing` when no stage recombines increments
    transport_mass_fluxes :: MF # the correction of the scalars' recombined mass fluxes; `nothing` when no stage corrects them
    stage :: Base.RefValue{Int}
    next_stage :: Base.RefValue{Int}
    current_Δt :: Base.RefValue{Float64}
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`MultirateInfinitesimalStep`](@ref) time stepper. `coefficients` is a `Symbol`
naming a set in [`multirate_infinitesimal_step_tableau`](@ref) (default `:ConvexLinearMIS6`) or a
[`MultirateInfinitesimalStepTableau`](@ref).

Select it with `AtmosphereModel(grid; dynamics, timestepper = :MultirateInfinitesimalStep,
timestepper_kwargs = (; coefficients = :ConvexLinearMIS6))`.

The default, `:ConvexLinearMIS6`, has six stages and is second-order accurate with the fast and
slow parts coupled. Its `β` is diagonal and each stage starts from and is forced with the
increment of the preceding stage alone, so it needs two registers besides the prognostic fields
(see `TwoRegisterStorage`), and its stage states are convex combinations of the outer-step-start
state and one forward-Euler step from the preceding stage. That makes the moisture and tracer
stages convex combinations of forward-Euler steps for an advective Courant number up to
``C = 1.77`` times the forward-Euler bound of the advection operator. The bound is on the time
stepping alone. Positive moisture also needs an advection operator that is positive under forward
Euler, which the clamp of bounded WENO reconstructions does not provide by itself; a flux limiter
would (Oceananigans.jl#6475). Its stages cover at most 0.37 of the step, and it is stable at about
twice the time step of `:SSPMIS43` and `:WickerSkamarock3` in moist convection.

Of the other sets, `:SSPMIS43` (four stages, ``C = 1.79``) recombines the slow tendencies of
earlier stages and needs more storage; `:TVDMISB` has ``C = 0.75``, and `:MIS4a` (third
order) and `:WickerSkamarock3` have ``C = 0``: no such bound at any Courant number.

The stages of every set but `:WickerSkamarock3` start from and are forced with the increments of
earlier stages, so their acoustic substeps are symmetric in the horizontal momentum and the
pressure (see `SymmetricSubsteps`).

By default (`post_loop_scalar_tendencies = true`) the moisture and tracer tendencies are built
after each stage's own acoustic loop, with that loop's time-averaged transport velocity (see
`PostLoopScalarTendencies`). Every stage then transports the scalars with the mass flux its own
loop applied to the dry air, so a uniform mass fraction stays uniform to round-off with every
coefficient set: the recombined tendencies of earlier stages carry those stages' mass fluxes, and
each stage transports the scalars with its own mass flux plus the one that makes up the difference
once the stages are recombined (see `ScalarMassFluxCorrection`). That needs a coefficient set whose
`β` is diagonal or has no zero on its diagonal.

The acoustic loop's pressure depends on the moisture and condensate tendencies, which are not
built yet when a stage's loop starts. The loop takes those that the stage before built in their
place (see `PredictedStageForcing`).

`post_loop_scalar_tendencies = false` builds the scalar tendencies between stages instead, with
the preceding loop's transport velocity, which does not keep a uniform mass fraction uniform. It
needs less storage: the post-loop tendencies take a field for each moisture and condensate
density, and, for a coefficient set whose `β` is not diagonal, three face fields for each stage
after the second.

`moisture_forcing_names` names the moisture and condensate densities, whose tendencies the loop's
pressure ramps with; `AtmosphereModel` provides them. With `nothing`, the time stepper cannot tell
them from the other scalars, and keeps a field for every prognostic that the loop does not advance
by continuity or momentum. The step is the same either way.

`Gⁿ` and `U⁰` are the tendency storage and the storage the first start states are formed from,
allocated by default. Passing them shares the storage of another time stepper, as the
adiabatic-balance twin does (see `adiabatic_balance_twin`).

The storage follows the structure of the coefficients. A set whose `β` is diagonal and whose `α`
and `γ` have entries on the first subdiagonal alone (`:WickerSkamarock3`, `:LowStorageMIS52`) keeps
two copies of the prognostic state, as `AcousticRungeKutta3` does: the step-start state `Uⁿ`, which
`U⁰` then is, and the tendencies, and with post-loop scalar tendencies the fields that keep the
moisture and condensate tendencies for the next stage's loop. Every stage forms the one increment it
needs from the prognostic fields themselves. Any other set with a
diagonal `β` and at most five stages (`:DiagonalMIS42`, `:DiagonalMIS52`) keeps two stage increments
besides, and a third for the moisture, microphysical and tracer fields of a five-stage set. The
remaining sets keep a forcing and an offset accumulator for each stage after the second. With every
layout the increment of a stage is that of the state `update_state!` leaves after it, so the
increments of the moisture and microphysical fields include its negative-moisture correction: a
stage starts from, and is forced with, the corrected state that its tendency was evaluated at.

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
                                    coefficients = :ConvexLinearMIS6,
                                    implicit_solver = nothing,
                                    cache_advecting_state = false,
                                    relinearize_stages = false,
                                    post_loop_scalar_tendencies = true,
                                    moisture_forcing_names = nothing,
                                    Gⁿ = map(similar, prognostic_fields),
                                    U⁰ = map(similar, prognostic_fields),
                                    rounding = RoundToNearest())

    FT = eltype(grid)
    rounding = materialize_rounding(rounding, grid.architecture)
    tableau = multirate_infinitesimal_step_tableau(coefficients, FT)
    s = number_of_stages(tableau)
    s ≥ 3 || throw(ArgumentError("MultirateInfinitesimalStep needs at least three stages; got $s."))

    substepper = build_acoustic_substepper(grid, dynamics, prognostic_fields; cache_advecting_state)

    storage = stage_storage(tableau)
    fold = fold_stage_increments[]
    folded_names = fold ? (keys(dynamics_prognostic_fields(dynamics))..., :ρu, :ρv, :ρw) : ()
    Uⁿ = outer_step_state_storage(storage, prognostic_fields, U⁰)
    accumulators = stage_accumulators(storage, prognostic_fields, U⁰, folded_names, s)
    scalar_tendencies = multirate_scalar_tendency_timing(storage, post_loop_scalar_tendencies, grid, prognostic_fields,
                                                         dynamics, moisture_forcing_names)
    fast_tendency = fast_tendency_storage(storage, scalar_tendencies, grid, prognostic_fields, dynamics, substepper,
                                          moisture_forcing_names)

    step_start = step_start_storage(grid, substepper, tableau, relinearize_stages)

    if relinearize_stages && !is_diagonal(tableau.β)
        @warn "relinearize_stages = true with a non-diagonal tableau is first-order accurate in time: " *
              "the slow tendencies of earlier stages were split against a different fast operator."
    end

    residual = recombines_stage_increments(tableau) ? stage_residual_storage(grid, dynamics, substepper) : nothing
    transport_mass_fluxes = transport_mass_flux_storage(scalar_tendencies, grid, tableau)

    return MultirateInfinitesimalStep(relinearize_stages, fold, tableau, storage, U⁰, Gⁿ, implicit_solver, substepper,
                                      rounding, Uⁿ, accumulators, fast_tendency, scalar_tendencies, step_start,
                                      transport_density_accumulator_storage(storage, grid, U⁰, fold, s), residual, transport_mass_fluxes,
                                      Ref(1), Ref(1), Ref(NaN))
end

#####
##### Storage layouts
#####
##### What a stage needs of the stages before it is its start state Uⁿ + Σⱼ αᵢⱼ ΔYⱼ and its forcing
##### (1/dᵢ) Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt). How much has to be stored for that depends on the structure of the
##### coefficients:
#####
#####   - With a diagonal `β` no slow tendency of an earlier stage is needed, only increments. If `α` and
#####     `γ` are also confined to the first subdiagonal, the only increment needed is that of the stage
#####     before, ΔYᵢ₋₁ = Yᵢ₋₁ - Uⁿ, and the prognostic fields hold Yᵢ₋₁ until the stage updates them.
#####     So nothing is stored besides Uⁿ and the tendencies (`TwoRegisterStorage`).
#####   - With a diagonal `β` and any `α` and `γ`, the increments themselves are stored instead of an
#####     accumulator pair per later stage (`IncrementStorage`, see `increment_registers`).
#####   - Otherwise the forcing and the offset of every later stage are accumulated as the stages
#####     complete (`AccumulatorStorage`).
#####

abstract type AbstractStageStorage end

struct TwoRegisterStorage <: AbstractStageStorage end
struct IncrementStorage <: AbstractStageStorage end
struct AccumulatorStorage <: AbstractStageStorage end

const DiagonalStorage = Union{TwoRegisterStorage, IncrementStorage}

# A coefficient that is zero but for the round-off of whatever produced the set.
negligible(a, A) = abs(a) ≤ 8 * eps(eltype(A)) * maximum(abs, A; init = zero(eltype(A)))

# Whether only the diagonal of `β` is nonzero: every stage then applies its own slow tendency alone.
is_diagonal(β) = all(i == j || negligible(β[i, j], β) for i in axes(β, 1), j in axes(β, 2))

# Whether only the first subdiagonal of `A` is nonzero: every stage then uses the increment of the
# stage before it alone.
is_first_subdiagonal(A) = all(j == i - 1 || negligible(A[i, j], A) for i in axes(A, 1), j in axes(A, 2))

# Choose the storage layout from the structure of the coefficients. Off, every coefficient set takes
# `AccumulatorStorage`, which reproduces the others to round-off. A time stepper reads the switch
# when it is constructed.
const structured_stage_storage = Ref(true)

# The storage layout of a coefficient set. `IncrementStorage` covers up to five stages: the five-stage
# layout rests on recovering two increments from two combinations of them (see
# `penultimate_start_weights`), which the coefficients must allow.
function stage_storage(tableau)
    (; α, β, γ) = tableau
    s = number_of_stages(tableau)
    structured_stage_storage[] || return AccumulatorStorage()
    is_diagonal(β) || return AccumulatorStorage()
    is_first_subdiagonal(α) && is_first_subdiagonal(γ) && return TwoRegisterStorage()
    s ≤ 4 && return IncrementStorage()
    s == 5 && recombination_is_well_conditioned(tableau) && return IncrementStorage()
    return AccumulatorStorage()
end

# Whether (α₅₁, α₅₂) and (γ₅₁, γ₅₂) are far enough from parallel for ΔY₁ and ΔY₂ to be recovered
# from the two combinations of them that the last stage of a five-stage set needs.
function recombination_is_well_conditioned(tableau)
    (; α, γ) = tableau
    determinant = α[5, 1] * γ[5, 2] - α[5, 2] * γ[5, 1]
    return abs(determinant) > hypot(α[5, 1], α[5, 2]) * hypot(γ[5, 1], γ[5, 2]) / 100
end

# The step-start state Uⁿ. With `TwoRegisterStorage` nothing else is stored, so it takes the storage
# of `U⁰`, which another time stepper may share.
outer_step_state_storage(::TwoRegisterStorage, prognostic_fields, U⁰) = U⁰
outer_step_state_storage(storage, prognostic_fields, U⁰) = map(similar, prognostic_fields)

stage_accumulators(::TwoRegisterStorage, prognostic_fields, U⁰, folded_names, s) = ()
stage_accumulators(::IncrementStorage, prognostic_fields, U⁰, folded_names, s) =
    increment_registers(prognostic_fields, U⁰, folded_names, s)

# The increment registers of a coefficient set with a diagonal `β`, a tuple of `NamedTuple`s of fields,
# of which the first is `U⁰`. A field that forms its increments in the assembly (`folded_names`) has
# min(s - 2, 2) of them: stage i < s keeps ΔYᵢ₋₁ in the next free one, and the last stage writes its
# offset over the first. With five stages, stage 4 has no free register for ΔY₃: it replaces ΔY₁ and
# ΔY₂ by the two combinations of ΔY₁, ΔY₂ and ΔY₃ that stage 5 needs, and forms its own start state
# from those and the prognostic field, which holds Y₃ until the stage ends (`penultimate_start_weights`).
#
# A field whose increments are stored when a stage completes has max(s - 2, 2): ΔY₁ to ΔYₛ₋₂ raw, and
# after stage s - 1 the offset and the forcing of the last stage in the first two. The thermodynamic
# density forms its increments in the assembly, but it is not told apart from the scalars here, where
# only the dynamics is known, so it is given as many registers as they are.
function increment_registers(prognostic_fields, U⁰, folded_names, s)
    names = keys(prognostic_fields)
    register_count(name) = name ∈ folded_names ? min(s - 2, 2) : max(s - 2, 2)

    return ntuple(max(s - 2, 2)) do m
        holders = filter(name -> register_count(name) ≥ m, names)
        NamedTuple{holders}(map(name -> m == 1 ? U⁰[name] : similar(prognostic_fields[name]), holders))
    end
end

# The accumulators of stages 3 to s. Stage 3's offset is `U⁰`. A field that forms its increments
# in the assembly (`folded_names`) reads stage 3's forcing for the last time in the kernel that
# first writes stage 4's offset, so the two share storage. The thermodynamic density also forms its
# increments in the assembly, but it is not told apart from the scalars here, where only the
# dynamics is known, so its stage-4 offset has storage of its own.
function stage_accumulators(::AccumulatorStorage, prognostic_fields, U⁰, folded_names, s)
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
function transport_density_accumulator_storage(::AccumulatorStorage, grid, U⁰, fold, s)
    second = fold ? U⁰.ρᵈ : CenterField(grid)
    return (second, ntuple(_ -> CenterField(grid), s - 2)...)
end

# With a diagonal `β` the combination is formed from the increments of the dry density where the
# stage is assembled.
transport_density_accumulator_storage(::DiagonalStorage, grid, U⁰, fold, s) = ()

# The correction of the scalars' recombined mass fluxes (see "Mass consistency of the recombined
# scalar fluxes" below).
struct ScalarMassFluxCorrection{W, F}
    weights :: W      # W = β⁻¹ diag(d) - I, the weight of each stage's mass flux in each stage's correction
    mass_fluxes :: F  # stages 3 to s: the part known so far of the correction's mass flux Rᵢ = Σⱼ Wᵢⱼ Fⱼ; the first holds F₁ through stage 2
end

# `nothing` with pre-loop scalar tendencies, whose mass fluxes are those of the preceding loops
# whatever the tableau, and with a diagonal `β`, whose stages each apply their own flux alone.
# Otherwise one face-located triple per stage after the second (see `combine_scalar_mass_fluxes!`).
transport_mass_flux_storage(::PreLoopScalarTendencies, grid, tableau) = nothing

function transport_mass_flux_storage(::PostLoopScalarTendencies, grid, tableau)
    is_diagonal(tableau.β) && return nothing
    s = number_of_stages(tableau)
    mass_fluxes = ntuple(_ -> (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid)), s - 2)
    return ScalarMassFluxCorrection(mass_flux_correction_weights(tableau), mass_fluxes)
end

# W = β⁻¹ diag(d) - I by forward substitution. Its rows sum to zero, since Σⱼ βᵢⱼ = dᵢ.
function mass_flux_correction_weights(tableau)
    (; β, d) = tableau
    s = length(d)
    any(i -> iszero(β[i, i]), 1:s) &&
        throw(ArgumentError("post_loop_scalar_tendencies = true needs a coefficient set whose β is diagonal " *
                            "or has no zero on its diagonal."))
    W = zeros(eltype(d), s, s)
    for k in 1:s, i in k:s
        W[i, k] = ((i == k) * d[k] - sum(β[i, j] * W[j, k] for j in k:i-1; init = zero(eltype(d)))) / β[i, i]
    end
    for i in 1:s
        W[i, i] -= 1
    end
    return W
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
    (starts_from_step_start(ts) || isnothing(ts.residual)) && return nothing
    θ_name = thermodynamic_density_name(model.formulation)
    names = (:ρᵈ, θ_name, :ρu, :ρv, :ρw)

    residuals = map(names) do name
        (; offset, weight, state_weight) = stage_offset(ts, model, name)
        recovered = iszero(state_weight) ? nothing : stage_residual(ts.residual, model, name)
        StartStateResidual(ts.Uⁿ[name], offset, weight, recovered, state_weight)
    end

    return NamedTuple{(:ρ, :ρᵡ, :ρu, :ρv, :ρw)}(residuals)
end

CompressibleEquations.recovered_state_residual(ts::MultirateInfinitesimalStep) =
    ts.stage[] == number_of_stages(ts.tableau) ? nothing : ts.residual

# Scratch for the fast parts of the tendencies (see the `fast_tendency` field).
#
# Pre-loop, the one scratch is written and read while a stage is assembled, so it shares the
# storage of the substepper's stage-end moisture content, which each acoustic loop writes when it
# starts and reads for the last time when it ends.
#
# Post-loop, the moisture and condensate densities each have a field of their own: the build
# writes their advective tendencies there, and the assembly leaves their slow tendencies, which
# the next stage's loop reads (`PredictedStageForcing`). No loop reads the tendency of any other
# scalar, so those take the substepper's field in turn, between two loops: each has its
# advective tendency built and is assembled before the next one's is built
# (`complete_post_loop_scalar_tendencies!`). The thermodynamic variable uses the same field
# before the loop, as it does pre-loop. Where the moisture and condensate densities are not known
# (`moisture_forcing_names = nothing`), every one of these prognostics has a field of its own.
#
# A coefficient set with a diagonal `β` (`DiagonalStorage`) applies every tendency whole, so it
# separates no fast part and needs no scratch.
fast_tendency_storage(::DiagonalStorage, scalar_tendencies, grid, prognostic_fields, dynamics, substepper,
                      moisture_forcing_names) = nothing

fast_tendency_storage(::AccumulatorStorage, ::PreLoopScalarTendencies, grid, prognostic_fields, dynamics, substepper,
                      moisture_forcing_names) = substepper.stage_end_moisture_content

fast_tendency_storage(::AccumulatorStorage, ::PostLoopScalarTendencies, grid, prognostic_fields, dynamics, substepper,
                      moisture_forcing_names) =
    previous_tendency_storage(grid, prognostic_fields, dynamics, moisture_forcing_names)

# Where the slow tendency of post-loop scalar `name` is between stages: in fields kept for that,
# or, where there are none, in the scratch of its fast tendency (see `multirate_scalar_tendency_timing`).
previous_scalar_tendency(ts::MultirateInfinitesimalStep, name) =
    previous_scalar_tendency(ts.scalar_tendencies.previous_tendencies, ts, name)

previous_scalar_tendency(::Nothing, ts, name) = fast_tendency_scratch(ts, name)
previous_scalar_tendency(previous_tendencies, ts, name) = previous_tendencies[name]

# When the scalar tendencies are built. Built after the loop, those of the moisture and condensate
# densities are kept until the next stage's loop, whose pressure they predict (see
# `PredictedStageForcing`). With `AccumulatorStorage` the stage assembly leaves them in the scratch
# of the scalar's fast tendency. A coefficient set with a diagonal `β` has no such scratch, and its
# tendency storage is rewritten before the next loop starts, so it keeps them in fields of their
# own, as `AcousticRungeKutta3` does.
multirate_scalar_tendency_timing(::AccumulatorStorage, post_loop_scalar_tendencies, grid, prognostic_fields, dynamics,
                                 moisture_forcing_names) =
    scalar_tendency_timing(post_loop_scalar_tendencies)

multirate_scalar_tendency_timing(::DiagonalStorage, post_loop_scalar_tendencies, grid, prognostic_fields, dynamics,
                                 moisture_forcing_names) =
    scalar_tendency_timing(post_loop_scalar_tendencies, grid, prognostic_fields, dynamics, moisture_forcing_names)

# The scratch for the fast part of `name`'s tendency: its own field, or the one the substepper
# shares between two loops (see `fast_tendency_storage`).
fast_tendency_scratch(ts::MultirateInfinitesimalStep, name) = fast_tendency_scratch(ts.fast_tendency, ts.substepper, name)
fast_tendency_scratch(scratch::NamedTuple, substepper, name) = get(scratch, name, substepper.stage_end_moisture_content)
fast_tendency_scratch(::Nothing, substepper, name) = nothing
fast_tendency_scratch(scratch, substepper, name) = scratch

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

# The interval over which microphysical sinks are budgeted so that they leave every species
# non-negative: `Δt / C`, with `C` the strong-stability coefficient of the coefficient set (see
# `strong_stability_coefficient`). Every MIS stage of a slow variable is then a convex combination
# of forward-Euler steps no longer than this, each from the state its tendency was evaluated at,
# which is the state `AtmosphereModels.StageStart` budgets against by default. A coefficient set
# with `C = 0` has no such decomposition; the full step `Δt` is used, which bounds the drain of the
# final stage's own tendency but guarantees nothing.
@inline function AtmosphereModels.stage_time_step(ts::MultirateInfinitesimalStep, clock)
    Δt = ifelse(isfinite(ts.current_Δt[]), ts.current_Δt[], clock.last_Δt)
    C = ts.tableau.strong_stability_coefficient
    return ifelse(C > 0, Δt / C, Δt)
end

# Stages that recombine earlier increments need the acoustic substeps to return the state they
# were given without a skew between momentum and pressure (see `SymmetricSubsteps`). The others
# (`:WickerSkamarock3`) keep the forward–backward substeps of `AcousticRungeKutta3`.
CompressibleEquations.horizontal_substep_form(ts::MultirateInfinitesimalStep) =
    recombines_stage_increments(ts.tableau) ? SymmetricSubsteps() : ForwardBackwardSubsteps()

# With the production model's coefficients and scalar tendency timing (see the `AcousticRungeKutta3` method).
AtmosphereModels.adiabatic_twin_timestepper(ts::MultirateInfinitesimalStep, ::SplitExplicitCompressibleDynamics) =
    :MultirateInfinitesimalStep, (; coefficients = ts.tableau,
                                    post_loop_scalar_tendencies = ts.scalar_tendencies isa PostLoopScalarTendencies)

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
struct TransientPartitionedTendencySplit{C}   # the same for a scalar whose slow tendency no loop reads:
    advective_tendency :: C                   # C is shared scratch, which the assembly leaves as it is
end
struct StageEntryForceSplit{R}                # momentum: G is slow, and so is the remainder R of the
    remainder :: R                            # stage-entry force that the loop adds at weight one
end

const AnyPartitionedTendencySplit = Union{PartitionedTendencySplit, TransientPartitionedTendencySplit}

Adapt.adapt_structure(to, s::FastSlowTendencySplit) = FastSlowTendencySplit(adapt(to, s.fast_tendency))
Adapt.adapt_structure(to, s::PartitionedTendencySplit) = PartitionedTendencySplit(adapt(to, s.advective_tendency))
Adapt.adapt_structure(to, s::TransientPartitionedTendencySplit) = TransientPartitionedTendencySplit(adapt(to, s.advective_tendency))
Adapt.adapt_structure(to, s::StageEntryForceSplit) = StageEntryForceSplit(adapt(to, s.remainder))

@inline slow_part(i, j, k, grid, ::SlowTendencySplit, G) = @inbounds G[i, j, k]
@inline slow_part(i, j, k, grid, ::ContinuityTendencySplit, G) = zero(eltype(G))
@inline slow_part(i, j, k, grid, s::FastSlowTendencySplit, G) = @inbounds G[i, j, k] - s.fast_tendency[i, j, k]
@inline slow_part(i, j, k, grid, s::AnyPartitionedTendencySplit, G) = @inbounds G[i, j, k] + s.advective_tendency[i, j, k]
@inline slow_part(i, j, k, grid, s::StageEntryForceSplit, G) = @inbounds G[i, j, k] + stage_entry_force_remainder(i, j, k, grid, s.remainder)

@inline full_tendency(i, j, k, grid, split, G) = @inbounds G[i, j, k]
@inline full_tendency(i, j, k, grid, s::AnyPartitionedTendencySplit, G) = @inbounds G[i, j, k] + s.advective_tendency[i, j, k]

# The slow tendency of a moisture or condensate density is kept where its advective part was, once
# the stage is assembled: it predicts the next stage's, for the moisture's part of that stage's
# loop pressure (see `PredictedStageForcing`).
@inline keep_slow_tendency!(i, j, k, split, slow) = nothing
@inline keep_slow_tendency!(i, j, k, s::PartitionedTendencySplit, slow) = @inbounds s.advective_tendency[i, j, k] = slow

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
        keep_slow_tendency!(i, j, k, split, slow)
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
        keep_slow_tendency!(i, j, k, split, S₂)
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
        keep_slow_tendency!(i, j, k, split, slow)
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
        keep_slow_tendency!(i, j, k, split, slow)
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
    StageStartState(adapt(to, s.Uⁿ), adapt(to, s.offset), adapt(to, s.weight), s.rounding, s.key)

Base.size(s::StageStartState) = size(s.Uⁿ)

Base.@propagate_inbounds Base.getindex(s::StageStartState, i::Int, j::Int, k::Int) =
    rounded_add(s.rounding, s.Uⁿ[i, j, k], s.weight * s.offset[i, j, k], i, j, k, s.key)

# What the round-to-nearest start state Uⁿ + w Z lacks of the exact sum, where the acoustic loop
# reads it (see `start_state_residual`).
#
# An offset that reads the increment ΔYᵢ₋₁ from the prognostic field (`StateIncrement`) reads it
# without the rounding error of the recovery of Yᵢ₋₁, which is only kept until the loop starts
# (`stage_residual_storage`), so that the start state is the same wherever it is read. The exact
# offset has `recovered_weight` times that error (`recovered`) more, which is added here.
struct StartStateResidual{T, U, Z, R} <: AbstractArray{T, 3}
    Uⁿ :: U
    offset :: Z
    weight :: T
    recovered :: R
    recovered_weight :: T
end

Adapt.adapt_structure(to, r::StartStateResidual) =
    StartStateResidual(adapt(to, r.Uⁿ), adapt(to, r.offset), adapt(to, r.weight), adapt(to, r.recovered), adapt(to, r.recovered_weight))

Base.size(r::StartStateResidual) = size(r.Uⁿ)

Base.@propagate_inbounds function Base.getindex(r::StartStateResidual, i::Int, j::Int, k::Int)
    uⁿ = r.Uⁿ[i, j, k]
    offset = r.weight * r.offset[i, j, k]
    return with_recovered_residual(rounding_error(uⁿ, offset, uⁿ + offset), r.recovered, r.recovered_weight, i, j, k)
end

@inline with_recovered_residual(error, ::Nothing, weight, i, j, k) = error
Base.@propagate_inbounds with_recovered_residual(error, recovered, weight, i, j, k) = error + weight * recovered[i, j, k]

# The increment ΔYᵢ₋₁ = Yᵢ₋₁ - Uⁿ of the stage before the current one, formed where it is read from
# the prognostic field `u`, which holds Yᵢ₋₁ until the stage updates it: the acoustic prognostics
# when the loop recovers them, the others in their own update, each point after it was read. With
# the rounding error of the recovery of Yᵢ₋₁ where one is given (`residual`), the increment is
# exact to its own precision (see `state_increment`).
struct StateIncrement{T, U, N, R} <: AbstractArray{T, 3}
    u :: U
    Uⁿ :: N
    residual :: R
end

StateIncrement(u, Uⁿ, residual) = StateIncrement{eltype(u), typeof(u), typeof(Uⁿ), typeof(residual)}(u, Uⁿ, residual)

Adapt.adapt_structure(to, d::StateIncrement) = StateIncrement(adapt(to, d.u), adapt(to, d.Uⁿ), adapt(to, d.residual))

Base.size(d::StateIncrement) = size(d.u)

Base.@propagate_inbounds Base.getindex(d::StateIncrement, i::Int, j::Int, k::Int) =
    with_residual(d.u[i, j, k] - d.Uⁿ[i, j, k], d.residual, i, j, k)

# Σₘ wₘ aₘ over a tuple of arrays, added from the left, so that it reproduces a sum accumulated
# term by term. Its length is known at compile time, so it unrolls.
@inline weighted_sum(i, j, k, ::Tuple{}, ::Tuple{}) = false
Base.@propagate_inbounds weighted_sum(i, j, k, a::Tuple, w::Tuple) =
    weighted_sum(i, j, k, w[1] * a[1][i, j, k], Base.tail(a), Base.tail(w))

@inline weighted_sum(i, j, k, sum, ::Tuple{}, ::Tuple{}) = sum
Base.@propagate_inbounds weighted_sum(i, j, k, sum, a::Tuple, w::Tuple) =
    weighted_sum(i, j, k, sum + w[1] * a[1][i, j, k], Base.tail(a), Base.tail(w))

# An offset Σⱼ αᵢⱼ ΔYⱼ summed where it is read, from increments in registers or in the prognostic field.
struct WeightedSum{T, A, W} <: AbstractArray{T, 3}
    terms :: A
    weights :: W
end

WeightedSum(terms, weights) = WeightedSum{eltype(weights), typeof(terms), typeof(weights)}(terms, weights)

Adapt.adapt_structure(to, s::WeightedSum) = WeightedSum(adapt(to, s.terms), adapt(to, s.weights))

Base.size(s::WeightedSum) = size(first(s.terms))

Base.@propagate_inbounds Base.getindex(s::WeightedSum, i::Int, j::Int, k::Int) = weighted_sum(i, j, k, s.terms, s.weights)

# Whether the current stage starts from Uⁿ itself.
function starts_from_step_start(ts::MultirateInfinitesimalStep)
    i = ts.stage[]
    i == 1 && return true
    return ts.storage isa TwoRegisterStorage && iszero(ts.tableau.α[i, i - 1])
end

# The offset Zᵢ = Σⱼ αᵢⱼ ΔYⱼ of field `name` in the current stage i ≥ 2: an array, the weight it
# takes, and the weight in the product of the two of an increment that is read from the prognostic
# field (zero when none is).
stage_offset(ts::MultirateInfinitesimalStep, model, name) = stage_offset(ts.storage, ts, model, name)

# Stage 2 starts from Uⁿ + α₂₁ ΔY₁, with the raw ΔY₁ in the storage of stage 3's offset, and stage
# i ≥ 3 from Uⁿ + Zᵢ.
function stage_offset(::AccumulatorStorage, ts, model, name)
    i = ts.stage[]
    FT = eltype(ts.tableau.d)
    weight = i == 2 ? ts.tableau.α[2, 1] : one(FT)
    return (; offset = ts.accumulators[max(i - 2, 1)].offset[name], weight, state_weight = zero(FT))
end

# Stage i starts from Uⁿ + αᵢ,ᵢ₋₁ ΔYᵢ₋₁, with ΔYᵢ₋₁ read from the prognostic field.
function stage_offset(::TwoRegisterStorage, ts, model, name)
    i = ts.stage[]
    weight = ts.tableau.α[i, i - 1]
    offset = StateIncrement(prognostic_fields(model)[name], ts.Uⁿ[name], nothing)
    return (; offset, weight, state_weight = weight)
end

function stage_offset(::IncrementStorage, ts, model, name)
    i = ts.stage[]
    α = ts.tableau.α
    s = number_of_stages(ts.tableau)
    FT = eltype(α)
    R = increment_registers(ts, name)
    source = stage_increment_source(model, name)

    # The raw ΔY₁, and in the last stage its completed offset
    i == 2 && return (; offset = R[1], weight = α[2, 1], state_weight = zero(FT))
    i == s && return (; offset = R[1], weight = one(FT), state_weight = zero(FT))

    if source isa IncrementFromState && i == 4 # of five stages
        p, q, c = penultimate_start_weights(ts.tableau, ts.current_Δt[])
        ΔY₃ = StateIncrement(source.u, ts.Uⁿ[name], nothing)
        return (; offset = WeightedSum((R[1], R[2], ΔY₃), (p, q, c)), weight = one(FT), state_weight = c)
    end

    # The raw increments ΔY₁ to ΔYᵢ₋₁
    js = ntuple(identity, i - 1)
    offset = WeightedSum(map(j -> R[j], js), map(j -> α[i, j], js))
    return (; offset, weight = one(FT), state_weight = zero(FT))
end

# The increment registers that field `name` has.
increment_registers(ts::MultirateInfinitesimalStep, name) =
    map(register -> register[name], filter(register -> haskey(register, name), ts.accumulators))

# Stage 4 of a five-stage set, for a field that forms its increments in the assembly. The assembly
# has replaced ΔY₁ and ΔY₂ in the registers by the combinations that stage 5 needs,
#
#   P = α₅₁ ΔY₁ + α₅₂ ΔY₂ + α₅₃ ΔY₃,    Q = w₅₁ ΔY₁ + w₅₂ ΔY₂ + w₅₃ ΔY₃,    w₅ⱼ = γ₅ⱼ / (d₅ Δt),
#
# and the prognostic field still holds Y₃ = Uⁿ + ΔY₃. Solving the two for ΔY₁ and ΔY₂ gives the
# offset of stage 4 as Z₄ = α₄₁ ΔY₁ + α₄₂ ΔY₂ + α₄₃ ΔY₃ = p P + q Q + c ΔY₃.
function penultimate_start_weights(tableau, Δt)
    (; α, γ, d) = tableau
    FT = eltype(d)
    w = ntuple(j -> γ[5, j] / (d[5] * convert(FT, Δt)), 3)
    determinant = α[5, 1] * w[2] - α[5, 2] * w[1]
    p = (α[4, 1] * w[2] - α[4, 2] * w[1]) / determinant
    q = (α[4, 2] * α[5, 1] - α[4, 1] * α[5, 2]) / determinant
    c = α[4, 3] - p * α[5, 3] - q * w[3]
    return p, q, c
end

# A field that carries a residual rounds its start state to nearest, and the loop adds the rounding
# error to the perturbation it starts from. The others round theirs with the time stepper's
# round-off treatment.
start_state_rounding(ts, ::Nothing) = ts.rounding
start_state_rounding(ts, residual) = RoundToNearest()

# `Uⁿ` in stage 1 and `U⁰ = Uⁿ + Zᵢ` after that, formed where it is read. Both are read-only
# during a stage.
function stage_start_state(ts::MultirateInfinitesimalStep, model)
    starts_from_step_start(ts) && return ts.Uⁿ
    names = keys(ts.Uⁿ)

    start_states = map(names) do name
        (; offset, weight) = stage_offset(ts, model, name)
        rounding = start_state_rounding(ts, stage_residual(ts.residual, model, name))
        key = start_state_rounding_key(ts, model, name)
        StageStartState(ts.Uⁿ[name], offset, weight, rounding, key)
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
    compute_fast_thermodynamic_tendency!(ts.storage, C, ts.substepper, model)
    compute_pressure_remainder!(ts.step_start, ts.substepper, model, ts.Uⁿ)
    assemble_transport_density_tendency!(ts, model, Δt)
    for name in pre_loop_assembly_names(ts.scalar_tendencies, model)
        assemble_stage_forcing!(ts, model, Δt, name, stage_tendency_split(ts, model, name, θ_name, C))
    end
    return nothing
end

# A stage with a diagonal `β` applies its full tendency as it is, so the slow part is not needed.
compute_fast_thermodynamic_tendency!(::AccumulatorStorage, C, substepper, model) =
    compute_linearized_thermodynamic_flux_divergence!(C, substepper, model)

compute_fast_thermodynamic_tendency!(::DiagonalStorage, C, substepper, model) = nothing

# Give the transport density tendency the stage combination that the moisture forcing has:
# (1/dᵢ) [Σⱼ βᵢⱼ Gᵗⱼ + Σⱼ γᵢⱼ ΔYⱼ / Δt], with ΔYⱼ the increments of the dry density.
assemble_transport_density_tendency!(ts::MultirateInfinitesimalStep, model, Δt) =
    assemble_transport_density_tendency!(ts.storage, ts, model, Δt)

# With a diagonal `β`: βᵢᵢ Gᵗᵢ / dᵢ plus the increment part of the dry density's own forcing, read
# before the dry density's assembly rewrites its registers.
function assemble_transport_density_tendency!(::DiagonalStorage, ts, model, Δt)
    i = ts.stage[]
    (; β, d) = ts.tableau
    grid = model.grid
    terms, weights = transport_density_increment_forcing(ts.storage, ts, model, Δt)
    wβᵢ = β[i, i] / d[i]
    all(is_known_zero, weights) && isone(wβᵢ) && return nothing
    launch!(architecture(grid), grid, :xyz, _combine_transport_density_tendency!,
            ts.substepper.transport_density_tendency, terms, weights, wβᵢ)
    return nothing
end

# The increment of the dry density as the transport density tendency has always taken it, without
# the rounding error of the recovery.
function transport_density_increment_forcing(::TwoRegisterStorage, ts, model, Δt)
    i = ts.stage[]
    i == 1 && return (), ()
    (; γ, d) = ts.tableau
    ΔY = StateIncrement(model.dynamics.dry_density, ts.Uⁿ.ρᵈ, nothing)
    return (ΔY,), (γ[i, i - 1] / (d[i] * Δt),)
end

transport_density_increment_forcing(::IncrementStorage, ts, model, Δt) =
    increment_forcing(stage_increment_source(model, :ρᵈ), ts, model, Δt, :ρᵈ)

@kernel function _combine_transport_density_tendency!(Gᵗ, terms, weights, wβᵢ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gᵗ[i, j, k] = weighted_sum(i, j, k, terms, weights) + wβᵢ * Gᵗ[i, j, k]
end

function assemble_transport_density_tendency!(::AccumulatorStorage, ts, model, Δt)
    i = ts.stage[]
    (; β, γ, d) = ts.tableau
    s = length(d)
    grid = model.grid
    ks = later_stages(i, s)
    T = ts.transport_density_accumulators
    Tᵢ = i == 1 ? nothing : T[i - 1]
    wγᵢ = i == 1 ? zero(eltype(d)) : γ[i, i - 1] / (d[i] * Δt)
    wγᵏ = i == 1 ? map(k -> zero(eltype(d)), ks) : increment_weights(γ, d, Δt, i - 1, ks)
    wβᵏ = transport_density_forcing_weights(ts.scalar_tendencies, β, d, i, ks)
    launch!(architecture(grid), grid, :xyz, _assemble_transport_density_tendency!,
            ts.substepper.transport_density_tendency, Tᵢ, map(k -> T[k - 1], ks),
            model.dynamics.dry_density, ts.Uⁿ.ρᵈ, β[i, i] / d[i], wγᵢ, wγᵏ, wβᵏ)
    return nothing
end

# What the assembly adds of the stage's transport density tendency to the accumulators of later
# stages. With `PostLoopScalarTendencies` the assembly holds the tendency of the stage before,
# which stands in for the stage's own in its forcing alone: the stage's own is added to the
# accumulators after its loop (`complete_transport_density_tendency!`).
transport_density_forcing_weights(::PreLoopScalarTendencies, β, d, i, ks) = forcing_weights(β, d, i, ks)
transport_density_forcing_weights(::PostLoopScalarTendencies, β, d, i, ks) = map(k -> zero(eltype(d)), ks)

# After the post-loop build, which left the dry air's share of the divergence of the stage's mass
# flux Fᵢ in the transport density tendency (`cache_post_loop_transport!`): add that of the mass
# flux Rᵢ of the stage's correction, which the scalars' tendencies carry too, and add the sum to
# the accumulators of later stages.
function complete_transport_density_tendency!(ts::MultirateInfinitesimalStep, model)
    i = ts.stage[]
    (; β, d) = ts.tableau
    s = length(d)
    is_diagonal(β) && return nothing
    grid = model.grid
    ks = later_stages(i, s)
    T = ts.transport_density_accumulators
    R = stage_correction_mass_fluxes(ts.transport_mass_fluxes, ts, model)
    launch!(architecture(grid), grid, :xyz, _complete_transport_density_tendency!,
            ts.substepper.transport_density_tendency, map(k -> T[k - 1], ks), grid, R,
            model.dynamics.dry_density, total_density(model.dynamics), forcing_weights(β, d, i, ks))
    return nothing
end

# `nothing` in stage 1, whose correction has no mass flux.
stage_correction_mass_fluxes(::Nothing, ts, model) = nothing
stage_correction_mass_fluxes(correction, ts, model) =
    ts.stage[] == 1 ? nothing : correction_mass_fluxes(correction, ts, model)

@inline correction_dry_density_tendency(i, j, k, grid, ::Nothing, ρᵈ, ρ) = zero(grid)

@inline function correction_dry_density_tendency(i, j, k, grid, R, ρᵈ, ρ)
    ∇_R = V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, correction_mass_flux, R.u) +
                                   δyᵃᶜᵃ(i, j, k, grid, correction_mass_flux, R.v) +
                                   δzᵃᵃᶜ(i, j, k, grid, correction_mass_flux, R.w))
    @inbounds ρ̂ = ifelse(ρ[i, j, k] == 0, one(eltype(ρ)), ρ[i, j, k])
    return @inbounds - ρᵈ[i, j, k] / ρ̂ * ∇_R
end

@kernel function _complete_transport_density_tendency!(Gᵗ, T, grid, R, ρᵈ, ρ, wβᵏ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        G = Gᵗ[i, j, k] + correction_dry_density_tendency(i, j, k, grid, R, ρᵈ, ρ)
        Gᵗ[i, j, k] = G
        accumulate!(i, j, k, T, wβᵏ, G)
    end
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
    complete_transport_density_tendency!(ts, model)
    deferred_names = deferred_advection_names(ts, model)
    for name in scalar_prognostic_names(model)
        name ∈ deferred_names && continue # assembled already (`complete_post_loop_scalar_tendencies!`)
        assemble_stage_forcing!(ts, model, Δt, name, post_loop_tendency_split(ts.storage, ts, name))
    end
    start_scalar_mass_flux_sums!(ts.transport_mass_fluxes, ts, model)
    return nothing
end

post_loop_tendency_split(::AccumulatorStorage, ts, name) = PartitionedTendencySplit(fast_tendency_scratch(ts, name))
post_loop_tendency_split(::DiagonalStorage, ts, name) = SlowTendencySplit() # the build has added the whole tendency to `Gⁿ`

# The post-loop scalars whose advective tendency has no field of its own (see
# `fast_tendency_storage`): the build left it out, and it is built and assembled one scalar at a
# time, in scratch they share.
deferred_advection_names(ts::MultirateInfinitesimalStep, model) =
    deferred_advection_names(ts.scalar_tendencies, ts.fast_tendency, model)

deferred_advection_names(scalar_tendencies, fast_tendency, model) = ()

deferred_advection_names(::PostLoopScalarTendencies, fast_tendency::NamedTuple, model) =
    filter(name -> !haskey(fast_tendency, name), scalar_prognostic_names(model))

# Build the advective tendency of each of those scalars, as the build would have, and assemble its
# stage forcing, before the next one's is built in the same scratch. This follows the
# microphysical sources and the `TendencyCallsite` callbacks, which complete the non-advective part
# in `Gⁿ` that the assembly adds the advective part to, in that order whichever way it was built.
# The tracers still hold their specific values, which the advection reads and the assembly does not:
# these scalars' increments are stored (`stage_increment_source`).
function complete_post_loop_scalar_tendencies!(ts::MultirateInfinitesimalStep, model, Δt)
    names = deferred_advection_names(ts, model)
    isempty(names) && return nothing

    grid = model.grid
    Δt = convert(eltype(grid), Δt)
    C = ts.substepper.stage_end_moisture_content
    R = stage_correction_mass_fluxes(ts.transport_mass_fluxes, ts, model)
    scalars = specific_scalar_fields(model)

    for name in names
        launch!(architecture(grid), grid, :xyz, _compute_scalar_advective_tendency!, C, grid, R,
                scalars[name], Val(name), model.advection[name], model.dynamics, transport_velocities(model),
                model.microphysics, model.microphysical_fields)
        assemble_stage_forcing!(ts, model, Δt, name, TransientPartitionedTendencySplit(C))
    end

    return nothing
end

@kernel function _compute_scalar_advective_tendency!(C, grid, R, c, name, advection, dynamics, velocities,
                                                     microphysics, microphysical_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds C[i, j, k] = stage_advective_tendency(i, j, k, grid, R, c, name, advection, dynamics, velocities,
                                                    microphysics, microphysical_fields)
end

# The advective tendency of a post-loop scalar in a stage whose correction has the mass flux `R`
# (see "Mass consistency of the recombined scalar fluxes" below), or none.
@inline stage_advective_tendency(i, j, k, grid, ::Nothing, c, name, advection, dynamics, velocities,
                                 microphysics, microphysical_fields) =
    scalar_advective_tendency(i, j, k, grid, c, name, advection, dynamics, velocities, microphysics, microphysical_fields)

@inline stage_advective_tendency(i, j, k, grid, R, c, name, advection, dynamics, velocities,
                                 microphysics, microphysical_fields) =
    mass_consistent_advective_tendency(i, j, k, grid, R, c, name, advection, dynamics, velocities,
                                       microphysics, microphysical_fields)

#####
##### The moisture's part of the loop's pressure with post-loop scalar tendencies
#####
##### The loop's pressure ramps with the stage forcing of the moisture and condensate densities
##### (Section 3b of the acoustic substepper), (1/dᵢ)[βᵢᵢ Sᵢ + Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt)] with the sum
##### over the stages j < i. Built after the loop, Sᵢ is not in hand when the loop starts, and the
##### loop is given the forcing with the slow tendency of the stage before, Sᵢ₋₁, in its place (in
##### stage 1 that of the last stage of the step before): the same transport, of the mass fractions
##### one stage earlier. The rest of the forcing is known. Sᵢ₋₁ is kept in the scalar's
##### `fast_tendency` scratch, where the assembly leaves it (`keep_slow_tendency!`) and nothing
##### writes until the next post-loop build, or, for a coefficient set with a diagonal `β`, which has
##### no such scratch, in a field of its own (`previous_scalar_tendency`). What the loop's pressure
##### missed of the moisture's change over a stage, βᵢᵢ Δt (Sᵢ − Sᵢ₋₁), reaches the next stage with
##### its start state.
#####
##### The dilution of the moisture is paired with the dry air's share of the divergence of the mass
##### flux those tendencies carry (`transport_density_tendency`), so that takes the same
##### combination: each stage's is evaluated after its loop, with the mass flux of the stage's
##### correction, added to the accumulators of later stages there, and predicts the next stage's.
#####

# The stage forcing of one post-loop scalar with the previous stage's slow tendency standing in
# for the stage's own, formed where it is read: `weight` times that tendency, plus the part that
# earlier stages contributed.
struct PredictedStageForcing{T, S, A, B} <: AbstractArray{T, 3}
    previous_tendency :: S # Sᵢ₋₁
    weight :: T            # βᵢᵢ / dᵢ
    first :: A             # stage 2: S₁; later stages: the forcing accumulated so far; `nothing` in stage 1
    first_weight :: T
    second :: B            # stage 2: ΔY₁; `nothing` otherwise
    second_weight :: T
end

Adapt.adapt_structure(to, G::PredictedStageForcing) =
    PredictedStageForcing(adapt(to, G.previous_tendency), adapt(to, G.weight), adapt(to, G.first),
                          adapt(to, G.first_weight), adapt(to, G.second), adapt(to, G.second_weight))

Base.size(G::PredictedStageForcing) = size(G.previous_tendency)

@inline weighted_term(::Nothing, w, i, j, k) = zero(w)
Base.@propagate_inbounds weighted_term(a, w, i, j, k) = w * a[i, j, k]

Base.@propagate_inbounds Base.getindex(G::PredictedStageForcing, i::Int, j::Int, k::Int) =
    G.weight * G.previous_tendency[i, j, k] + weighted_term(G.first, G.first_weight, i, j, k) +
    weighted_term(G.second, G.second_weight, i, j, k)

CompressibleEquations.moisture_forcing(ts::MultirateInfinitesimalStep, model, name) =
    stage_moisture_forcing(ts.scalar_tendencies, ts, model, name)

stage_moisture_forcing(::PostLoopScalarTendencies, ts::MultirateInfinitesimalStep, model, name) =
    predicted_stage_forcing(ts.storage, ts, model, name)

# The increments of the scalars are stored before the next stage starts (`store_stage_increment!`), so
# the accumulated forcing of stage i ≥ 3 is complete but for its own tendency when its loop starts.
function predicted_stage_forcing(::AccumulatorStorage, ts, model, name)
    i = ts.stage[]
    (; β, γ, d) = ts.tableau
    FT = eltype(model.grid)
    S = previous_scalar_tendency(ts, name)
    w = convert(FT, β[i, i] / d[i])

    i == 1 && return PredictedStageForcing(S, w, nothing, one(FT), nothing, one(FT))

    if i == 2
        Δt = convert(FT, ts.current_Δt[])
        return PredictedStageForcing(S, w, ts.accumulators[1].forcing[name], convert(FT, β[2, 1] / d[2]),
                                     ts.accumulators[1].offset[name], convert(FT, γ[2, 1] / (d[2] * Δt)))
    end

    return PredictedStageForcing(S, w, ts.accumulators[i - 2].forcing[name], one(FT), nothing, one(FT))
end

# With a diagonal `β` the stage applies its own tendency whole (βᵢᵢ = dᵢ), so the prediction is
# Sᵢ₋₁ plus the increments' part of the stage forcing, read where the stage's assembly will read it
# after the loop (`increment_forcing`): nothing changes the scalars or their registers in between.
# A stage that no increment forces (every stage of Wicker–Skamarock RK3) takes Sᵢ₋₁ as it is.
function predicted_stage_forcing(::DiagonalStorage, ts, model, name)
    FT = eltype(model.grid)
    S = previous_scalar_tendency(ts, name)
    Δt = convert(FT, ts.current_Δt[])
    terms, weights = increment_forcing(stage_increment_source(model, name), ts, model, Δt, name)
    all(is_known_zero, weights) && return S
    return PredictedStageForcing(S, one(FT), WeightedSum(terms, weights), one(FT), nothing, one(FT))
end

#####
##### Mass consistency of the recombined scalar fluxes
#####
##### Stage i moves the dry air with the mass flux Fᵢ of its own loop, over dᵢ Δt, while the scalars'
##### stage forcing recombines the advective tendencies of the stages so far, -∇·(Fⱼ ĉⱼ₋₁) with
##### weights βᵢⱼ (Σⱼ βᵢⱼ = dᵢ): the mass flux Σⱼ βᵢⱼ Fⱼ they carry is not dᵢ Fᵢ unless `β` is
##### diagonal, so a uniform mass fraction does not stay uniform, and the scalars are first-order
##### accurate in time wherever the mass flux changes during a step.
#####
##### Each stage therefore transports the scalars with the mass flux Fⱼ + Rⱼ,
#####
#####   Rⱼ = Σₖ Wⱼₖ Fₖ,   W = β⁻¹ diag(d) - I   (k ≤ j, since β is lower triangular),
#####
##### for which the recombined fluxes carry the stage's own: Σⱼ βᵢⱼ (Fⱼ + Rⱼ) = dᵢ Fᵢ. Every stage
##### then applies dᵢ Fᵢ to a uniform mass fraction exactly. The rows of W sum to zero, so Rⱼ is a
##### combination of differences of the stages' mass fluxes, O(Δt) smaller than Fⱼ. Fⱼ + Rⱼ is the
##### slow mass flux that the stages would recombine into the dry air's own, so with it the scalars
##### have the order of accuracy of the slow method where the mass flux changes during a step.
#####
##### Both mass fluxes are carried by one reconstruction of the scalar: the advection operator
##### reads the velocity whose mass flux is Fⱼ + Rⱼ (`MassConsistentVelocity`). A separate flux
##### of Rⱼ with a reconstruction of its own, a first-order upwind one in its own direction say,
##### would differ from the operator's by a truncation error that does not decrease with the time
##### step, times Rⱼ: first-order accurate in time.
#####
##### The correction is part of the slow tendency Sⱼ that the stages recombine, rather than an
##### addition to one stage's forcing, so that the scalars are still advanced by the slow method
##### alone: with a strong-stability-preserving coefficient set every stage remains a convex
##### combination of forward-Euler steps (see `strong_stability_coefficient`), each a step of the
##### scalar's own advection operator with the velocity of Fⱼ + Rⱼ. Added to a stage at weight one
##### the correction would enter later stages through their increments with weights of either
##### sign, and a species that a microphysical sink has emptied would go negative.
#####
##### Fⱼ is the flux the tendencies were built with: the stage-entry total density times the loop's
##### transport velocity, of which adaptive implicit vertical advection leaves a fraction to the
##### implicit solve. That fraction is taken from the advection scheme of the prognostic moisture
##### and the velocity of the air. A scalar with another vertical time discretization, and one that
##### sediments (whose fraction is that of the air's velocity plus its own), is corrected exactly
##### only where the vertical advection is fully explicit.
#####

prepare_post_loop_scalar_tendencies!(ts::MultirateInfinitesimalStep, model) =
    combine_scalar_mass_fluxes!(ts.transport_mass_fluxes, ts, model)

combine_scalar_mass_fluxes!(::Nothing, ts, model) = nothing

# The sums Aₖ = Σⱼ Wₖⱼ Fⱼ over the stages j completed so far are kept for the stages k ≥ 3 that
# follow, in `mass_fluxes[k - 2]`, and stage k completes its Rₖ over them with its own mass flux.
# The mass fluxes depend on the loop's transport velocity alone, so this precedes the post-loop
# build. Stage 1 (R₁ = 0) only has F₁ to keep, in the storage of A₃. Stage 2 forms its R₂ from F₁
# and F₂ where the build reads it (`SecondStageCorrectionMassFlux`), and starts the sums over F₁
# once the build is done (`start_scalar_mass_flux_sums!`).
function combine_scalar_mass_fluxes!(correction, ts, model)
    i = ts.stage[]
    i == 2 && return nothing
    s = number_of_stages(ts.tableau)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    (; weights, mass_fluxes) = correction
    (; advection, ρ, U) = scalar_mass_flux_arguments(model)

    ks = later_stages(max(i, 2), s)
    wᵢ = convert(FT, weights[i, i])
    wᵏ = map(k -> convert(FT, weights[k, i]), ks)

    for (name, direction) in pairs(mass_flux_directions)
        parameters = mass_flux_parameters(grid, direction)
        if i == 1
            launch!(arch, grid, parameters, _store_scalar_mass_flux!, mass_fluxes[1][name], grid, direction, advection, ρ, U)
        else
            A = map(k -> mass_fluxes[k - 2][name], ks)
            launch!(arch, grid, parameters, _combine_scalar_mass_fluxes!,
                    mass_fluxes[i - 2][name], A, grid, direction, advection, ρ, U, wᵢ, wᵏ)
        end
    end

    return nothing
end

start_scalar_mass_flux_sums!(::Nothing, ts, model) = nothing

# After stage 2's build: Aₖ = Wₖ₁ F₁ + Wₖ₂ F₂ for the stages k ≥ 3.
function start_scalar_mass_flux_sums!(correction, ts, model)
    ts.stage[] == 2 || return nothing
    s = number_of_stages(ts.tableau)
    grid = model.grid
    FT = eltype(grid)
    (; weights, mass_fluxes) = correction
    (; advection, ρ, U) = scalar_mass_flux_arguments(model)

    ks = later_stages(2, s)
    w¹ = map(k -> convert(FT, weights[k, 1]), ks)
    w² = map(k -> convert(FT, weights[k, 2]), ks)

    for (name, direction) in pairs(mass_flux_directions)
        A = map(k -> mass_fluxes[k - 2][name], ks)
        launch!(architecture(grid), grid, mass_flux_parameters(grid, direction), _start_scalar_mass_flux_sums!,
                A, mass_fluxes[1][name], grid, direction, advection, ρ, U, w¹, w²)
    end

    return nothing
end

const mass_flux_directions = (u = Val(:x), v = Val(:y), w = Val(:z))

# What the scalars' mass flux is formed from (see `scalar_mass_flux`).
function scalar_mass_flux_arguments(model)
    moisture_name = moisture_prognostic_name(model.microphysics)
    return (advection = field_advection_scheme(model.advection, moisture_name),
            ρ = total_density(model.dynamics),
            U = transport_velocities(model))
end

# The mass flux Rᵢ of stage i's correction, i ≥ 2, as the post-loop build reads it.
function correction_mass_fluxes(correction, ts, model)
    i = ts.stage[]
    i == 2 || return correction.mass_fluxes[i - 2]
    FT = eltype(model.grid)
    (; weights, mass_fluxes) = correction
    (; advection, ρ, U) = scalar_mass_flux_arguments(model)
    return map(mass_flux_directions, mass_fluxes[1]) do direction, F₁
        SecondStageCorrectionMassFlux(direction, advection, ρ, U, F₁, convert(FT, weights[2, 2]), convert(FT, weights[2, 1]))
    end
end

# The faces whose mass flux a cell's flux divergence reads: in the flux's own direction, both faces
# of every cell. The last is a halo point of the mass flux unless the direction is bounded. The
# kernels cover it, so the mass fluxes need no halo filled: the density and the velocity are valid
# there, where the scalars' advective tendencies read them.
function mass_flux_parameters(grid, ::Val{direction}) where direction
    dimension = direction === :x ? 1 : direction === :y ? 2 : 3
    faces = ntuple(3) do n
        N = size(grid, n)
        n == dimension && topology(grid, n) !== Flat ? N + 1 : N
    end
    return KernelParameters(faces, (0, 0, 0))
end

# The area-weighted mass flux that carries the scalars' explicit advective tendency, as `div_ρUc` forms it.
@inline scalar_mass_flux(i, j, k, grid, ::Val{:x}, advection, ρ, U) =
    @inbounds Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, ρ) * U.u[i, j, k]

@inline scalar_mass_flux(i, j, k, grid, ::Val{:y}, advection, ρ, U) =
    @inbounds Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, ρ) * U.v[i, j, k]

@inline scalar_mass_flux(i, j, k, grid, ::Val{:z}, advection, ρ, U) =
    @inbounds Azᶜᶜᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, ρ) *
              explicit_vertical_fraction(i, j, k, grid, advection, U.w) * U.w[i, j, k]

# The fraction of the vertical velocity `w` that `advection` transports explicitly.
@inline explicit_vertical_fraction(i, j, k, grid, ::Nothing, w) = one(grid)

@inline function explicit_vertical_fraction(i, j, k, grid, advection, w)
    scheme = vertical_scheme(advection)
    return explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, OceananigansTimeSteppers.time_discretization(scheme), w)
end

# R₂ = W₂₂ F₂ + W₂₁ F₁ in one direction, evaluated where it is read from the stored F₁ and the
# state F₂ is formed from. Stored, it would need a field of its own next to the sums that stage 2
# starts for every later stage.
struct SecondStageCorrectionMassFlux{D, A, R, V, F, T}
    direction :: D
    advection :: A
    density :: R
    velocities :: V
    first_mass_flux :: F # F₁
    weight :: T          # W₂₂
    first_weight :: T    # W₂₁
end

Adapt.adapt_structure(to, R::SecondStageCorrectionMassFlux) =
    SecondStageCorrectionMassFlux(R.direction, adapt(to, R.advection), adapt(to, R.density), adapt(to, R.velocities),
                                  adapt(to, R.first_mass_flux), adapt(to, R.weight), adapt(to, R.first_weight))

@inline correction_mass_flux(i, j, k, grid, R) = @inbounds R[i, j, k]

@inline function correction_mass_flux(i, j, k, grid, R::SecondStageCorrectionMassFlux)
    F₂ = scalar_mass_flux(i, j, k, grid, R.direction, R.advection, R.density, R.velocities)
    return R.weight * F₂ + R.first_weight * @inbounds R.first_mass_flux[i, j, k]
end

@kernel function _store_scalar_mass_flux!(F₁, grid, direction, advection, ρ, U)
    i, j, k = @index(Global, NTuple)
    @inbounds F₁[i, j, k] = scalar_mass_flux(i, j, k, grid, direction, advection, ρ, U)
end

# Aₖ = Wₖ₁ F₁ + Wₖ₂ F₂ for the stages k ≥ 3. The first of them is written over F₁.
@kernel function _start_scalar_mass_flux_sums!(A, F₁, grid, direction, advection, ρ, U, w¹, w²)
    i, j, k = @index(Global, NTuple)
    F₂ = scalar_mass_flux(i, j, k, grid, direction, advection, ρ, U)
    combine!(i, j, k, A, w¹, w², @inbounds(F₁[i, j, k]), F₂)
end

# Rᵢ = Wᵢᵢ Fᵢ + Σⱼ Wᵢⱼ Fⱼ with the sum over the earlier stages j < i, which it is written over,
# and Fᵢ added to the sums of the later stages.
@kernel function _combine_scalar_mass_fluxes!(Rᵢ, A, grid, direction, advection, ρ, U, wᵢ, wᵏ)
    i, j, k = @index(Global, NTuple)
    F = scalar_mass_flux(i, j, k, grid, direction, advection, ρ, U)
    @inbounds Rᵢ[i, j, k] = wᵢ * F + Rᵢ[i, j, k]
    accumulate!(i, j, k, A, wᵏ, F)
end

# The post-loop build adds the non-advective part of each scalar's tendency to `Gⁿ` and writes the
# advective part to the scalar's `fast_tendency` scratch (see `assemble_post_loop_stage_forcing!`),
# transported with the stage's mass-flux correction where it has one: every stage but the first of
# a coefficient set whose `β` is not diagonal.
#
# A coefficient set with a diagonal `β` recombines no tendency (`DiagonalStorage`), so its build adds
# the whole tendency to `Gⁿ`, as that of `AcousticRungeKutta3` does, with the stage's own mass flux
# alone: its correction has none (R = 0). The moisture and condensate tendencies are then copied for
# the next stage's loop before the assembly adds the increments' part of the stage forcing to them
# (`keep_previous_scalar_tendencies!`).
post_loop_scalar_tendency_kernel(ts::MultirateInfinitesimalStep, model) =
    post_loop_scalar_tendency_kernel(ts.storage, ts, model)

post_loop_scalar_tendency_kernel(::DiagonalStorage, ts, model) = add_scalar_tendency!

post_loop_scalar_tendency_kernel(::AccumulatorStorage, ts, model) =
    post_loop_scalar_tendency_kernel(ts.transport_mass_fluxes, ts, model)

post_loop_scalar_tendency_kernel(::Nothing, ts, model) = SplitScalarTendencies(ts.fast_tendency)

function post_loop_scalar_tendency_kernel(correction, ts, model)
    split = SplitScalarTendencies(ts.fast_tendency)
    R = stage_correction_mass_fluxes(correction, ts, model)
    isnothing(R) && return split # R₁ = 0
    return MassConsistentScalarTendencies(split, R)
end

# `SplitScalarTendencies` with each scalar transported by the stage's mass flux plus its correction's.
struct MassConsistentScalarTendencies{S, R}
    split :: S      # where the two parts go
    mass_flux :: R  # Rᵢ, the mass flux of the stage's correction
end

AtmosphereModels.launch_scalar_tendency!(kernel::MassConsistentScalarTendencies, arch, grid, name, Gc, args) =
    launch_mass_consistent_scalar_tendency!(advective_tendency_storage(kernel.split, name), kernel, arch, grid, Gc, args)

launch_mass_consistent_scalar_tendency!(C, kernel, arch, grid, Gc, args) =
    launch!(arch, grid, :xyz, _add_mass_consistent_scalar_tendency!, Gc, C, grid, kernel.mass_flux, args)

# A scalar with no storage for its advective part gets the non-advective part alone, as the split does.
launch_mass_consistent_scalar_tendency!(::Nothing, kernel, arch, grid, Gc, args) =
    launch_split_scalar_tendency!(nothing, arch, grid, Gc, args)

@kernel function _add_mass_consistent_scalar_tendency!(Gc, C, grid, R, args)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        C[i, j, k] = mass_consistent_advective_tendency(i, j, k, grid, R, args...)
        Gc[i, j, k] += nonadvective_scalar_tendency(i, j, k, grid, args...)
    end
end

# `AtmosphereModels.scalar_advective_tendency` with the mass flux `R` added to the one the
# velocities carry.
@inline function mass_consistent_advective_tendency(i, j, k, grid, R, c, name, advection, dynamics, velocities,
                                                    microphysics, microphysical_fields)
    Uᵖ = microphysical_velocities(microphysics, microphysical_fields, name)
    Uᵗ = sum_of_velocities(velocities, Uᵖ)
    ρ = total_density(dynamics)
    U = (u = MassConsistentVelocity(Val(:x), Uᵗ.u, R.u, ρ, grid, nothing),
         v = MassConsistentVelocity(Val(:y), Uᵗ.v, R.v, ρ, grid, nothing),
         w = MassConsistentVelocity(Val(:z), Uᵗ.w, R.w, ρ, grid, advection))
    return - div_ρUc(i, j, k, grid, advection, ρ, U, c)
end

# The same, taking the argument list of `AtmosphereModels.scalar_tendency` after `R`.
@inline mass_consistent_advective_tendency(i, j, k, grid, R, c, id, name, c_forcing, advection, dynamics, formulation,
                                           constants, specific_prognostic_moisture, velocities, microphysics,
                                           microphysical_fields, args...) =
    mass_consistent_advective_tendency(i, j, k, grid, R, c, name, advection, dynamics, velocities,
                                       microphysics, microphysical_fields)

# One component of the velocity whose mass flux is that of `velocity` plus the area-weighted mass
# flux `mass_flux` of a stage's correction, formed where the advection operator reads it, so the
# operator carries both with one reconstruction of the scalar. The vertical component is the
# explicitly transported one: the fraction of `velocity` that `advection` transports explicitly,
# as the mass fluxes of the correction are (`scalar_mass_flux`), plus the correction's, which the
# operator takes as it is (`explicit_velocity_scaleᶜᶜᶠ`).
struct MassConsistentVelocity{D, U, R, P, G, A}
    direction :: D
    velocity :: U
    mass_flux :: R
    density :: P
    grid :: G
    advection :: A # the scheme whose explicit fraction scales the vertical velocity; `nothing` in the horizontal
end

Adapt.adapt_structure(to, u::MassConsistentVelocity) =
    MassConsistentVelocity(u.direction, adapt(to, u.velocity), adapt(to, u.mass_flux), adapt(to, u.density),
                           adapt(to, u.grid), adapt(to, u.advection))

const MCV = MassConsistentVelocity

# The velocity that the area-weighted mass flux `R` is at density `ρᶠ` through the area `A`. Where
# there is no mass there is no mass flux, and no velocity.
@inline function mass_flux_velocity(R, A, ρᶠ)
    Aρ = A * ρᶠ
    return R / ifelse(Aρ == 0, one(Aρ), Aρ)
end

@inline Base.getindex(u::MCV{Val{:x}}, i, j, k) =
    @inbounds u.velocity[i, j, k] + mass_flux_velocity(correction_mass_flux(i, j, k, u.grid, u.mass_flux),
                                                       Axᶠᶜᶜ(i, j, k, u.grid), ℑxᶠᵃᵃ(i, j, k, u.grid, u.density))

@inline Base.getindex(v::MCV{Val{:y}}, i, j, k) =
    @inbounds v.velocity[i, j, k] + mass_flux_velocity(correction_mass_flux(i, j, k, v.grid, v.mass_flux),
                                                       Ayᶜᶠᶜ(i, j, k, v.grid), ℑyᵃᶠᵃ(i, j, k, v.grid, v.density))

@inline Base.getindex(w::MCV{Val{:z}}, i, j, k) =
    @inbounds explicit_vertical_fraction(i, j, k, w.grid, w.advection, w.velocity) * w.velocity[i, j, k] +
              mass_flux_velocity(correction_mass_flux(i, j, k, w.grid, w.mass_flux),
                                 Azᶜᶜᶠ(i, j, k, w.grid), ℑzᵃᵃᶠ(i, j, k, w.grid, w.density))

# The vertical component is already the explicitly transported velocity.
@inline Oceananigans.Advection.explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, ::AdaptiveVerticallyImplicitDiscretization,
                                                         ::MCV) = one(grid)

# Assemble one field's start state and forcing, its tendency split into fast and slow parts by `split`.
assemble_stage_forcing!(ts::MultirateInfinitesimalStep, model, Δt, name, split) =
    assemble_stage_forcing!(ts.storage, ts, model, Δt, name, split)

#####
##### Stage assembly with a diagonal β
#####
##### Every stage applies its own slow tendency alone, βᵢᵢ / dᵢ = 1, so the stage forcing is the full
##### tendency plus the increments' part, G = Gᵢ + Σⱼ wᵢⱼ ΔYⱼ with wᵢⱼ = γᵢⱼ / (dᵢ Δt), and the start
##### state is Uⁿ + Σⱼ αᵢⱼ ΔYⱼ. Both sums are over increments, held in registers or read from the
##### prognostic field (`StateIncrement`), and are added in the order in which `AccumulatorStorage`
##### accumulates them.
#####

# Write Σₘ wₘ aₘ to the register `target`.
struct RegisterUpdate{R, A, W}
    target :: R
    terms :: A
    weights :: W
end

Adapt.adapt_structure(to, update::RegisterUpdate) =
    RegisterUpdate(adapt(to, update.target), adapt(to, update.terms), update.weights)

# The new values of every register are formed before any is written: a register may be among the
# terms of another's update, or of its own.
@inline updated_values(i, j, k, ::Tuple{}) = ()
Base.@propagate_inbounds updated_values(i, j, k, updates::Tuple) =
    (weighted_sum(i, j, k, updates[1].terms, updates[1].weights), updated_values(i, j, k, Base.tail(updates))...)

@inline write_registers!(i, j, k, ::Tuple{}, ::Tuple{}) = nothing
Base.@propagate_inbounds function write_registers!(i, j, k, updates::Tuple, values::Tuple)
    updates[1].target[i, j, k] = values[1]
    return write_registers!(i, j, k, Base.tail(updates), Base.tail(values))
end

@kernel function _assemble_increment_forcing!(G, grid, split, terms, weights, updates)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        forcing = weighted_sum(i, j, k, terms, weights)
        values = updated_values(i, j, k, updates)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + forcing
        write_registers!(i, j, k, updates, values)
    end
end

@kernel function _update_registers!(updates)
    i, j, k = @index(Global, NTuple)
    @inbounds write_registers!(i, j, k, updates, updated_values(i, j, k, updates))
end

function assemble_stage_forcing!(storage::DiagonalStorage, ts, model, Δt, name, split)
    grid = model.grid
    G = ts.Gⁿ[name]
    source = stage_increment_source(model, name)
    terms, weights = increment_forcing(source, ts, model, Δt, name)
    updates = assembly_register_updates(storage, source, ts, model, Δt, name)

    # Nothing to do in stage 1, nor in any stage of Wicker–Skamarock RK3
    forced = !all(is_known_zero, weights)
    forced || !isempty(updates) || return nothing
    terms, weights = forced ? (terms, weights) : ((), ())

    launch!(architecture(grid), grid, stage_assembly_parameters(model, G), _assemble_increment_forcing!,
            G, grid, split, terms, weights, updates)

    return nothing
end

# The increment ΔYᵢ₋₁ of the stage before the current one, read from the prognostic field.
prognostic_increment(source::IncrementFromState, ts, model, name) =
    StateIncrement(source.u, ts.Uⁿ[name], stage_residual(ts.residual, model, name))

forcing_weight(tableau, Δt, i, j) = tableau.γ[i, j] / (tableau.d[i] * Δt)

# Whether a stage weight is known to be zero, so its term can be skipped. A weight that holds a traced
# time step (Reactant) is not known, and its term is added with whatever value it has.
is_known_zero(w) = iszero(w)

# The increments that force field `name` in the current stage and their weights wᵢⱼ, as the
# registers hold them when the stage is assembled.
increment_forcing(source, ts, model, Δt, name) = increment_forcing(ts.storage, source, ts, model, Δt, name)

function increment_forcing(::TwoRegisterStorage, source, ts, model, Δt, name)
    i = ts.stage[]
    i == 1 && return (), ()
    return (prognostic_increment(source, ts, model, name),), (forcing_weight(ts.tableau, Δt, i, i - 1),)
end

# A field that forms its increments in the assembly: ΔY₁ to ΔYᵢ₋₂ are in the registers and ΔYᵢ₋₁
# in the prognostic field, except in stage 5, where the second register holds the forcing but for
# the part of ΔY₄ (see `assembly_register_updates`).
function increment_forcing(::IncrementStorage, source::IncrementFromState, ts, model, Δt, name)
    i = ts.stage[]
    i == 1 && return (), ()
    R = increment_registers(ts, name)
    ΔY = prognostic_increment(source, ts, model, name)
    i == 5 && return (R[2], ΔY), (one(Δt), forcing_weight(ts.tableau, Δt, 5, 4))
    js = ntuple(identity, i - 1)
    return (ntuple(j -> R[j], i - 2)..., ΔY), map(j -> forcing_weight(ts.tableau, Δt, i, j), js)
end

# A field whose increments are stored: ΔY₁ to ΔYᵢ₋₁ are in the registers, except in the last stage,
# whose forcing `store_stage_increment!` has left in the second.
function increment_forcing(::IncrementStorage, ::StoredIncrement, ts, model, Δt, name)
    i = ts.stage[]
    i == 1 && return (), ()
    R = increment_registers(ts, name)
    i == number_of_stages(ts.tableau) && return (R[2],), (one(Δt),)
    js = ntuple(identity, i - 1)
    return map(j -> R[j], js), map(j -> forcing_weight(ts.tableau, Δt, i, j), js)
end

# What the assembly of the current stage writes to the registers of a field that forms its
# increments there (see `increment_registers`).
assembly_register_updates(storage, source, ts, model, Δt, name) = ()

function assembly_register_updates(::IncrementStorage, source::IncrementFromState, ts, model, Δt, name)
    i = ts.stage[]
    i == 1 && return ()
    tableau = ts.tableau
    α = tableau.α
    s = number_of_stages(tableau)
    R = increment_registers(ts, name)
    ΔY = prognostic_increment(source, ts, model, name)
    held = (ntuple(j -> R[j], min(i - 2, 2))..., ΔY)
    js = ntuple(identity, i - 1)

    if i == 5
        # The offset of stage 5, completed with ΔY₄
        return (RegisterUpdate(R[1], (R[1], ΔY), (one(Δt), α[5, 4])),)
    elseif i == s
        # The offset of the last stage, over ΔY₁
        return (RegisterUpdate(R[1], held, map(j -> α[s, j], js)),)
    elseif i == 4
        # Of five stages: the offset and the forcing of stage 5 but for the parts of ΔY₄, over ΔY₁ and ΔY₂
        return (RegisterUpdate(R[1], held, map(j -> α[5, j], js)),
                RegisterUpdate(R[2], held, map(j -> forcing_weight(tableau, Δt, 5, j), js)))
    else
        # ΔYᵢ₋₁, in the next free register
        return (RegisterUpdate(R[i - 1], (ΔY,), (one(Δt),)),)
    end
end

# After stage i and its `update_state!`, for a field whose increments are stored: ΔYᵢ in the next free register, or, after
# stage s - 1, the offset and the forcing of the last stage over ΔY₁ and ΔY₂.
function store_stage_increment!(::IncrementStorage, ts, model, i, Δt)
    tableau = ts.tableau
    α = tableau.α
    s = number_of_stages(tableau)
    grid = model.grid

    for (name, u) in pairs(prognostic_fields(model))
        stage_increment_source(model, name) isa StoredIncrement || continue
        R = increment_registers(ts, name)
        ΔY = StateIncrement(u, ts.Uⁿ[name], stage_residual(ts.residual, model, name))

        updates = if i == s - 1
            held = (ntuple(j -> R[j], s - 2)..., ΔY)
            js = ntuple(identity, s - 1)
            (RegisterUpdate(R[1], held, map(j -> α[s, j], js)),
             RegisterUpdate(R[2], held, map(j -> forcing_weight(tableau, Δt, s, j), js)))
        else
            (RegisterUpdate(R[i], (ΔY,), (one(Δt),)),)
        end

        launch!(architecture(grid), grid, stage_assembly_parameters(model, u), _update_registers!, updates)
    end

    return nothing
end

store_stage_increment!(::TwoRegisterStorage, ts, model, i, Δt) = nothing

#####
##### Stage assembly with accumulators
#####

function assemble_stage_forcing!(::AccumulatorStorage, ts, model, Δt, name, split)
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

# The increment ΔYᵢ₋₁ = Yᵢ₋₁ - Uⁿ of every field is that of the state `update_state!` leaves after
# stage i - 1: the state the next slow tendencies are evaluated at, with the negative-moisture
# correction of the moisture and microphysical fields. The acoustic prognostics still hold that
# state when the next stage is assembled, so the assembly forms their increment. Those of the other
# fields are stored once `update_state!` has run (`store_stage_increment!`): with post-loop scalar
# tendencies the loop's pressure needs their part of the stage forcing before the loop
# (`PredictedStageForcing`).
stage_increment_source(model, name) = stage_increment_source(model.timestepper.storage, model, name)

function stage_increment_source(storage, model, name)
    folded = model.timestepper.fold_stage_increments && name ∈ acoustic_prognostic_names(model)
    return folded ? IncrementFromState(prognostic_fields(model)[name]) : StoredIncrement()
end

# With two registers there is nowhere to store an increment: every field forms its own from the
# prognostic field, which holds the state `update_state!` left until the stage updates it.
stage_increment_source(::TwoRegisterStorage, model, name) = IncrementFromState(prognostic_fields(model)[name])

# Fold the increment ΔYᵢ = Yᵢ - Uⁿ of the stage just completed into the later stages' accumulators,
# with Yᵢ the state `update_state!` has left.
store_stage_increment!(ts::MultirateInfinitesimalStep, model, i, Δt) =
    store_stage_increment!(ts.storage, ts, model, i, convert(eltype(model.grid), Δt))

function store_stage_increment!(::AccumulatorStorage, ts, model, i, Δt)
    (; α, γ, d) = ts.tableau
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
            ts.next_stage[] = i + 1
            Δc = i == 1 ? c[1] : c[i] - c[i-1]
            tick_stage!(model.clock, Δc * Δt)
            update_state!(model, callbacks; compute_tendencies = true)
            # After `update_state!`, so that the increments are those of the state it leaves,
            # with its negative-moisture correction (see `stage_increment_source`).
            store_stage_increment!(ts, model, i, Δt)
            cache_transport_velocity!(model)
        end
    end

    ts.next_stage[] = 1
    complete_acoustic_time_step!(model, Δt, tⁿ⁺¹, callbacks)

    return nothing
end
