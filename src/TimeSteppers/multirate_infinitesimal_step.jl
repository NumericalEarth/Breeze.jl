using Breeze.AtmosphereModels: thermodynamic_density_name, dynamics_prognostic_fields, SplitScalarTendencies
using Breeze.CompressibleEquations: compute_linearized_thermodynamic_flux_divergence!,
                                    StageEntryForceRemainder, stage_entry_force_remainder,
                                    step_start_force_state, store_step_start_force_state!,
                                    compute_step_start_moisture_content!

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
evaluated at ``Yⱼ₋₁``. The fast tendency is linearized once per step, about ``Uⁿ``.

The slow tendency excludes everything the acoustic loop integrates: momentum tendencies never
contain the pressure gradient or buoyancy; the continuity tendency ``-∇·(ρu)`` is entirely fast;
and the thermodynamic slow tendency is the full tendency minus ``-∇·(θ (ρu))`` in the loop's
discretization (see `compute_linearized_thermodynamic_flux_divergence!`). With
the `:WickerSkamarock3` coefficients the scheme reduces to `AcousticRungeKutta3`.

Fields
======

- `tableau`: the [`MultirateInfinitesimalStepTableau`](@ref).
- `U⁰`: start state of the acoustic integration of stages 2 to ``s``. Stage 1 starts from `Uⁿ` itself
  (see `stage_start_state`).
- `Gⁿ`: tendency storage; holds the combined slow forcing during each acoustic loop.
- `implicit_solver`, `substepper`: as for `AcousticRungeKutta3`.
- `rounding`: Round-off treatment of the final stage's update, [`RoundToNearest`](@ref)
  or [`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding).
- `Uⁿ`: the outer-step-start state.
- `accumulators`: for each stage ``k = 3, …, s``, the parts of its forcing and start state known so far,
  `forcing` ``= (1/dₖ) Σⱼ (βₖⱼ Sⱼ + γₖⱼ ΔYⱼ / Δt)`` and `offset` ``= Σⱼ αₖⱼ ΔYⱼ``, accumulated as each
  ``Sⱼ`` and ``ΔYⱼ = Yⱼ - Uⁿ`` becomes available, so no ``Sⱼ`` or ``ΔYⱼ`` is stored. Until stage 2
  consumes them, stage 3's pair holds the raw ``S₁`` and, for the moisture, microphysical and tracer
  fields, the raw ``ΔY₁`` (see `stage_increment_source`).
- `fast_tendency`: scratch for the fast parts of the tendencies: with `PreLoopScalarTendencies`
  one field, for the stage-entry fast thermodynamic tendency; with `PostLoopScalarTendencies`
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
- `step_start`: the outer-step-start pressure, thermodynamic density and water density that the
  slow remainder of the stage-entry pressure gradient and buoyancy is measured from, or `nothing`
  where that remainder is not recombined (see `step_start_storage`).
- `scalar_tendencies`: when the moisture and tracer tendencies are built,
  `PreLoopScalarTendencies` (between stages, the default) or
  `PostLoopScalarTendencies` (after each stage's own acoustic loop). With the latter, the
  scalars' slow tendency ``Sᵢ`` stored for stage ``i`` is the post-loop one, and their stage forcing
  and start state are assembled after the loop.
"""
struct MultirateInfinitesimalStep{TB, U0, TG, TI, AS, R, UN, AC, FC, ST, SS, TA} <: AcousticTimeStepper
    relinearize_stages :: Bool
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
    transport_density_accumulators :: TA # stages 2 to s: the parts known so far of the transport density tendency's stage combination
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
the advection operator. Of the other sets, `:TVDMISA` and `:TVDMISB` have ``r = 0.75``, `:MIS3C`
``r = 0.23``, and `:MIS4a`, `:MIS4`, `:MIS2` and `:WickerSkamarock3` have ``r = 0``: no such bound
at any Courant number. The bound is on the time stepping alone. Positive moisture also needs an
advection operator that is positive under forward Euler, which the clamp of bounded WENO
reconstructions does not provide by itself; a flux limiter would (Oceananigans.jl#6475).

`:SSPMIS43` has four stages and is second-order accurate with the fast and slow parts coupled. Its
first stage covers 0.56 of the step, which takes at least two acoustic substeps. It needs the
moisture's pressure effect in the acoustic loop to stay robust in moist convection: with the
stage-entry pressure remainder recombined across stages, the part of that remainder the loop does
not carry is integrated explicitly. `:MIS4a` is third order and stable to a larger Courant number,
without the positivity bound. `:MIS2` relies on the dissipation of third-order upwinding and is
unstable beyond a Courant number of about 0.2 with fifth-order schemes.

`post_loop_scalar_tendencies = true` rebuilds the moisture and tracer tendencies after each
stage's own acoustic loop, with that loop's time-averaged transport velocity, instead of
between stages with the preceding loop's (see `PostLoopScalarTendencies`).

`rounding` sets the round-off treatment of the step's final update (default:
[`RoundToNearest`](@ref)), as for `AcousticRungeKutta3`: with
[`StochasticRounding`](@ref Breeze.RoundingTreatments.StochasticRounding), the last stage's
recovery of the acoustic prognostics from their stage-start values plus the accumulated
perturbations, and its update ``u⁰ + Δt G`` of the other scalars, are rounded stochastically.
The perturbations themselves, the earlier stages, and the vertically-implicit solves stay
round-to-nearest.
"""
function MultirateInfinitesimalStep(grid, prognostic_fields;
                                    dynamics,
                                    coefficients = :SSPMIS43,
                                    implicit_solver = nothing,
                                    cache_advecting_state = false,
                                    relinearize_stages = false,
                                    post_loop_scalar_tendencies = false,
                                    rounding = RoundToNearest())

    FT = eltype(grid)
    rounding = materialize_rounding(rounding, grid.architecture)
    tableau = multirate_infinitesimal_step_tableau(coefficients, FT)
    s = number_of_stages(tableau)
    s ≥ 3 || throw(ArgumentError("MultirateInfinitesimalStep needs at least three stages; got $s."))

    substepper = build_acoustic_substepper(grid, dynamics, prognostic_fields; cache_advecting_state)

    U⁰ = map(similar, prognostic_fields)
    Gⁿ = map(similar, prognostic_fields)
    Uⁿ = map(similar, prognostic_fields)
    accumulators = ntuple(_ -> (forcing = map(similar, prognostic_fields),
                                offset = map(similar, prognostic_fields)), s - 2)
    scalar_tendencies = scalar_tendency_timing(post_loop_scalar_tendencies)
    fast_tendency = fast_tendency_storage(scalar_tendencies, grid, prognostic_fields, dynamics)

    step_start = step_start_storage(grid, dynamics, tableau, relinearize_stages)

    if relinearize_stages && !is_diagonal(tableau.β)
        @warn "relinearize_stages = true with a non-diagonal tableau is first-order accurate in time: " *
              "the slow tendencies of earlier stages were split against a different fast operator."
    end

    return MultirateInfinitesimalStep(relinearize_stages, tableau, U⁰, Gⁿ, implicit_solver, substepper,
                                      rounding, Uⁿ, accumulators, fast_tendency, scalar_tendencies, step_start,
                                      ntuple(_ -> CenterField(grid), s - 1),
                                      Ref(1), Ref(1), Ref(NaN))
end

# Scratch for the fast parts of the tendencies (see the `fast_tendency` field). Post-loop
# scalars each need their own: their advective tendencies are all in hand when the
# microphysical sources and the `TendencyCallsite` callbacks complete the slow parts, before
# any scalar is assembled. The thermodynamic variable is not told apart from the scalars here,
# where only the dynamics is known, so it also gets one, which the stage uses before the loop.
fast_tendency_storage(::PreLoopScalarTendencies, grid, prognostic_fields, dynamics) = CenterField(grid)

function fast_tendency_storage(::PostLoopScalarTendencies, grid, prognostic_fields, dynamics)
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

# Whether the next `update_state!` builds the tendencies of the first stage of an outer step.
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

# `nothing` when the stage-entry force remainder is not recombined: with a diagonal `β`, with
# `relinearize_stages`, or with terrain-following coordinates.
# TODO: recombine the terrain-following remainder, with the metric terms of
# `terrain_x_pressure_gradient` and the contravariant vertical force.
function step_start_storage(grid, dynamics, tableau, relinearize_stages)
    recombined = !is_diagonal(tableau.β) && !relinearize_stages && !(dynamics isa TerrainCompressibleDynamics)
    return recombined ? step_start_force_state(grid) : nothing
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

@inline assign!(i, j, k, ::Tuple{}, ::Tuple{}, x) = nothing
@inline function assign!(i, j, k, A::Tuple, w::Tuple, x)
    @inbounds A[1][i, j, k] = w[1] * x
    return assign!(i, j, k, Base.tail(A), Base.tail(w), x)
end

# The stage forcing is
#   G = (fast part at Yᵢ₋₁) + (1/dᵢ)[βᵢᵢ Sᵢ + Σⱼ (βᵢⱼ Sⱼ + γᵢⱼ ΔYⱼ / Δt)] = Gᵢ + (βᵢᵢ/dᵢ - 1) Sᵢ + Fᵢ,
# with Gᵢ the full tendency (`full_tendency`), written so a diagonal tableau (Wicker–Skamarock)
# leaves Gᵢ bitwise intact, and the start state is U⁰ = Uⁿ + Zᵢ with Zᵢ = Σⱼ αᵢⱼ ΔYⱼ. Each
# kernel makes one pass over a field. These kernels do no arithmetic to speak of: their cost is
# the memory they stream, so they cover only the points that are read afterwards
# (`stage_assembly_parameters`) and touch each array once.

# Stage 1: F₁ = Z₁ = 0, so the start state is Uⁿ itself (`stage_start_state`) and nothing is
# written to U⁰. Keep S₁ (in stage 3's forcing accumulator) for later stages.
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

# ΔY₁, for stage 2: `Z[1]` is where `store_stage_increment!` keeps it raw.
@inline first_increment(source::IncrementFromState, Z, uⁿ, i, j, k) = @inbounds source.u[i, j, k] - uⁿ
@inline first_increment(::StoredIncrement, Z, uⁿ, i, j, k) = @inbounds Z[1][i, j, k]

# Stage i's own accumulators, completed with ΔYᵢ₋₁ in registers (they are not needed again) when
# it has not been folded in already.
@inline function completed_accumulators(source::IncrementFromState, Fᵢ, Zᵢ, wγᵢ, αᵢ, uⁿ, i, j, k)
    ΔY = @inbounds source.u[i, j, k] - uⁿ
    return @inbounds (Fᵢ[i, j, k] + wγᵢ * ΔY, Zᵢ[i, j, k] + αᵢ * ΔY)
end

@inline completed_accumulators(::StoredIncrement, Fᵢ, Zᵢ, wγᵢ, αᵢ, uⁿ, i, j, k) = @inbounds (Fᵢ[i, j, k], Zᵢ[i, j, k])

# Add Sᵢ, and ΔYᵢ₋₁ when it has not been folded in already, to the accumulators of later stages.
@inline function accumulate_later_stages!(source::IncrementFromState, F, Z, wγᵏ, wβᵏ, αᵏ, slow, uⁿ, i, j, k)
    ΔY = @inbounds source.u[i, j, k] - uⁿ
    accumulate!(i, j, k, F, wγᵏ, ΔY, wβᵏ, slow)
    accumulate!(i, j, k, Z, αᵏ, ΔY)
    return nothing
end

@inline accumulate_later_stages!(::StoredIncrement, F, Z, wγᵏ, wβᵏ, αᵏ, slow, uⁿ, i, j, k) =
    accumulate!(i, j, k, F, wβᵏ, slow)

# Stage 2 reads the raw S₁ and turns it, with ΔY₁ and S₂, into the accumulators of stages k ≥ 3
# (whose first pair is where S₁ and a stored ΔY₁ were kept: they are read before being overwritten).
@kernel function _second_multirate_stage!(G, grid, U⁰, Uⁿ, source, split, wᵢ, wβ₁, wγ₁, α₁, F, Z, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, α₁ᵏ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        S₁ = F[1][i, j, k]
        uⁿ = Uⁿ[i, j, k]
        ΔY₁ = first_increment(source, Z, uⁿ, i, j, k)
        S₂ = slow_part(i, j, k, grid, split, G)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * S₂ + wβ₁ * S₁ + wγ₁ * ΔY₁
        U⁰[i, j, k] = uⁿ + α₁ * ΔY₁
        combine!(i, j, k, F, wβ₁ᵏ, wγ₁ᵏ, wβ₂ᵏ, S₁, ΔY₁, S₂)
        assign!(i, j, k, Z, α₁ᵏ, ΔY₁)
    end
end

# Stage i ≥ 3 uses its completed accumulators and adds Sᵢ to those of later stages.
@kernel function _later_multirate_stage!(G, grid, U⁰, Uⁿ, source, split, wᵢ, Fᵢ, Zᵢ, wγᵢ, αᵢ, F, Z, wγᵏ, wβᵏ, αᵏ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        uⁿ = Uⁿ[i, j, k]
        slow = slow_part(i, j, k, grid, split, G)
        forcing, offset = completed_accumulators(source, Fᵢ, Zᵢ, wγᵢ, αᵢ, uⁿ, i, j, k)
        G[i, j, k] = full_tendency(i, j, k, grid, split, G) + wᵢ * slow + forcing
        U⁰[i, j, k] = uⁿ + offset
        accumulate_later_stages!(source, F, Z, wγᵏ, wβᵏ, αᵏ, slow, uⁿ, i, j, k)
    end
end

# After stage i, for the fields whose increment is stored: ΔYᵢ = Yᵢ - Uⁿ, kept raw after stage 1
# and accumulated after later stages.
@kernel function _store_first_increment!(ΔY₁, u, Uⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔY₁[i, j, k] = u[i, j, k] - Uⁿ[i, j, k]
end

@kernel function _accumulate_increment!(F, Z, u, Uⁿ, wγᵏ, αᵏ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ΔY = u[i, j, k] - Uⁿ[i, j, k]
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

# `Uⁿ` in stage 1 and `U⁰ = Uⁿ + Zᵢ` after that. Both are read-only during a stage.
stage_start_state(ts::MultirateInfinitesimalStep) = ts.stage[] == 1 ? ts.Uⁿ : ts.U⁰

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
    assemble_scalar_start_states!(ts.scalar_tendencies, ts, model)
    compute_step_start_moisture_content!(ts.step_start, ts.substepper, model, ts.Uⁿ)
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

# The loop's pressure reads the start state of the moisture and condensate, so with
# `PostLoopScalarTendencies` the scalars' start states U⁰ = Uⁿ + Zᵢ are written before the loop.
assemble_scalar_start_states!(::PreLoopScalarTendencies, ts, model) = nothing

function assemble_scalar_start_states!(::PostLoopScalarTendencies, ts::MultirateInfinitesimalStep, model)
    i = ts.stage[]
    i == 1 && return nothing # stage 1 starts from Uⁿ itself (see `stage_start_state`)

    grid = model.grid
    # `accumulators[1].offset` holds the raw ΔY₁ in stage 2, and Zᵢ is in `accumulators[i - 2]` after that.
    αᵢ = i == 2 ? ts.tableau.α[2, 1] : one(eltype(ts.tableau.d))
    offsets = ts.accumulators[max(i - 2, 1)].offset

    for name in scalar_prognostic_names(model)
        stage_increment_source(model, name) isa StoredIncrement || continue
        U⁰ = ts.U⁰[name]
        launch!(architecture(grid), grid, stage_assembly_parameters(model, U⁰), _assemble_start_state!,
                U⁰, ts.Uⁿ[name], offsets[name], αᵢ)
    end

    return nothing
end

@kernel function _assemble_start_state!(U⁰, Uⁿ, Z, α)
    i, j, k = @index(Global, NTuple)
    @inbounds U⁰[i, j, k] = Uⁿ[i, j, k] + α * Z[i, j, k]
end

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
    U⁰ = ts.U⁰[name]
    Uⁿ = ts.Uⁿ[name]
    source = stage_increment_source(model, name)
    F = accumulator_fields(ts, name, ks, :forcing)
    parameters = stage_assembly_parameters(model, G)

    if i == 1
        launch!(arch, grid, parameters, _first_multirate_stage!, G, grid, split, wᵢ,
                ts.accumulators[1].forcing[name])
    elseif i == 2
        Z = accumulator_fields(ts, name, ks, :offset)
        launch!(arch, grid, parameters, _second_multirate_stage!, G, grid, U⁰, Uⁿ, source, split, wᵢ,
                β[2, 1] / d[2], γ[2, 1] / (d[2] * Δt), α[2, 1], F, Z,
                forcing_weights(β, d, 1, ks), increment_weights(γ, d, Δt, 1, ks),
                forcing_weights(β, d, 2, ks), offset_weights(α, 1, ks))
    else
        Fᵢ = ts.accumulators[i - 2].forcing[name]
        Zᵢ = ts.accumulators[i - 2].offset[name]
        Z = accumulator_fields(ts, name, ks, :offset)
        launch!(arch, grid, parameters, _later_multirate_stage!, G, grid, U⁰, Uⁿ, source, split, wᵢ,
                Fᵢ, Zᵢ, γ[i, i - 1] / (d[i] * Δt), α[i, i - 1], F, Z,
                increment_weights(γ, d, Δt, i - 1, ks), forcing_weights(β, d, i, ks),
                offset_weights(α, i - 1, ks))
    end

    return nothing
end

# Form the increments of the acoustic prognostics in the next stage's assembly, which reads `Uⁿ`
# anyway, instead of storing them when a stage completes. Off, every field's increment is stored.
const fold_stage_increments = Ref(true)

# The acoustic prognostics still hold Yᵢ₋₁ when the next stage is assembled, so the assembly forms
# their increment. `update_state!` changes the other scalars in between, so theirs is stored.
function stage_increment_source(model, name)
    folded = fold_stage_increments[] && name ∈ acoustic_prognostic_names(model)
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
        if i == 1
            launch!(arch, grid, parameters, _store_first_increment!,
                    ts.accumulators[1].offset[name], u, Uⁿ)
        else
            launch!(arch, grid, parameters, _accumulate_increment!,
                    accumulator_fields(ts, name, ks, :forcing), accumulator_fields(ts, name, ks, :offset),
                    u, Uⁿ, increment_weights(γ, d, Δt, i, ks), offset_weights(α, i, ks))
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
        # Only the last stage's rounding reaches Uⁿ⁺¹ directly, and only that stage takes the
        # time stepper's round-off treatment (as in `AcousticRungeKutta3`).
        rounding = i == s ? ts.rounding : RoundToNearest()
        acoustic_stage!(model, Δt, d[i], rounding, callbacks)

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
