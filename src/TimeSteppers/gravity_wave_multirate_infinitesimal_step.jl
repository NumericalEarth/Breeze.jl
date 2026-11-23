using Oceananigans: TendencyCallsite
using Breeze.HydrostaticEquations: HydrostaticEquations, HydrostaticDynamics, GravityWaveSubstepper,
                                   VerletSubsteps, gravity_wave_prognostic_names,
                                   freeze_gravity_wave_linearization!, subtract_fast_tendencies!,
                                   gravity_wave_substep_loop!, stage_substeps, maximum_gravity_wave_speed,
                                   compute_integrated_vertical_velocity!, add_perturbation_transport!,
                                   gravity_wave_seam_budgets, correct_completed_stage!,
                                   convex_stage_seam_budgets, record_stage_seam_budgets!,
                                   compute_stage_transport_velocities!, advance_scalar_with_stage_transport!,
                                   complete_scalar_stage_transport!,
                                   withhold_scalar_transport!

#####
##### How a stage advances the prognostic scalars that the substeps do not
#####

"""
$(TYPEDEF)

The scalars take the stages of the tableau as they are written: the increments of the earlier
stages and their full tendencies are recombined with the coefficients of the tableau, and the
transport by the velocity perturbation of the substeps is added with a centered flux of the
specific value at the start of the time step. Valid for any tableau, and not bounds preserving.
"""
struct RecombinedScalarStages end

"""
$(TYPEDEF)

Every stage transports the scalars once, after its substeps, with the mass flux that changes the
layer mass over the stage, and is a convex combination of the state at the start of the time
step and of one forward Euler step from the state the stage started from. With a transport
scheme that keeps a mixing ratio within bounds over a forward Euler step, the scalars stay within
them. For moist `HydrostaticDynamics`, with a tableau for which [`convex_scalar_stages`](@ref)
holds.
"""
struct ConvexScalarStages end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Whether every stage of `tableau`, for a variable without a fast tendency, is a convex combination
of the state at the start of the time step and of one forward Euler step from the stage before,

```math
Yᵢ = (1 - wᵢ) Uⁿ + wᵢ [Yᵢ₋₁ + (βᵢᵢ / wᵢ) Δt S(Yᵢ₋₁)] ,  \\qquad wᵢ = αᵢ,ᵢ₋₁ + γᵢ,ᵢ₋₁ ∈ (0, 1] :
```

`β` is diagonal and positive, and `α + γ` vanishes off the first subdiagonal. The
`:ConvexLinearMIS` sets are of this kind; `:WickerSkamarock3`, whose stages restart from ``Uⁿ``
(``wᵢ = 0``), is not.
"""
function convex_scalar_stages(tableau::MultirateInfinitesimalStepTableau)
    (; α, β, γ) = tableau
    stages = number_of_stages(tableau)

    # The structure that the multirate infinitesimal step stores with two registers
    stage_storage(tableau) isa TwoRegisterStorage || return false

    for i in 1:stages
        β[i, i] > 0 || return false
        i > 1 && (0 < α[i, i-1] + γ[i, i-1] ≤ 1 || return false)
    end

    return true
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the length of the longest forward Euler step, as a fraction of the time step, of which the
stages of `tableau` are convex combinations for a variable without a fast tendency: the largest
``βᵢᵢ / wᵢ`` (see [`convex_scalar_stages`](@ref), which must hold). A transport scheme that
preserves bounds up to a Courant number ``C`` over a forward Euler step preserves them over the
time step ``Δt`` when the Courant number of this fraction of ``Δt`` is at most ``C``.
"""
function largest_euler_step_fraction(tableau::MultirateInfinitesimalStepTableau)
    (; α, β, γ) = tableau
    fractions = [i == 1 ? β[1, 1] : β[i, i] / (α[i, i-1] + γ[i, i-1]) for i in 1:number_of_stages(tableau)]
    return maximum(fractions)
end

# The Knoth–Wensch MIS3C coefficient set, which the acoustic `MultirateInfinitesimalStep` no longer lists
# (transcribed from ClimaTimeSteppers.jl's `mis.jl`, as the sets of multirate_infinitesimal_step_coefficients.jl were)
function multirate_infinitesimal_step_tableau(::Val{:MIS3C}, FT)
    α = [0                0                0
         0.589557277145   0                0
         0.544036601551   0.565511042564   0]
    β = [ 0.397525189225    0                0
         -0.227036463644    0.624528794618   0
         -0.00295238076840 -0.270971764284   0.671323159437]
    γ = [ 0                 0                 0
          0.142798786398    0                 0
         -0.0428918957402   0.0202720980282   0]
    return MultirateInfinitesimalStepTableau(FT, α, β, γ)
end

"""
$(TYPEDEF)

Split-explicit time stepper for [`HydrostaticDynamics`](@ref Breeze.HydrostaticEquations.HydrostaticDynamics):
a multirate infinitesimal step (MIS) method ([Knoth and Wensch 2014](@cite KnothWensch2014))
whose fast system is the linear gravity-wave operator of the hydrostatic equations, linearized
about the state of each column at the start of the time step and integrated by substeps.

With ``Uⁿ`` the state at the start of the step, ``ΔY_j = Y_j - Uⁿ`` the increment of stage
``j``, ``L`` the gravity-wave operator and ``G`` the full tendency, stage ``i`` integrates

```math
∂_τ U′ = L U′ + \\frac{1}{d_i} \\left [ \\sum_{j < i} γ_{ij} \\frac{ΔY_j}{Δt} + \\sum_{j ≤ i} β_{ij} S_j \\right ] ,
\\qquad U′(0) = \\sum_{j < i} α_{ij} ΔY_j ,
```

over ``d_i Δt``, with the slow tendency ``S_j = G(Y_{j-1}) - L ΔY_{j-1}``, and sets
``ΔY_i = U′(d_i Δt)``. The full tendency is the one `:SSPRungeKutta3` integrates and the
operator ``L`` subtracted from it is the one the substeps integrate, so the split scheme
discretizes the same equations as the unsplit one.

The prognostic scalars that the gravity waves do not involve (the tracers) take the same
stages, with ``L`` their transport by the mass fluxes of the velocity perturbation. Since they
do not act on the waves, that transport is applied once per stage, with the mass fluxes
accumulated over the substeps: the fluxes that change the layer mass are the fluxes that carry
the tracers, and a uniform mixing ratio stays uniform.

Fields
======

- `tableau`: the [`MultirateInfinitesimalStepTableau`](@ref) of ``α``, ``β`` and ``γ``.
- `U⁰`: the state at the start of the time step.
- `Gⁿ`: the full tendencies, recomputed for every stage.
- `slow_tendencies`: ``S_j`` of every stage.
- `stage_increments`: ``ΔY_j`` of every stage.
- `implicit_solver`: unused; `HydrostaticDynamics` has no vertically implicit terms.
- `substepper`: the `GravityWaveSubstepper` that holds the linearization and integrates the fast system.
- `storage`: the storage layout of the coefficient set, chosen as for the acoustic
  `MultirateInfinitesimalStep` (see `stage_storage`). With `TwoRegisterStorage` one slow tendency
  and two increments are stored; otherwise one of each per stage.
- `scalar_stages`: how the stages advance the other scalars, [`RecombinedScalarStages`](@ref) or
  [`ConvexScalarStages`](@ref).
"""
struct GravityWaveMultirateInfinitesimalStep{TB, U0, TG, SS, DY, TI, GS, SL, SC} <: AbstractTimeStepper
    tableau :: TB
    U⁰ :: U0
    Gⁿ :: TG
    slow_tendencies :: SS
    stage_increments :: DY
    implicit_solver :: TI
    substepper :: GS
    storage :: SL
    scalar_stages :: SC
end

@regional GravityWaveMultirateInfinitesimalStep

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a `GravityWaveMultirateInfinitesimalStep` time stepper for
[`HydrostaticDynamics`](@ref Breeze.HydrostaticEquations.HydrostaticDynamics).

Neither the outer scheme nor the inner solver has a default: no combination is established as
the one to use, and both must be chosen.

Keyword arguments
=================

- `tableau`: the outer scheme, `:MIS3C` or `:MIS4a`
  ([Knoth and Wensch 2014](@cite KnothWensch2014)), `:WickerSkamarock3`
  ([Wicker and Skamarock 2002](@cite WickerSkamarock2002)), or one of the convex sets
  `:ConvexLinearMIS3` to `:ConvexLinearMIS6`; or a [`MultirateInfinitesimalStepTableau`](@ref).
  Required. With moist `HydrostaticDynamics` and a tableau for which
  [`convex_scalar_stages`](@ref) holds, every stage transports the water and the tracers once,
  with the mass flux of the stage, as a convex combination of forward Euler steps
  ([`ConvexScalarStages`](@ref)): a bounds-preserving transport scheme then keeps the water
  non-negative. With any other tableau the scalar update is not bounds preserving.
- `inner_solver`: the integrator of the gravity waves,
  [`VerletSubsteps`](@ref Breeze.HydrostaticEquations.VerletSubsteps)`()` or
  [`ForwardBackwardSubsteps`](@ref Breeze.HydrostaticEquations.ForwardBackwardSubsteps)`(divergence_damping=0.1)`. Required.
- `substeps`: number of substeps in a time step; a stage that covers the fraction ``d`` of it
  takes ``⌈d N⌉``. With `nothing`, the default, every stage takes the smallest number that keeps
  its Courant number below `cfl`.
- `moist_gravity_wave_operator`: with moist `HydrostaticDynamics`, integrate the gravity-wave
  operator with the coefficients of moist air frozen at the start of the step (the density
  potential temperature and the response of the total pressure to the dry-mass surface
  pressure). Required (`true`) for a model with microphysics, which is rejected otherwise: the
  split scheme integrates the equations of `:SSPRungeKutta3`, but its stability in cloud, where
  the slow tendency carries the latent heating of the waves, is measured and not derived.
  Default: `false`. Without water it changes nothing.
- `cfl`: the stage Courant number ``ω Δτ`` aimed at, with ``ω`` an upper bound of the largest
  gravity-wave frequency. Default: 0.7 times the stability bound of the inner solver, which is
  ``2 / \\sqrt{1 + 2 βᵈ}`` for forward-backward substeps with divergence damping ``βᵈ`` and 2
  for Verlet substeps. A stage whose Courant number exceeds the bound is an error.

```jldoctest
using Oceananigans, Breeze
using Breeze.HydrostaticEquations: VerletSubsteps

grid = RectilinearGrid(size=(8, 4), x=(0, 8e5), z=(0, 1), topology=(Periodic, Flat, Bounded))
model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(),
                              momentum_advection = VectorInvariant(),
                              timestepper = :GravityWaveMultirateInfinitesimalStep,
                              timestepper_kwargs = (tableau = :MIS3C, inner_solver = VerletSubsteps()))
model.timestepper

# output
GravityWaveMultirateInfinitesimalStep
├── stages: 3
├── inner_solver: VerletSubsteps()
├── substeps: nothing
└── cfl: 1.4
```
"""
function GravityWaveMultirateInfinitesimalStep(grid, prognostic_fields;
                                               dynamics,
                                               tableau = nothing,
                                               inner_solver = nothing,
                                               substeps = nothing,
                                               cfl = nothing,
                                               moist_gravity_wave_operator = false,
                                               implicit_solver::TI = nothing,
                                               cache_advecting_state = false, # accepted for TimeStepper-call uniformity; unused
                                               moisture_forcing_names = nothing, # accepted for TimeStepper-call uniformity; unused
                                               microphysical_names = (),         # accepted for TimeStepper-call uniformity; unused
                                               Gⁿ::TG = map(similar, prognostic_fields),
                                               U⁰::U0 = map(similar, prognostic_fields)) where {TI, TG, U0}

    dynamics isa HydrostaticDynamics ||
        throw(ArgumentError("GravityWaveMultirateInfinitesimalStep integrates HydrostaticDynamics only."))

    # With water the gravity-wave operator of the substeps takes the coefficients of moist air (the
    # density potential temperature, the response of the total pressure). What it leaves to the slow
    # tendency in cloud is not small, and no stability bound of the split scheme is derived for it,
    # so it is not chosen without being asked for.
    moist = !isnothing(dynamics.moisture)
    (moist && !moist_gravity_wave_operator) &&
        throw(ArgumentError("GravityWaveMultirateInfinitesimalStep does not support HydrostaticDynamics with " *
                            "moisture by default: the stability of the split scheme in cloud is not established. " *
                            "Use timestepper = :SSPRungeKutta3, or ask for the gravity-wave operator of moist air " *
                            "with timestepper_kwargs = (moist_gravity_wave_operator = true, ...)."))

    (isnothing(tableau) || isnothing(inner_solver)) &&
        throw(ArgumentError("GravityWaveMultirateInfinitesimalStep has no default scheme. Choose both with " *
                            "timestepper_kwargs = (tableau = :MIS3C, :MIS4a or :WickerSkamarock3, " *
                            "inner_solver = VerletSubsteps() or ForwardBackwardSubsteps(divergence_damping=0.1))."))

    # Qualified: the unqualified name is the substep form of the acoustic loop in this module
    inner_solver isa Union{HydrostaticEquations.ForwardBackwardSubsteps, VerletSubsteps} ||
        throw(ArgumentError("inner_solver must be VerletSubsteps() or ForwardBackwardSubsteps(), got $inner_solver."))

    FT = eltype(grid)
    tableau isa Symbol && !hasmethod(multirate_infinitesimal_step_tableau, Tuple{Val{tableau}, Type{FT}}) &&
        throw(ArgumentError("Unknown multirate infinitesimal step tableau :$tableau."))
    tableau = multirate_infinitesimal_step_tableau(tableau, FT)
    stages = number_of_stages(tableau)

    all(d -> d > 0, tableau.d) ||
        throw(ArgumentError("Every stage of the tableau must integrate forward in time."))

    substepper = GravityWaveSubstepper(grid, inner_solver, prognostic_fields; substeps, cfl, moist)

    storage = stage_storage(tableau)
    slow_tendencies = ntuple(_ -> map(similar, prognostic_fields), tendency_registers(storage, stages))
    stage_increments = ntuple(_ -> map(similar, prognostic_fields), increment_registers(storage, stages))

    # The convex stages transport the mixing ratio per unit dry mass of moist dynamics
    scalar_stages = moist && convex_scalar_stages(tableau) ? ConvexScalarStages() : RecombinedScalarStages()

    return GravityWaveMultirateInfinitesimalStep(tableau, U⁰, Gⁿ, slow_tendencies, stage_increments,
                                                 implicit_solver, substepper, storage, scalar_stages)
end

const ConvexGravityWaveMultirateInfinitesimalStep =
    GravityWaveMultirateInfinitesimalStep{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConvexScalarStages}

# The stages transport the scalars: their tendencies keep the forcing alone
HydrostaticEquations.leave_scalar_transport_to_stages!(timestepper::ConvexGravityWaveMultirateInfinitesimalStep, model) =
    withhold_scalar_transport!(model, keys(timestepper.substepper.specific_scalars))

# The registers of the stages. With `TwoRegisterStorage` (diagonal `β`, `α` and `γ` on the first
# subdiagonal) stage `i` reads only its own slow tendency and the increment of stage `i - 1`, so one
# tendency register and two alternating increment registers suffice; the coefficients that this
# drops are those that `stage_storage` judges negligible.
tendency_registers(::TwoRegisterStorage, stages) = 1
tendency_registers(storage, stages) = stages
increment_registers(::TwoRegisterStorage, stages) = min(2, stages)
increment_registers(storage, stages) = stages

tendency_register(::TwoRegisterStorage, i) = 1
tendency_register(storage, i) = i
increment_register(::TwoRegisterStorage, i) = mod1(i, 2)
increment_register(storage, i) = i

# The earlier stages whose increments, and the stages whose slow tendencies, stage `i` reads
increment_stages(::TwoRegisterStorage, i) = max(1, i - 1):i-1
increment_stages(storage, i) = 1:i-1
tendency_stages(::TwoRegisterStorage, i) = i:i
tendency_stages(storage, i) = 1:i

stage_slow_tendencies(ts::GravityWaveMultirateInfinitesimalStep, j) = ts.slow_tendencies[tendency_register(ts.storage, j)]
stage_increment(ts::GravityWaveMultirateInfinitesimalStep, j) = ts.stage_increments[increment_register(ts.storage, j)]

# The interval over which a stage-aware sink budget (`AtmosphereModels.StageStart`) drains a
# species, as for the acoustic `MultirateInfinitesimalStep`: `Δt / C` with `C` the strong-stability
# coefficient of the coefficient set. For convex stages (see `convex_scalar_stages`), stage `i` is
# `(1 - wᵢ) Uⁿ + wᵢ [Yᵢ₋₁ + (βᵢᵢ / wᵢ) Δt S(Yᵢ₋₁)]` and `1 / C = maxᵢ βᵢᵢ / wᵢ` (see
# `largest_euler_step_fraction`), so a sink budgeted against the state the stage starts from,
# `Yᵢ₋₁` (the default `stage_starts_from_initial_state = false`), over this interval leaves every
# species of every stage non-negative. This is not the substep interval `dᵢ Δt` of the gravity
# waves. The time step is that of the last step, as the stepper keeps no current one; before the
# first step it is not finite and the budgets fall back to their fixed timescales.
# A scheme that runs through the generic interface on moist `HydrostaticDynamics` reads this, for
# its sink budgets and for the implicit fall of its condensate.
@inline function AtmosphereModels.stage_time_step(ts::GravityWaveMultirateInfinitesimalStep, clock)
    Δt = clock.last_Δt
    C = ts.tableau.strong_stability_coefficient
    return ifelse(C > 0, Δt / C, Δt)
end

# The weights of the stage tendencies in the state at the end of a time step. A Shu-Osher stage
# u⁽ᵐ⁾ = (1 - α) u⁰ + α [u⁽ᵐ⁻¹⁾ + Δt G] carries the tendencies of the stages before it with the weight α.
HydrostaticEquations.stage_tendency_weights(ts::SSPRungeKutta3) = (ts.α³ * ts.α² * ts.α¹, ts.α³ * ts.α², ts.α³)

# A variable without a fast tendency takes ΔYᵢ = Σₗ (αᵢₗ + γᵢₗ) ΔYₗ + Δt Σⱼ βᵢⱼ Sⱼ over the stages that the
# storage of the coefficient set reads (the convex scalar stages are the same recursion), so the
# weight of Sⱼ in ΔYᵢ is bᵢⱼ = βᵢⱼ + Σₗ (αᵢₗ + γᵢₗ) bₗⱼ, and the state at the end of the step is Uⁿ + ΔYₛ.
function HydrostaticEquations.stage_tendency_weights(ts::GravityWaveMultirateInfinitesimalStep)
    (; α, β, γ) = ts.tableau
    storage = ts.storage
    stages = number_of_stages(ts.tableau)
    b = zeros(eltype(β), stages, stages)

    for i in 1:stages
        for j in tendency_stages(storage, i)
            b[i, j] += β[i, j]
        end

        for l in increment_stages(storage, i), j in 1:l
            b[i, j] += (α[i, l] + γ[i, l]) * b[l, j]
        end
    end

    return b[stages, :]
end

# The substeps advance ρθ as an increment per unit volume (`set_substepped_scalar_sum!`)
HydrostaticEquations.substepped_potential_temperature(::GravityWaveMultirateInfinitesimalStep) = true

# What the substeps of a stage move through the seam of a Yin-Yang grid
HydrostaticEquations.stage_seam_budgets(timestepper::GravityWaveMultirateInfinitesimalStep, model) =
    gravity_wave_seam_budgets(timestepper.substepper, timestepper.U⁰.pˢ, model)

# The seam budgets of the substeps are those of the integrated velocity perturbation and of the
# specific values at the start of the step, which the convex scalar stages replace by the mass flux
# of the stage and the mixing ratio it starts from: the fixer has no budget for them.
# The convex scalar stages carry the mass and the scalars with the mass flux of each stage: their seam budgets
# are those of that flux, recorded before the stage changes the scalars (`record_stage_seam_budgets!`)
HydrostaticEquations.stage_seam_budgets(timestepper::ConvexGravityWaveMultirateInfinitesimalStep, model) =
    convex_stage_seam_budgets(timestepper, model)

const GravityWaveMultirateModel = AtmosphereModel{<:HydrostaticDynamics, <:Any, <:Any, <:GravityWaveMultirateInfinitesimalStep}

Base.summary(::GravityWaveMultirateInfinitesimalStep) = "GravityWaveMultirateInfinitesimalStep"

function Base.show(io::IO, timestepper::GravityWaveMultirateInfinitesimalStep)
    substepper = timestepper.substepper
    print(io, summary(timestepper), '\n')
    print(io, "├── stages: ", number_of_stages(timestepper.tableau), '\n')
    print(io, "├── inner_solver: ", summary(substepper.inner_solver), '\n')
    print(io, "├── substeps: ", substepper.substeps, '\n')
    print(io, "└── cfl: ", substepper.cfl)
    timestepper.scalar_stages isa ConvexScalarStages &&
        print(io, '\n', "    scalars: convex stages, forward Euler steps up to ",
              round(largest_euler_step_fraction(timestepper.tableau), digits=3), " Δt")
end

#####
##### Field arithmetic of the stages
#####

@kernel function _add_scaled!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] += c * b[i, j, k]
end

@kernel function _set_sum!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k] + c[i, j, k]
end

# a += c b, skipped for a vanishing coefficient
function add_scaled!(a, b, c)
    iszero(c) && return nothing
    grid = a.grid
    arch = grid.architecture
    @apply_regionally launch!(arch, grid, ssp_rk3_substep_parameters(grid, a), _add_scaled!,
                              a, b, convert(eltype(grid), c))
    return nothing
end

function set_sum!(a, b, c)
    grid = a.grid
    @apply_regionally launch!(grid.architecture, grid, ssp_rk3_substep_parameters(grid, a), _set_sum!, a, b, c)
    return nothing
end

#####
##### Stages
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Store the slow tendencies of stage `i`: for the variables the substeps advance, the full
tendencies of the present state ``Y_{i-1}`` less the gravity-wave operator applied to
``ΔY_{i-1}``, which the substepper still holds from the previous stage; for the other scalars,
the full tendencies, from which the transport by the velocity perturbation is taken when the
stage is completed.
"""
function store_slow_tendencies!(timestepper::GravityWaveMultirateInfinitesimalStep, model, i)
    S = stage_slow_tendencies(timestepper, i)

    for (s, G) in zip(S, timestepper.Gⁿ)
        copy_parent!(s, G)
    end

    # ΔY₀ = 0, on which the linear operator vanishes
    i > 1 && subtract_fast_tendencies!(S, timestepper.substepper, model)

    # On a pressure-level grid the scalars that the substeps do not advance take their tendencies
    # in the units of the stages, the mass per unit area (see `layer_mass_scalar_stages.jl`), at the
    # depth of the start of this stage: pˢ⁻ ← pˢ first (`initialize_stage!` repeats it)
    store_previous_surface_pressure!(model.grid)
    convert_tendencies_to_layer_mass!(model.grid, S, keys(timestepper.substepper.specific_scalars))
    i > 1 && subtract_fast_scalar_transport!(model.grid, timestepper.scalar_stages, timestepper, model, i)

    return nothing
end

# The slow tendencies of the surface pressure hold the fast operator applied to ΔYᵢ₋₁ on the
# metrics of stage `i`. On the label grid the fast operator is frozen over the step, and the
# scalars take the same term out when the stage completes, as transport by the integrated
# velocity less `Δt βᵢⱼ ΔYⱼ₋₁`. On a pressure-level grid the operator follows the surface pressure
# of the stage, and a stage that recombines the slow tendencies of earlier stages would take it
# out on the wrong metrics: the scalars subtract it here instead, with their slow tendencies, on
# the metrics of the stage that stores them.
subtract_fast_scalar_transport!(grid, scalar_stages, timestepper, model, i) = nothing

function subtract_fast_scalar_transport!(grid::PressureLevelGrid, ::RecombinedScalarStages, timestepper, model, i)
    substepper = timestepper.substepper
    ΔYᵢ₋₁ = stage_increment(timestepper, i - 1)
    ∫𝐮′ = substepper.integrated_velocities
    copy_parent!(∫𝐮′.u, ΔYᵢ₋₁.u)
    copy_parent!(∫𝐮′.v, ΔYᵢ₋₁.v)
    compute_integrated_vertical_velocity!(substepper, model)

    S = stage_slow_tendencies(timestepper, i)
    for name in keys(substepper.specific_scalars)
        add_perturbation_transport!(S[name], substepper, model, name, -1)
    end

    return nothing
end

# Whether the scalars take the fast part of the slow tendencies out through the integrated velocity
corrects_integrated_velocity(grid, scalar_stages) = true
corrects_integrated_velocity(::PressureLevelGrid, ::RecombinedScalarStages) = false

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set the perturbations from which the substeps of stage `i` start, ``Σ_j α_{ij} ΔY_j``, and the
forcing they hold constant, ``[Σ_j γ_{ij} ΔY_j / Δt + Σ_j β_{ij} S_j] / d_i``.
"""
function initialize_stage!(timestepper::GravityWaveMultirateInfinitesimalStep, i, Δt)
    (; α, β, γ, d) = timestepper.tableau
    substepper = timestepper.substepper
    storage = timestepper.storage

    # The stage starts from the present surface pressure (on a pressure-level grid; nothing otherwise)
    store_previous_surface_pressure!(timestepper.U⁰.pˢ.grid)

    for name in gravity_wave_prognostic_names
        U′ = substepper.perturbations[name]
        F = substepper.forcing[name]
        fill!(U′, 0)
        fill!(F, 0)

        for j in increment_stages(storage, i)
            ΔYⱼ = stage_increment(timestepper, j)
            add_scaled!(U′, ΔYⱼ[name], α[i, j])
            add_scaled!(F, ΔYⱼ[name], γ[i, j] / (d[i] * Δt))
        end

        for j in tendency_stages(storage, i)
            add_scaled!(F, stage_slow_tendencies(timestepper, j)[name], β[i, j] / d[i])
        end
    end

    fill_halo_regions!((substepper.perturbations.u, substepper.perturbations.v))

    for ∫u′ in substepper.integrated_velocities
        fill!(∫u′, 0)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Complete stage `i` after its substeps: advance the scalars that the substeps did not, store the
increment ``ΔY_i`` of the variables they advanced and recover their state ``Uⁿ + ΔY_i``.

The scalars are advanced first, while `model` still holds the state the stage started from, with
the velocity perturbation integrated over the substeps less the part of it that the full
tendencies ``G_j`` already carry, ``Δt Σ_j β_{ij} u′_{j-1}``. The mass fluxes of this net
perturbation are those that the surface pressure perturbation accounts for.
"""
function complete_stage!(timestepper::GravityWaveMultirateInfinitesimalStep, model, i, Δt, callbacks=())
    β = timestepper.tableau.β
    substepper = timestepper.substepper
    storage = timestepper.storage
    ΔYᵢ = stage_increment(timestepper, i)
    U⁰ = timestepper.U⁰
    prognostic = prognostic_fields(model)

    ∫𝐮′ = substepper.integrated_velocities
    for j in tendency_stages(storage, i)
        j > 1 && corrects_integrated_velocity(model.grid, timestepper.scalar_stages) || continue
        ΔYⱼ₋₁ = stage_increment(timestepper, j - 1)
        add_scaled!(∫𝐮′.u, ΔYⱼ₋₁.u, - Δt * β[i, j])
        add_scaled!(∫𝐮′.v, ΔYⱼ₋₁.v, - Δt * β[i, j])
    end

    complete_scalar_stage!(timestepper.scalar_stages, timestepper, model, i, Δt, callbacks)

    # The surface pressure first: on a pressure-level grid the densities of the scalars are those
    # of their masses at the surface pressure of the end of the stage
    for name in gravity_wave_prognostic_names
        copy_parent!(ΔYᵢ[name], substepper.perturbations[name])
        set_gravity_wave_prognostic_sum!(Val(name), prognostic[name], U⁰[name], ΔYᵢ[name])
    end

    set_completed_scalars!(timestepper.scalar_stages, timestepper, model, i)

    return nothing
end

set_gravity_wave_prognostic_sum!(name, u, U⁰, ΔY) = set_sum!(u, U⁰, ΔY)
set_gravity_wave_prognostic_sum!(::Val{:ρθ}, ρθ, U⁰, ΔY) = set_substepped_scalar_sum!(ρθ, U⁰, ΔY)

# The scalars that `complete_scalar_stage!` advanced, set once the surface pressure of the stage is
# complete: from the increment of the stage, or from the state the convex stage left in the
# storage of that increment
function set_completed_scalars!(::RecombinedScalarStages, timestepper, model, i)
    U⁰ = timestepper.U⁰
    prognostic = prognostic_fields(model)
    for name in keys(timestepper.substepper.specific_scalars)
        set_scalar_sum!(prognostic[name], U⁰[name], stage_increment(timestepper, i)[name])
    end
    return nothing
end

function set_completed_scalars!(::ConvexScalarStages, timestepper, model, i)
    prognostic = prognostic_fields(model)
    for name in keys(timestepper.substepper.specific_scalars)
        complete_scalar_stage_transport!(prognostic[name], stage_increment(timestepper, i)[name])
    end
    return nothing
end

# The increment ``Σ_j (α_{ij} + γ_{ij}) ΔY_j + Δt Σ_j β_{ij} G_j`` of the full tendencies, and the
# transport of the specific value at the start of the step by the net velocity perturbation, in the
# units of the stages; `set_completed_scalars!` sets the scalars from it.
function complete_scalar_stage!(::RecombinedScalarStages, timestepper, model, i, Δt, callbacks=())
    (; α, β, γ) = timestepper.tableau
    substepper = timestepper.substepper
    storage = timestepper.storage
    U⁰ = timestepper.U⁰
    prognostic = prognostic_fields(model)

    compute_integrated_vertical_velocity!(substepper, model)

    for name in keys(substepper.specific_scalars)
        δρc = stage_increment(timestepper, i)[name]
        fill!(δρc, 0)

        for j in increment_stages(storage, i)
            add_scaled!(δρc, stage_increment(timestepper, j)[name], α[i, j] + γ[i, j])
        end

        for j in tendency_stages(storage, i)
            add_scaled!(δρc, stage_slow_tendencies(timestepper, j)[name], Δt * β[i, j])
        end

        add_perturbation_transport!(δρc, substepper, model, name)
    end

    return nothing
end

# One transport by the mass flux of the stage, from the state the stage started from: see
# `moist_stage_scalar_transport.jl`. The slow tendencies of the scalars hold everything but
# their transport, and the storage of their increment holds the new state until
# `set_completed_scalars!` sets the scalars from it.
function complete_scalar_stage!(::ConvexScalarStages, timestepper, model, i, Δt, callbacks=())
    (; α, β, γ) = timestepper.tableau
    substepper = timestepper.substepper
    N = stage_slow_tendencies(timestepper, i)
    scratch = stage_increment(timestepper, i)
    U⁰ = timestepper.U⁰
    prognostic = prognostic_fields(model)

    weight = i == 1 ? one(eltype(β)) : α[i, i-1] + γ[i, i-1]
    Δtˢ = β[i, i] * Δt
    Δtᵉ = Δtˢ / weight

    compute_stage_transport_velocities!(substepper, model, Δtˢ)

    # The seam budgets of the stage, before it changes the scalars
    for callback in callbacks
        callback.callsite isa TendencyCallsite && record_stage_seam_budgets!(callback.func, model, Δtˢ)
    end

    for name in keys(substepper.specific_scalars)
        advance_scalar_with_stage_transport!(prognostic[name], U⁰[name], scratch[name], N[name],
                                             model, substepper, name, weight, Δtˢ, Δtᵉ)
    end

    return nothing
end

# The substeps of the stage moved mass and scalars with the velocity perturbation. On a Yin-Yang
# grid a conservation fixer among the tendency callbacks returns what they moved through the
# seam, to the increment of the stage and to the state built from it.
function return_stage_seam_budgets!(timestepper::GravityWaveMultirateInfinitesimalStep, model, i, callbacks)
    targets = (stage_increment(timestepper, i), prognostic_fields(model))

    for callback in callbacks
        callback.callsite isa TendencyCallsite && correct_completed_stage!(callback.func, model, targets)
    end

    return nothing
end

#####
##### Time stepping
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Step forward `model` one time step `Δt` with a multirate infinitesimal step method and
substepped gravity waves.

The gravity-wave system is linearized about the state at the start of the step, and the number
of substeps of each stage follows from the speed of the fastest gravity wave of that state.
"""
function OceananigansTimeSteppers.time_step!(model::GravityWaveMultirateModel, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    timestepper = model.timestepper
    substepper = timestepper.substepper
    tableau = timestepper.tableau
    stages = number_of_stages(tableau)

    tⁿ⁺¹ = model.clock.time + Δt

    store_initial_state!(model)
    freeze_gravity_wave_linearization!(substepper, model)
    wave_speed = maximum_gravity_wave_speed(model)

    for i in 1:stages
        # `update_state!` built the full tendencies of the state the stage starts from
        compute_flux_bc_tendencies!(model)
        store_slow_tendencies!(timestepper, model, i)
        initialize_stage!(timestepper, i, Δt)

        Δt_stage = tableau.d[i] * Δt
        Nτ, Δτ = stage_substeps(substepper, Δt_stage, tableau.d[i], wave_speed)
        gravity_wave_substep_loop!(substepper, model, Nτ, Δτ)

        complete_stage!(timestepper, model, i, Δt, callbacks)
        return_stage_seam_budgets!(timestepper, model, i, callbacks)

        if i < stages
            cⁱ⁻¹ = i == 1 ? zero(eltype(tableau.c)) : tableau.c[i-1]
            tick_stage!(model.clock, (tableau.c[i] - cⁱ⁻¹) * Δt)
            update_state!(model, callbacks; compute_tendencies = true)
        end
    end

    # Adjust the final stage to reduce the accumulation of floating point error in the time
    corrected_Δt = time_difference_in_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    implicit_dynamics_step!(model.dynamics, model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)

    # The operator-split microphysics update, once per step on the state just refreshed, as
    # `:SSPRungeKutta3` applies it. Nothing without microphysics.
    microphysics_model_update!(model.microphysics, model)
    end_of_microphysics_update!(model.dynamics, model)

    return nothing
end

Oceananigans.prognostic_state(::GravityWaveMultirateInfinitesimalStep) = nothing
Oceananigans.restore_prognostic_state!(timestepper::GravityWaveMultirateInfinitesimalStep, ::Nothing) = timestepper
