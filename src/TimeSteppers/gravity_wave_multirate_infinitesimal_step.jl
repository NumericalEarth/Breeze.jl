using Oceananigans: TendencyCallsite
using Breeze.HydrostaticEquations: HydrostaticEquations, HydrostaticDynamics, GravityWaveSubstepper,
                                   VerletSubsteps, gravity_wave_prognostic_names,
                                   freeze_gravity_wave_linearization!, subtract_fast_tendencies!,
                                   gravity_wave_substep_loop!, stage_substeps, maximum_gravity_wave_speed,
                                   compute_integrated_vertical_velocity!, add_perturbation_transport!,
                                   gravity_wave_seam_budgets, correct_completed_stage!

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
"""
struct GravityWaveMultirateInfinitesimalStep{TB, U0, TG, SS, DY, TI, GS} <: AbstractTimeStepper
    tableau :: TB
    U⁰ :: U0
    Gⁿ :: TG
    slow_tendencies :: SS
    stage_increments :: DY
    implicit_solver :: TI
    substepper :: GS
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
  ([Knoth and Wensch 2014](@cite KnothWensch2014)), or `:WickerSkamarock3`
  ([Wicker and Skamarock 2002](@cite WickerSkamarock2002)); or a
  [`MultirateInfinitesimalStepTableau`](@ref). Required.
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
    tableau = multirate_infinitesimal_step_tableau(tableau, FT)
    stages = number_of_stages(tableau)

    all(d -> d > 0, tableau.d) ||
        throw(ArgumentError("Every stage of the tableau must integrate forward in time."))

    substepper = GravityWaveSubstepper(grid, inner_solver, prognostic_fields; substeps, cfl, moist)

    slow_tendencies = ntuple(_ -> map(similar, prognostic_fields), stages)
    stage_increments = ntuple(_ -> map(similar, prognostic_fields), stages)

    return GravityWaveMultirateInfinitesimalStep(tableau, U⁰, Gⁿ, slow_tendencies, stage_increments,
                                                 implicit_solver, substepper)
end

# What the substeps of a stage move through the seam of a Yin-Yang grid
HydrostaticEquations.stage_seam_budgets(timestepper::GravityWaveMultirateInfinitesimalStep, model) =
    gravity_wave_seam_budgets(timestepper.substepper, timestepper.U⁰.pˢ, model)

const GravityWaveMultirateModel = AtmosphereModel{<:HydrostaticDynamics, <:Any, <:Any, <:GravityWaveMultirateInfinitesimalStep}

Base.summary(::GravityWaveMultirateInfinitesimalStep) = "GravityWaveMultirateInfinitesimalStep"

function Base.show(io::IO, timestepper::GravityWaveMultirateInfinitesimalStep)
    substepper = timestepper.substepper
    print(io, summary(timestepper), '\n')
    print(io, "├── stages: ", number_of_stages(timestepper.tableau), '\n')
    print(io, "├── inner_solver: ", summary(substepper.inner_solver), '\n')
    print(io, "├── substeps: ", substepper.substeps, '\n')
    print(io, "└── cfl: ", substepper.cfl)
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
    S = timestepper.slow_tendencies[i]

    for (s, G) in zip(S, timestepper.Gⁿ)
        copy_parent!(s, G)
    end

    # ΔY₀ = 0, on which the linear operator vanishes
    i > 1 && subtract_fast_tendencies!(S, timestepper.substepper, model)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set the perturbations from which the substeps of stage `i` start, ``Σ_j α_{ij} ΔY_j``, and the
forcing they hold constant, ``[Σ_j γ_{ij} ΔY_j / Δt + Σ_j β_{ij} S_j] / d_i``.
"""
function initialize_stage!(timestepper::GravityWaveMultirateInfinitesimalStep, i, Δt)
    (; α, β, γ, d) = timestepper.tableau
    substepper = timestepper.substepper
    S = timestepper.slow_tendencies
    ΔY = timestepper.stage_increments

    for name in gravity_wave_prognostic_names
        U′ = substepper.perturbations[name]
        F = substepper.forcing[name]
        fill!(U′, 0)
        fill!(F, 0)

        for j in 1:i-1
            add_scaled!(U′, ΔY[j][name], α[i, j])
            add_scaled!(F, ΔY[j][name], γ[i, j] / (d[i] * Δt))
        end

        for j in 1:i
            add_scaled!(F, S[j][name], β[i, j] / d[i])
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

Complete stage `i` after its substeps: store its increment ``ΔY_i``, recover the state
``Uⁿ + ΔY_i`` of the variables the substeps advanced, and advance the other scalars.

The scalars take the increment ``Σ_j (α_{ij} + γ_{ij}) ΔY_j + Δt Σ_j β_{ij} G_j`` of their full
tendencies, and the transport of their specific value at the start of the step by the velocity
perturbation integrated over the substeps, less the part of it that the full tendencies
``G_j`` already carry, ``Δt Σ_j β_{ij} u′_{j-1}``. The mass fluxes of this net perturbation are
those that the surface pressure perturbation accounts for.
"""
function complete_stage!(timestepper::GravityWaveMultirateInfinitesimalStep, model, i, Δt)
    (; α, β, γ) = timestepper.tableau
    substepper = timestepper.substepper
    S = timestepper.slow_tendencies
    ΔY = timestepper.stage_increments
    U⁰ = timestepper.U⁰
    prognostic = prognostic_fields(model)

    for name in gravity_wave_prognostic_names
        copy_parent!(ΔY[i][name], substepper.perturbations[name])
        set_sum!(prognostic[name], U⁰[name], ΔY[i][name])
    end

    ∫𝐮′ = substepper.integrated_velocities
    for j in 2:i
        add_scaled!(∫𝐮′.u, ΔY[j-1].u, - Δt * β[i, j])
        add_scaled!(∫𝐮′.v, ΔY[j-1].v, - Δt * β[i, j])
    end

    compute_integrated_vertical_velocity!(substepper, model)

    for name in keys(substepper.specific_scalars)
        δρc = ΔY[i][name]
        fill!(δρc, 0)

        for j in 1:i-1
            add_scaled!(δρc, ΔY[j][name], α[i, j] + γ[i, j])
        end

        for j in 1:i
            add_scaled!(δρc, S[j][name], Δt * β[i, j])
        end

        add_perturbation_transport!(δρc, substepper, model, name)
        set_sum!(prognostic[name], U⁰[name], δρc)
    end

    return nothing
end

# The substeps of the stage moved mass and scalars with the velocity perturbation. On a Yin-Yang
# grid a conservation fixer among the tendency callbacks returns what they moved through the
# seam, to the increment of the stage and to the state built from it.
function return_stage_seam_budgets!(timestepper::GravityWaveMultirateInfinitesimalStep, model, i, callbacks)
    targets = (timestepper.stage_increments[i], prognostic_fields(model))

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

        complete_stage!(timestepper, model, i, Δt)
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

    return nothing
end

Oceananigans.prognostic_state(::GravityWaveMultirateInfinitesimalStep) = nothing
Oceananigans.restore_prognostic_state!(timestepper::GravityWaveMultirateInfinitesimalStep, ::Nothing) = timestepper
