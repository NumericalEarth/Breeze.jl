using KernelAbstractions: @kernel, @index

using Oceananigans: AbstractModel, prognostic_fields, fields, architecture
using Oceananigans.Utils: launch!, time_difference_seconds

using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!,
    implicit_step!

using Breeze.AtmosphereModels: AtmosphereModel

using Breeze.CompressibleEquations:
    AcousticSubstepper,
    acoustic_substep_loop!,
    prepare_acoustic_cache!

"""
$(TYPEDEF)

A strong stability preserving (SSP) third-order Runge-Kutta time stepper with
acoustic substepping for fully compressible dynamics.

This time stepper implements the Wicker-Skamarock scheme used in CM1:
- Outer RK3 loop for slow tendencies (advection, buoyancy, turbulence)
- Inner acoustic substep loop for fast tendencies (pressure gradient, compression)

The acoustic substepping separates time scales:
- Slow modes (advection, buoyancy): CFL ≈ 10-20 m/s → Δtˢˡᵒʷ ~ 1-10 s
- Fast modes (acoustic): CFL ≈ 340 m/s → Δtˢ ~ 0.1-0.3 s

By substepping the fast modes, we can use ~6 acoustic substeps per slow step
instead of reducing the overall time step by a factor of ~30.

Fields
======

- `α¹, α², α³`: SSP RK3 stage coefficients (1, 1/4, 2/3)
- `U⁰`: Storage for state at beginning of time step
- `Gⁿ`: Tendency fields at current stage
- `implicit_solver`: Optional implicit solver for diffusion
- `substepper`: AcousticSubstepper for acoustic substepping infrastructure
"""
struct AcousticSSPRungeKutta3{FT, U0, TG, TI, AS} <: AbstractTimeStepper
    α¹ :: FT
    α² :: FT
    α³ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
end

"""
    AcousticSSPRungeKutta3(grid, prognostic_fields;
                           dynamics,
                           implicit_solver = nothing,
                           Gⁿ = map(similar, prognostic_fields))

Construct an `AcousticSSPRungeKutta3` time stepper for fully compressible dynamics.

This combines the SSP RK3 scheme from [Shu and Osher (1988)](@cite Shu1988Efficient)
with acoustic substepping from [Wicker and Skamarock (2002)](@cite WickerSkamarock2002).

The acoustic substepping parameters are configured via the `time_discretization` field
of the [`CompressibleDynamics`](@ref) object passed as `dynamics`.

Keyword Arguments
=================

- `dynamics`: The [`CompressibleDynamics`](@ref) object containing the `time_discretization`.
- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`

References
==========

Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially non-oscillatory
    shock-capturing schemes. Journal of Computational Physics, 77(2), 439-471.

Wicker, L.J. and Skamarock, W.C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130, 2088-2097.
"""
function AcousticSSPRungeKutta3(grid, prognostic_fields;
                                dynamics,
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = map(similar, prognostic_fields)) where {TI, TG}

    FT = eltype(grid)

    # SSP RK3 stage coefficients
    α¹ = FT(1)
    α² = FT(1//4)
    α³ = FT(2//3)

    # Create storage for initial state (used in stages 2 and 3)
    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    # Create acoustic substepping infrastructure
    substepper = AcousticSubstepper(grid, dynamics.time_discretization)
    AS = typeof(substepper)

    return AcousticSSPRungeKutta3{FT, U0, TG, TI, AS}(α¹, α², α³, U⁰, Gⁿ, implicit_solver, substepper)
end

#####
##### Stage-frozen tendency computation
#####

using Oceananigans.Operators: divᶜᶜᶜ

using Breeze.AtmosphereModels:
    AtmosphereModels,
    SlowTendencyMode,
    dynamics_density,
    thermodynamic_density,
    compute_x_momentum_tendency!,
    compute_y_momentum_tendency!,
    compute_z_momentum_tendency!,
    compute_dynamics_tendency!

"""
$(TYPEDSIGNATURES)

Compute slow momentum tendencies (advection, Coriolis, turbulence, forcing).

The pressure gradient and buoyancy are excluded using [`SlowTendencyMode`](@ref).
These "fast" terms are handled by the acoustic substep loop through perturbation
variables: ``-ψ ∂ρ''/∂x`` (horizontal pressure) and ``-ψ ∂ρ''/∂z - g ρ''``
(vertical pressure + buoyancy).

!!! note "Limitation with SSP RK3"
    The perturbation pressure ``ψ ∂ρ''/∂x`` only captures density-driven pressure
    changes. Pressure perturbations from temperature/θ variations (the ``ρ ∂ψ/∂x``
    term) are not resolved. Including the full pressure gradient in the slow tendency
    is unstable with SSP RK3 due to acoustic mode interference from the convex
    combination. The Wicker-Skamarock RK3 (stage fractions Δt/3, Δt/2, Δt) is
    needed to properly handle the full pressure gradient.
"""
function compute_slow_momentum_tendencies!(model)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    # Wrap dynamics in SlowTendencyMode so pressure gradient and buoyancy return zero.
    # The acoustic substep provides perturbation pressure/buoyancy instead.
    slow_dynamics = SlowTendencyMode(model.dynamics)

    model_fields = fields(model)

    momentum_args = (
        dynamics_density(model.dynamics),
        model.advection.momentum,
        model.velocities,
        model.closure,
        model.closure_fields,
        model.momentum,
        model.coriolis,
        model.clock,
        model_fields)

    u_args = tuple(momentum_args..., model.forcing.ρu, slow_dynamics)
    v_args = tuple(momentum_args..., model.forcing.ρv, slow_dynamics)

    w_args = tuple(momentum_args..., model.forcing.ρw,
                   slow_dynamics,
                   model.formulation,
                   model.temperature,
                   model.specific_moisture,
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamic_constants)

    Gˢm = substepper.slow_momentum_tendencies

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gˢm.ρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gˢm.ρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gˢm.ρw, grid, w_args)

    return nothing
end

#####
##### Slow density and thermodynamic tendencies
#####

using Breeze.AtmosphereModels: compute_dynamics_tendency!, thermodynamic_density

"""
$(TYPEDSIGNATURES)

Compute slow tendencies for density and thermodynamic variable.

In the perturbation-variable approach, the slow tendencies are simply the full
RHS ``R^t`` evaluated at the stage-level state. No correction is needed because
the acoustic loop advances perturbation variables, not full fields.

- ``G^s_ρ = -\\boldsymbol{∇·m}^t``: full density tendency (continuity equation)
- ``G^s_χ``: full thermodynamic tendency (advection + physics)
"""
function compute_slow_scalar_tendencies!(model)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    # Compute Gˢρ = -∇·m^t (full density tendency at stage start)
    compute_dynamics_tendency!(model)
    χ_ρ_name = :ρ
    Gρ_full = getproperty(model.timestepper.Gⁿ, χ_ρ_name)
    parent(substepper.Gˢρ) .= parent(Gρ_full)

    # Compute Gˢχ = full thermodynamic tendency (no correction needed)
    Gⁿ = model.timestepper.Gⁿ
    common_args = (
        model.dynamics,
        model.formulation,
        model.thermodynamic_constants,
        model.specific_moisture,
        model.velocities,
        model.microphysics,
        model.microphysical_fields,
        model.closure,
        model.closure_fields,
        model.clock,
        fields(model))

    AtmosphereModels.compute_thermodynamic_tendency!(model, common_args)

    χ_name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    Gχ_full = getproperty(Gⁿ, χ_name)
    parent(substepper.Gˢρχ) .= parent(Gχ_full)

    return nothing
end

#####
##### SSP RK3 substep with acoustic substepping
#####

"""
$(TYPEDSIGNATURES)

Apply an SSP RK3 substep with acoustic substepping.

The acoustic substep loop handles momentum, density, and the thermodynamic
variable (ρθ or ρe). Remaining scalars (tracers) are updated using standard
SSP RK3 with time-averaged velocities from the acoustic loop.
"""
function acoustic_ssp_rk3_substep!(model, Δt, α, stage)
    grid = model.grid
    arch = grid.architecture
    substepper = model.timestepper.substepper
    U⁰ = model.timestepper.U⁰

    # Prepare stage-frozen reference state FIRST (needed by slow tendency correction)
    prepare_acoustic_cache!(substepper, model)

    # Compute slow momentum tendencies (everything except fast pressure gradient)
    compute_slow_momentum_tendencies!(model)

    # Compute slow density and thermodynamic tendencies
    # (requires χᵣ and ρᵣ from prepare_acoustic_cache!)
    compute_slow_scalar_tendencies!(model)

    # Execute acoustic substep loop for momentum, density, and thermodynamic variable
    # Pass full Δt, α, and U⁰ for SSP RK3 convex combination
    acoustic_substep_loop!(model, substepper, Δt, α, U⁰)

    # Update remaining scalars (tracers) using standard SSP RK3
    scalar_ssp_rk3_substep!(model, Δt, α)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Update scalar fields using standard SSP RK3 with time-averaged velocities.

For scalars (θ, moisture, tracers), we use the time-averaged velocities
from the acoustic loop for advection, ensuring stability.
"""
function scalar_ssp_rk3_substep!(model, Δt, α)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ

    prognostic = prognostic_fields(model)
    n_acoustic = 5  # ρ, ρu, ρv, ρw, ρθ (handled by acoustic loop)

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic, U⁰, Gⁿ))
        if i <= n_acoustic  # Skip fields handled by acoustic loop
            continue
        end

        launch!(arch, grid, :xyz, _ssp_rk3_substep!, u, u⁰, G, Δt, α)

        # Implicit diffusion step
        field_index = Val(i - n_acoustic)
        implicit_step!(u,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       field_index,
                       model.clock,
                       fields(model),
                       α * Δt)
    end

    return nothing
end

#####
##### Import SSP RK3 kernel from ssp_runge_kutta_3.jl (avoid duplicate definition)
#####

# The _ssp_rk3_substep! kernel is already defined in ssp_runge_kutta_3.jl
# We reuse it here instead of defining it again

#####
##### Store initial state
#####

function store_initial_state!(model::AbstractModel{<:AcousticSSPRungeKutta3})
    U⁰ = model.timestepper.U⁰
    for (u⁰, u) in zip(U⁰, prognostic_fields(model))
        parent(u⁰) .= parent(u)
    end
    return nothing
end

#####
##### Time stepping (main entry point)
#####

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with SSP RK3 and acoustic substepping.

The algorithm is the Wicker-Skamarock scheme:
- Outer loop: 3-stage SSP RK3 for slow tendencies
- Inner loop: Acoustic substeps for fast (pressure) tendencies

Each RK stage:
1. Compute slow tendencies (advection, Coriolis, diffusion)
2. Execute acoustic substep loop for momentum and density
3. Update scalars using standard RK update with time-averaged velocities
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:AcousticSSPRungeKutta3}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    ts = model.timestepper
    α¹ = ts.α¹
    α² = ts.α²
    α³ = ts.α³

    # Compute the next time step a priori
    tⁿ⁺¹ = model.clock.time + Δt

    # Store u⁰ for use in stages 2 and 3
    store_initial_state!(model)

    #
    # Stage 1: u¹ = u⁰ + Δt L(u⁰)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_ssp_rk3_substep!(model, Δt, α¹, 1)

    tick!(model.clock, Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, Δt)

    #
    # Stage 2: u² = ¾ u⁰ + ¼ (u¹ + Δt L(u¹))
    #

    compute_flux_bc_tendencies!(model)
    acoustic_ssp_rk3_substep!(model, Δt, α², 2)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, α² * Δt)

    #
    # Stage 3: u³ = ⅓ u⁰ + ⅔ (u² + Δt L(u²))
    #

    compute_flux_bc_tendencies!(model)
    acoustic_ssp_rk3_substep!(model, Δt, α³, 3)

    # Adjust final time-step
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick!(model.clock, corrected_Δt)
    model.clock.last_stage_Δt = corrected_Δt
    model.clock.last_Δt = Δt

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, α³ * Δt)

    return nothing
end
