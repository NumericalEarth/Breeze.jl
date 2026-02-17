using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields, architecture
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
    CompressibleDynamics,
    AcousticSubstepper,
    acoustic_rk3_substep_loop!,
    prepare_acoustic_cache!

"""
$(TYPEDEF)

Wicker-Skamarock third-order Runge-Kutta time stepper with acoustic substepping
for fully compressible dynamics.

Unlike [`AcousticSSPRungeKutta3`](@ref) which uses convex combinations,
this scheme uses stage fractions `Δt/3, Δt/2, Δt`:

- Stage 1: ``U^* = U^n + (Δt/3) \\, R(U^n)``
- Stage 2: ``U^{**} = U^n + (Δt/2) \\, R(U^*)``
- Stage 3: ``U^{n+1} = U^n + Δt \\, R(U^{**})``

Each stage evaluates the RHS at the current stage state, then resets to ``U^n``
and advances by ``β Δt``. The absence of convex combinations makes this scheme
compatible with split-explicit acoustic substepping, allowing the full pressure
gradient and buoyancy to be included in the slow tendency.

This is the scheme used by WRF and CM1 for compressible atmospheric dynamics.

Fields
======

- `β₁, β₂, β₃`: Stage fractions (1/3, 1/2, 1)
- `U⁰`: Storage for state at beginning of time step
- `Gⁿ`: Tendency fields at current stage
- `implicit_solver`: Optional implicit solver for diffusion
- `substepper`: AcousticSubstepper for acoustic substepping infrastructure

References
==========

Wicker, L.J. and Skamarock, W.C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130, 2088-2097.

Klemp, J.B., Skamarock, W.C. and Dudhia, J. (2007). Conservative Split-Explicit
    Time Integration Methods for the Compressible Nonhydrostatic Equations.
    Monthly Weather Review, 135, 2897-2913.
"""
struct AcousticRungeKutta3{FT, U0, TG, TI, AS} <: AbstractTimeStepper
    β₁ :: FT
    β₂ :: FT
    β₃ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    substepper :: AS
end

"""
    AcousticRungeKutta3(grid, prognostic_fields;
                         dynamics,
                         implicit_solver = nothing,
                         Gⁿ = map(similar, prognostic_fields))

Construct an `AcousticRungeKutta3` time stepper for fully compressible dynamics.

Keyword Arguments
=================

- `dynamics`: The [`CompressibleDynamics`](@ref) object containing the `time_discretization`.
- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
"""
function AcousticRungeKutta3(grid, prognostic_fields;
                              dynamics,
                              implicit_solver::TI = nothing,
                              Gⁿ::TG = map(similar, prognostic_fields)) where {TI, TG}

    FT = eltype(grid)

    # Wicker-Skamarock RK3 stage fractions
    β₁ = FT(1//3)
    β₂ = FT(1//2)
    β₃ = FT(1)

    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    substepper = AcousticSubstepper(grid, dynamics.time_discretization)
    AS = typeof(substepper)

    return AcousticRungeKutta3{FT, U0, TG, TI, AS}(β₁, β₂, β₃, U⁰, Gⁿ, implicit_solver, substepper)
end

#####
##### Slow tendency computation (excludes PGF and buoyancy for split-explicit)
#####

using Breeze.AtmosphereModels:
    AtmosphereModels,
    SlowTendencyMode,
    dynamics_density,
    compute_x_momentum_tendency!,
    compute_y_momentum_tendency!,
    compute_z_momentum_tendency!,
    compute_dynamics_tendency!

"""
$(TYPEDSIGNATURES)

Compute slow momentum tendencies (advection, Coriolis, turbulence, forcing).

The pressure gradient and buoyancy are excluded using [`SlowTendencyMode`](@ref).
The acoustic substep loop handles the full PGF (reference + perturbation) and buoyancy:

- Reference PGF: ``∂(p_{stage} - p_{ref})/∂x``
- Perturbation PGF: ``ψ_θ ∂(ρθ'')/∂x`` where ``ψ_θ = c²/θ``
- Perturbation buoyancy: ``-g ρ''``
- Reference buoyancy: ``-g(ρ_{stage} - ρ_{ref})``
- Thermal buoyancy: ``B_θ (ρθ)''``

The outer RK3 time step is limited only by the advective CFL, not the acoustic CFL.
The acoustic CFL is resolved by the substep loop with constant ``Δτ = Δt/N``.
"""
function compute_slow_momentum_tendencies!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:AcousticRungeKutta3})
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    # Wrap dynamics in SlowTendencyMode so pressure gradient and buoyancy return zero.
    # The acoustic substep loop handles full PGF through the Exner pressure π'.
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

    Gⁿ = model.timestepper.Gⁿ

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gⁿ.ρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gⁿ.ρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gⁿ.ρw, grid, w_args)

    return nothing
end

#####
##### WS-RK3 substep with acoustic substepping
#####

"""
$(TYPEDSIGNATURES)

Apply a Wicker-Skamarock RK3 substep with acoustic substepping.

The acoustic substep loop handles momentum, density, and the thermodynamic
variable (ρθ or ρe). The substep size is constant ``Δτ = Δt/N`` across all
stages, with the substep count varying as ``Nτ = \\mathrm{round}(β N)``.
Remaining scalars (tracers) are updated with standard RK3.
"""
function acoustic_rk3_substep!(model, Δt, β)
    substepper = model.timestepper.substepper
    U⁰ = model.timestepper.U⁰

    # Prepare stage-frozen reference state (needed by slow tendency computation)
    prepare_acoustic_cache!(substepper, model)

    # Compute slow momentum tendencies (advection, Coriolis, diffusion — PGF/buoyancy handled by acoustic loop)
    compute_slow_momentum_tendencies!(model)

    # Compute slow density and thermodynamic tendencies
    # (reuses function defined in acoustic_ssp_runge_kutta_3.jl)
    compute_slow_scalar_tendencies!(model)

    # Execute acoustic substep loop: constant Δτ = Δt/N, varying Nτ = round(β*N)
    acoustic_rk3_substep_loop!(model, substepper, Δt, β, U⁰)

    # Update remaining scalars (tracers) using WS-RK3
    scalar_rk3_substep!(model, β * Δt)

    return nothing
end

#####
##### Scalar update with time-averaged velocities
#####

"""
$(TYPEDSIGNATURES)

Update scalar fields using Wicker-Skamarock RK3 with time-averaged velocities.

For scalars beyond the acoustic fields (moisture, tracers), we apply the
simple RK update: ``u = u⁰ + Δt_{stage} G``.
"""
function scalar_rk3_substep!(model, Δt_stage)
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

        launch!(arch, grid, :xyz, _rk3_substep!, u, u⁰, G, Δt_stage)

        # Implicit diffusion step
        field_index = Val(i - n_acoustic)
        implicit_step!(u,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       field_index,
                       model.clock,
                       fields(model),
                       Δt_stage)
    end

    return nothing
end

@kernel function _rk3_substep!(u, u⁰, G, Δt_stage)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # Wicker-Skamarock RK3: u = u⁰ + β * Δt * G
        u[i, j, k] = u⁰[i, j, k] + Δt_stage * G[i, j, k]
    end
end

#####
##### Store initial state
#####

function store_initial_state!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:AcousticRungeKutta3})
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

Step forward `model` one time step `Δt` with Wicker-Skamarock RK3 and acoustic substepping.

The algorithm follows [Wicker and Skamarock (2002)](@cite WickerSkamarock2002):
- Outer loop: 3-stage RK3 with stage fractions `Δt/3, Δt/2, Δt`
- Inner loop: Acoustic substeps for fast (pressure) tendencies

Each RK stage:
1. Compute slow tendencies (advection, Coriolis, diffusion only — PGF/buoyancy in acoustic loop)
2. Execute acoustic substep loop for momentum and density (full PGF + buoyancy)
3. Update scalars using standard RK update with time-averaged velocities
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:AcousticRungeKutta3}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0
    maybe_initialize_state!(model, callbacks)

    ts = model.timestepper
    β₁ = ts.β₁
    β₂ = ts.β₂
    β₃ = ts.β₃

    # Compute the next time step a priori
    tⁿ⁺¹ = model.clock.time + Δt

    # Store u⁰ for use in all stages
    store_initial_state!(model)

    #
    # Stage 1: U* = Uⁿ + (Δt/3) R(Uⁿ)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₁)

    tick!(model.clock, β₁ * Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₁ * Δt)

    #
    # Stage 2: U** = Uⁿ + (Δt/2) R(U*)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₂)

    tick!(model.clock, (β₂ - β₁) * Δt; stage=true)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₂ * Δt)

    #
    # Stage 3: Uⁿ⁺¹ = Uⁿ + Δt R(U**)
    #

    compute_flux_bc_tendencies!(model)
    acoustic_rk3_substep!(model, Δt, β₃)

    # Adjust final time-step
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick!(model.clock, corrected_Δt)
    model.clock.last_stage_Δt = corrected_Δt
    model.clock.last_Δt = Δt

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, β₃ * Δt)

    return nothing
end
