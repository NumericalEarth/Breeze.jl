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
                          implicit_solver = nothing,
                          Gⁿ = map(similar, prognostic_fields),
                          split_explicit = SplitExplicitTimeDiscretization())

Construct an `AcousticSSPRungeKutta3` time stepper for fully compressible dynamics.

This combines the SSP RK3 scheme from [Shu and Osher (1988)](@cite Shu1988Efficient)
with acoustic substepping from [Wicker and Skamarock (2002)](@cite WickerSkamarock2002).

The acoustic substepping parameters (`Ns`, `vertical_time_discretization`, `κᵈ`) are
configured via the [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization) object,
which is typically set on [`CompressibleDynamics`](@ref) and passed through automatically.

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `split_explicit`: [`SplitExplicitTimeDiscretization`](@ref) configuration with `substeps`, `vertical_time_discretization`, and `κᵈ`.

References
==========

Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially non-oscillatory
    shock-capturing schemes. Journal of Computational Physics, 77(2), 439-471.

Wicker, L.J. and Skamarock, W.C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130, 2088-2097.
"""
function AcousticSSPRungeKutta3(grid, prognostic_fields;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = map(similar, prognostic_fields),
                                split_explicit = SplitExplicitTimeDiscretization()) where {TI, TG}

    FT = eltype(grid)

    # SSP RK3 stage coefficients
    α¹ = FT(1)
    α² = FT(1//4)
    α³ = FT(2//3)

    # Create storage for initial state (used in stages 2 and 3)
    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    # Create acoustic substepping infrastructure from SplitExplicitTimeDiscretization configuration
    substepper = AcousticSubstepper(grid, split_explicit)
    AS = typeof(substepper)

    return AcousticSSPRungeKutta3{FT, U0, TG, TI, AS}(α¹, α², α³, U⁰, Gⁿ, implicit_solver, substepper)
end

#####
##### Slow tendency computation (excludes pressure gradient and buoyancy)
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

Compute slow tendencies for momentum (advection, Coriolis, turbulence, forcing).

The pressure gradient and buoyancy are NOT included here - they are "fast" terms
that are computed during the acoustic substep loop. In hydrostatic equilibrium,
pressure gradient and buoyancy nearly cancel, so treating them together in the
fast loop maintains stability.

This function uses [`SlowTendencyMode`](@ref Breeze.AtmosphereModels.SlowTendencyMode)
to wrap the dynamics, causing pressure gradient and buoyancy functions to return zero.
"""
function compute_slow_momentum_tendencies!(model)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    # Wrap dynamics in SlowTendencyMode so that pressure gradient and buoyancy return zero
    slow_dynamics = SlowTendencyMode(model.dynamics)

    model_fields = fields(model)

    # Build momentum tendency arguments with slow dynamics
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

    # Extra arguments for vertical velocity are required to compute buoyancy
    # (which will return zero due to SlowTendencyMode)
    w_args = tuple(momentum_args..., model.forcing.ρw,
                   slow_dynamics,
                   model.formulation,
                   model.temperature,
                   model.specific_moisture,
                   model.microphysics,
                   model.microphysical_fields,
                   model.thermodynamic_constants)

    # Compute slow tendencies directly into substepper storage
    Gˢρu = substepper.Gˢρu
    Gˢρv = substepper.Gˢρv
    Gˢρw = substepper.Gˢρw

    launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gˢρu, grid, u_args)
    launch!(arch, grid, :xyz, compute_y_momentum_tendency!, Gˢρv, grid, v_args)
    launch!(arch, grid, :xyz, compute_z_momentum_tendency!, Gˢρw, grid, w_args)

    return nothing
end

#####
##### Slow density and thermodynamic tendencies
#####

using Breeze.AtmosphereModels: compute_dynamics_tendency!, thermodynamic_density

"""
$(TYPEDSIGNATURES)

Compute slow tendencies for density and thermodynamic variable.

For split-explicit time-stepping, the density equation is entirely handled
by the acoustic substep loop (backward step using mass flux divergence),
so the slow density tendency is zero (or contains only mass source terms).

The slow thermodynamic tendency is the full tendency corrected to remove
the fast acoustic flux divergence term that will be computed in the
acoustic substep loop. Following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007),
the slow tendency is effectively the advective-form transport plus physics:

``G^s_χ = G^{\\mathrm{full}}_χ + \\bar{s} \\, \\boldsymbol{∇·m}``

where ``\\bar{s} = \\bar{χ}/\\bar{ρ}`` is the stage-frozen specific thermodynamic
variable and ``\\boldsymbol{∇·m}`` is the mass flux divergence at the stage start.
"""
function compute_slow_scalar_tendencies!(model)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    # Slow density tendency is zero for dry dynamics (the full continuity
    # equation is handled by the acoustic backward step)
    fill!(substepper.Gˢρ, 0)

    # Compute full thermodynamic tendency into Gˢχ using existing tendency machinery.
    # The full tendency includes advection (flux form), diffusion, microphysics, forcing.
    compute_full_thermodynamic_tendency_into!(substepper.Gˢχ, model)

    # Correct the slow thermodynamic tendency by adding back the fast flux divergence
    # term that the acoustic loop will handle: Gˢχ += s̄ ∇·m
    # This converts from flux-form to advective-form transport.
    launch!(arch, grid, :xyz, _correct_slow_thermodynamic_tendency!,
            substepper.Gˢχ, grid,
            substepper.χᵣ, substepper.ρᵣ,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw)

    return nothing
end

@kernel function _correct_slow_thermodynamic_tendency!(Gˢχ, grid, χᵣ, ρᵣ, ρu, ρv, ρw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Stage-frozen specific thermodynamic variable: s̄ = χ̄ / ρ̄
        s̄ = χᵣ[i, j, k] / ρᵣ[i, j, k]

        # Mass flux divergence at stage start
        div_m = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)

        # Correct: Gˢχ += s̄ ∇·m
        # This removes the acoustic flux divergence from the full flux-form tendency,
        # leaving only the advective-form transport plus physics terms.
        Gˢχ[i, j, k] += s̄ * div_m
    end
end

"""
$(TYPEDSIGNATURES)

Compute the full thermodynamic tendency into a target field.

This calls the formulation-specific tendency computation and stores the
result in `target`, which is typically the substepper's `Gˢχ` field.
"""
function compute_full_thermodynamic_tendency_into!(target, model)
    grid = model.grid
    arch = grid.architecture
    Gⁿ = model.timestepper.Gⁿ

    # Use the existing tendency computation infrastructure
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

    # Compute the thermodynamic tendency using the formulation's method
    # This writes into the standard Gⁿ storage
    AtmosphereModels.compute_thermodynamic_tendency!(model, common_args)

    # Copy the result from the standard storage to our target field.
    # The thermodynamic tendency field name depends on the formulation.
    χ_name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    Gχ_full = getproperty(Gⁿ, χ_name)
    parent(target) .= parent(Gχ_full)

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

    # Prepare stage-frozen reference state FIRST (needed by slow tendency correction)
    prepare_acoustic_cache!(substepper, model)

    # Compute slow momentum tendencies (everything except fast pressure gradient)
    compute_slow_momentum_tendencies!(model)

    # Compute slow density and thermodynamic tendencies
    # (requires χᵣ and ρᵣ from prepare_acoustic_cache!)
    compute_slow_scalar_tendencies!(model)

    # Effective time step for this RK stage
    Δtˢᵗᵃᵍᵉ = α * Δt

    # Execute acoustic substep loop for momentum, density, and thermodynamic variable
    acoustic_substep_loop!(model, substepper, stage, Δtˢᵗᵃᵍᵉ)

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
    n_momentum = 3  # ρu, ρv, ρw

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic, U⁰, Gⁿ))
        if i <= n_momentum  # Skip momentum (handled by acoustic loop)
            continue
        end

        launch!(arch, grid, :xyz, _ssp_rk3_substep!, u, u⁰, G, Δt, α)

        # Implicit diffusion step
        field_index = Val(i - n_momentum)
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
