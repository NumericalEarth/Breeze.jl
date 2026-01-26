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

using Breeze.CompressibleEquations:
    AcousticSubstepper,
    acoustic_substep_loop!

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
                          Ns = 6,
                          α = 0.5,
                          κᵈ = 0.05)

Construct an `AcousticSSPRungeKutta3` time stepper for fully compressible dynamics.

This combines the SSP RK3 scheme from [Shu and Osher (1988)](@cite Shu1988Efficient)
with acoustic substepping from [Wicker and Skamarock (2002)](@cite WickerSkamarock2002).

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `Ns`: Number of acoustic substeps per full time step. Default: 6
- `α`: Implicitness parameter for vertical acoustic solve. Default: 0.5
- `κᵈ`: Divergence damping coefficient. Default: 0.05

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
                                Ns = 6,
                                α = 0.5,
                                κᵈ = 0.05) where {TI, TG}

    FT = eltype(grid)

    # SSP RK3 stage coefficients
    α¹ = FT(1)
    α² = FT(1//4)
    α³ = FT(2//3)

    # Create storage for initial state (used in stages 2 and 3)
    U⁰ = map(similar, prognostic_fields)
    U0 = typeof(U⁰)

    # Create acoustic substepping infrastructure
    substepper = AcousticSubstepper(grid; Ns, α, κᵈ)
    AS = typeof(substepper)

    return AcousticSSPRungeKutta3{FT, U0, TG, TI, AS}(α¹, α², α³, U⁰, Gⁿ, implicit_solver, substepper)
end

#####
##### Slow tendency computation (excludes pressure gradient, computed during acoustic loop)
#####

"""
$(TYPEDSIGNATURES)

Compute slow tendencies for momentum (advection, Coriolis, turbulence, forcing).

The pressure gradient and buoyancy are NOT included here - they are "fast" terms
that are computed during the acoustic substep loop. In hydrostatic equilibrium,
pressure gradient and buoyancy nearly cancel, so treating them together in the
fast loop maintains stability.
"""
function compute_slow_momentum_tendencies!(model)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)

    Gⁿ = model.timestepper.Gⁿ
    dynamics = model.dynamics

    # The full tendencies include pressure gradient and buoyancy.
    # For acoustic substepping, we subtract both to get slow tendencies.
    launch!(arch, grid, :xyz, _compute_slow_momentum_tendencies!,
            substepper.Gˢρu, substepper.Gˢρv, substepper.Gˢρw,
            Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw,
            dynamics, grid, model.thermodynamic_constants)

    return nothing
end

using Oceananigans.Operators: ℑzᵃᵃᶠ
using Breeze.AtmosphereModels: x_pressure_gradient, y_pressure_gradient, z_pressure_gradient, dynamics_density

@kernel function _compute_slow_momentum_tendencies!(Gˢρu, Gˢρv, Gˢρw,
                                                     Gρu, Gρv, Gρw,
                                                     dynamics, grid, constants)
    i, j, k = @index(Global, NTuple)

    # Full tendencies minus (pressure gradient + buoyancy) = slow tendencies
    ∂ₓp = x_pressure_gradient(i, j, k, grid, dynamics)
    ∂ᵧp = y_pressure_gradient(i, j, k, grid, dynamics)
    ∂zp = z_pressure_gradient(i, j, k, grid, dynamics)

    # Buoyancy term: ρb = -gρ at cell faces
    ρ = dynamics_density(dynamics)
    g = constants.gravitational_acceleration
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρb = -g * ρᶜᶜᶠ

    @inbounds begin
        Gˢρu[i, j, k] = Gρu[i, j, k] + ∂ₓp
        Gˢρv[i, j, k] = Gρv[i, j, k] + ∂ᵧp
        # Remove both pressure gradient AND buoyancy from vertical momentum
        Gˢρw[i, j, k] = Gρw[i, j, k] + ∂zp - ρb
    end
end

#####
##### SSP RK3 substep with acoustic substepping
#####

"""
Apply an SSP RK3 substep with acoustic substepping.

For momentum: acoustic substep loop handles the update
For scalars: standard SSP RK3 update using time-averaged velocities
"""
function acoustic_ssp_rk3_substep!(model, Δt, α, stage)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    substepper = model.timestepper.substepper

    # Compute slow momentum tendencies (everything except fast pressure gradient)
    compute_slow_momentum_tendencies!(model)

    # Effective time step for this RK stage
    Δtˢᵗᵃᵍᵉ = α * Δt

    # Execute acoustic substep loop for momentum and density
    acoustic_substep_loop!(model, substepper, stage, Δtˢᵗᵃᵍᵉ)

    # For non-momentum fields (scalars), use standard SSP RK3 update
    for (i, (u, u⁰, G)) in enumerate(zip(prognostic_fields(model), U⁰, Gⁿ))
        if i <= 3  # Skip momentum (handled by acoustic loop)
            continue
        end

        # Handle density specially - it's updated in acoustic loop
        if i == 4 + length(model.tracers) + 1  # Rough heuristic; TODO: improve
            continue
        end

        launch!(arch, grid, :xyz, _ssp_rk3_substep!, u, u⁰, G, Δt, α)

        # Field index for implicit solver
        field_index = Val(i - 3)

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
