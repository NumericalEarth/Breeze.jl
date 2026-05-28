using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields
using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!,
    implicit_step!

using Breeze.AtmosphereModels: AtmosphereModel, compute_pressure_correction!, make_pressure_correction!,
                                microphysics_model_update!
using Oceananigans.Utils: launch!, time_difference_seconds
using Oceananigans.TurbulenceClosures: step_closure_prognostics!

"""
$(TYPEDEF)

A strong stability preserving (SSP) third-order Runge-Kutta time stepper.

This time stepper uses the classic SSP RK3 scheme ([Shu-Osher 2006](@cite Shu1988Efficient) form):

```math
\\begin{align*}
u^{(1)} &= u^{(0)} + Δt \\, G(u^{(0)}) \\\\
u^{(2)} &= \\frac{3}{4} u^{(0)} + \\frac{1}{4} u^{(1)} + \\frac{1}{4} Δt \\, G(u^{(1)}) \\\\
u^{(3)} &= \\frac{1}{3} u^{(0)} + \\frac{2}{3} u^{(2)} + \\frac{2}{3} Δt \\, G(u^{(2)})
\\end{align*}
```

where ``G`` above is the right-hand-side, e.g., ``\\partial_t u = G(u)``.

Each stage can be written in the form:
```math
u^{(m)} = (1 - α) u^{(0)} + α \\left[u^{(m-1)} + Δt \\, G(u^{(m-1)}) \\right]
```
with ``α = 1, 1/4, 2/3`` for stages 1, 2, 3 respectively.

This scheme has CFL coefficient equal to 1 and it is TVD (total variation diminishing).

Fields
======

- `α¹, α², α³`: Stage coefficients (1, 1/4, 2/3)
- `U⁰`: Storage for state at beginning of time step
- `Gⁿ`: Tendency fields at current stage
- `implicit_solver`: Optional implicit solver for diffusion
"""
struct SSPRungeKutta3{FT, U0, TG, TI} <: AbstractTimeStepper
    α¹ :: FT
    α² :: FT
    α³ :: FT
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
end

"""
$(TYPEDSIGNATURES)

Construct an `SSPRungeKutta3` on `grid` with `prognostic_fields` as described
by [Shu and Osher (1988)](@cite Shu1988Efficient).

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`

References
==========

Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially non-oscillatory
    shock-capturing schemes. Journal of Computational Physics, 77(2), 439-471.
"""
function SSPRungeKutta3(grid, prognostic_fields;
                        dynamics = nothing,
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

    return SSPRungeKutta3{FT, U0, TG, TI}(α¹, α², α³, U⁰, Gⁿ, implicit_solver)
end

#####
##### Stage update kernel
#####

"""
$(TYPEDSIGNATURES)

Apply an SSP RK3 substep with coefficient ``α``:
```math
u^{(m)} = (1 - α) u^{(0)} + α \\left[ u^{(m-1)} + Δt \\, G \\right]
```
where ``u^{(0)}`` is stored in the time stepper, ``u^{(m-1)}`` is the current field value,
and ``G`` is the current tendency.
"""
function ssp_rk3_substep!(model, Δt, α)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    Δt_FT = convert(eltype(grid), Δt)

    for (i, (u, u⁰, G)) in enumerate(zip(prognostic_fields(model), U⁰, Gⁿ))
        launch!(arch, grid, :xyz, _ssp_rk3_substep!, u, u⁰, G, Δt_FT, α)

        # Field index for implicit solver:
        # - indices 1, 2, 3 are momentum (ρu, ρv, ρw)
        # - indices 4+ are scalars (ρθ/ρe, ρqᵗ, microphysics, tracers)
        # For scalars, we use Val(i - 3) to get Val(1), Val(2), etc.
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

@kernel function _ssp_rk3_substep!(u, u⁰, G, Δt, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # u^(m) = (1 - α) * u^(0) + α * (u^(m-1) + Δt * G)
        u[i, j, k] = (1 - α) * u⁰[i, j, k] + α * (u[i, j, k] + Δt * G[i, j, k])
    end
end

"""
$(TYPEDSIGNATURES)

Copy prognostic fields to `U⁰` storage for use in later RK3 stages.
"""
function store_initial_state!(model)
    U⁰ = model.timestepper.U⁰
    for (u⁰, u) in zip(U⁰, prognostic_fields(model))
        parent(u⁰) .= parent(u)
    end
    return nothing
end

#####
##### Time stepping
#####

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step ``Δt`` with the SSP RK3 method.

The algorithm is:

```math
\\begin{align*}
u^{(1)} &= u^{(0)} + Δt \\, G(u^{(0)}) \\\\
u^{(2)} &= \\frac{3}{4} u^{(0)} + \\frac{1}{4} u^{(1)} + \\frac{1}{4} Δt \\, G(u^{(1)}) \\\\
u^{(3)} &= \\frac{1}{3} u^{(0)} + \\frac{2}{3} u^{(2)} + \\frac{2}{3} Δt \\, G(u^{(2)})
\\end{align*}
```

where ``G`` above is the right-hand-side, e.g., ``∂u/∂t = G(u)``.
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:SSPRungeKutta3}, Δt; callbacks=[])

    # Be paranoid and prepare at iteration 0, in case run! is not used:
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    ts = model.timestepper
    α¹ = ts.α¹
    α² = ts.α²
    α³ = ts.α³

    # Compute the next time step a priori to reduce floating point error accumulation
    tⁿ⁺¹ = model.clock.time + Δt

    # Store u^(0) for use in stages 2 and 3
    store_initial_state!(model)

    #
    # First stage: u^(1) = u^(0) + Δt * G(u^(0))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α¹)

    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)

    tick_stage!(model.clock, Δt)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, Δt)

    #
    # Second stage: u^(2) = 3/4 u^(0) + 1/4 (u^(1) + Δt * G(u^(1)))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α²)

    compute_pressure_correction!(model, α² * Δt)
    make_pressure_correction!(model, α² * Δt)

    # Don't tick - still at t + Δt for time-dependent forcing
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, α² * Δt)

    #
    # Third stage: u^(3) = 1/3 u^(0) + 2/3 (u^(2) + Δt * G(u^(2)))
    #

    compute_flux_bc_tendencies!(model)
    ssp_rk3_substep!(model, Δt, α³)

    compute_pressure_correction!(model, α³ * Δt)
    make_pressure_correction!(model, α³ * Δt)

    # Adjust final time-step to reduce floating point error accumulation
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    # Operator-split microphysics: applied once per time step on the post-RK
    # state (rather than once per stage from `update_state!`), so that the
    # full Δt of autoconversion / accretion / condensation / sedimentation
    # is applied exactly once. Required for `DCMIP2016KesslerMicrophysics`,
    # which bypasses the standard tendency interface and updates state via
    # this hook.
    microphysics_model_update!(model.microphysics, model)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, α³ * Δt)

    return nothing
end
