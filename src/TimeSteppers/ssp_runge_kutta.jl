using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields
using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!,
    implicit_step!

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel,
                                compute_pressure_correction!, make_pressure_correction!,
                                microphysics_model_update!, field_advection_scheme,
                                compute_closure_tendencies!,
                                closure_scalar_index, skip_vertical_diffusion,
                                implicit_advection_density, implicit_advection_velocities,
                                implicit_step_scheme
using Oceananigans.Utils: launch!, time_difference_seconds
using Oceananigans.TurbulenceClosures: step_closure_prognostics!

"""
$(TYPEDEF)

A strong stability preserving (SSP) Runge-Kutta time stepper with `N` stages, written in
the Shu-Osher form ([Shu and Osher 1988](@cite Shu1988Efficient)) in which every stage
combines the initial state ``u^{(0)}`` with the previous stage and its tendency:

```math
u^{(m)} = (1 - αₘ) u^{(0)} + αₘ u^{(m-1)} + βₘ \\, Δt \\, G(u^{(m-1)}) , \\qquad m = 1, …, N ,
```

where ``G`` is the right-hand side of ``∂_t u = G(u)`` and ``u^{n+1} = u^{(N)}``.
Each stage is a convex combination of ``u^{(0)}`` and a forward Euler step of size
``(βₘ / αₘ) Δt`` from ``u^{(m-1)}``, so the scheme inherits any strong stability property
of forward Euler (total variation diminishing, positivity, ...) under the time step
restriction ``Δt ≤ 𝒞 \\, Δt_{\\mathrm{FE}}``, where ``𝒞 = \\min_m αₘ / βₘ`` is the SSP
coefficient and ``Δt_{\\mathrm{FE}}`` is the forward Euler limit.

Two members are provided:

- [`SSPRungeKutta3`](@ref): three stages, third order, ``𝒞 = 1``.
- [`SSPRungeKutta43`](@ref): four stages, third order, ``𝒞 = 2``.

The tendencies are evaluated at the Butcher abscissae ``cₘ = αₘ cₘ₋₁ + βₘ`` (with
``c₀ = 0``), which the clock follows stage by stage; for both members ``u^{(N-1)}`` sits
at the midpoint of the step and the clock steps *back* by ``Δt / 2`` before the last
stage. This keeps third-order accuracy for a time-dependent right-hand side.

Fields
======

- `α`: Convex combination weights of the previous stage, one per stage
- `β`: Tendency weights, one per stage
- `c`: Butcher abscissa reached after each stage, in units of ``Δt``
- `U⁰`: Storage for the state at the beginning of the time step
- `Gⁿ`: Tendency fields at the current stage
- `implicit_solver`: Optional implicit solver for vertical diffusion
"""
struct SSPRungeKutta{N, FT, U0, TG, TI} <: AbstractTimeStepper
    α :: NTuple{N, FT}
    β :: NTuple{N, FT}
    c :: NTuple{N, FT}
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
end

"""
$(TYPEDSIGNATURES)

Return the Shu-Osher weights `(α, β)` of the `N`-stage SSP Runge-Kutta scheme as tuples
of `FT`, in the form documented for [`SSPRungeKutta`](@ref).
"""
ssp_runge_kutta_weights(::Val{N}, FT) where N =
    throw(ArgumentError("No $(N)-stage SSP Runge-Kutta scheme is defined."))

# SSP RK3 of Shu and Osher (1988): three stages, third order, 𝒞 = 1.
#
#   u¹ = u⁰ + Δt G(u⁰)
#   u² = 3/4 u⁰ + 1/4 u¹ + 1/4 Δt G(u¹)
#   u³ = 1/3 u⁰ + 2/3 u² + 2/3 Δt G(u²)
ssp_runge_kutta_weights(::Val{3}, FT) = (FT.((1, 1//4, 2//3)), FT.((1, 1//4, 2//3)))

# SSP RK(4,3) of Kraaijevanger (1991): four stages, third order, 𝒞 = 2.
#
#   u¹ = u⁰ + 1/2 Δt G(u⁰)
#   u² = u¹ + 1/2 Δt G(u¹)
#   u³ = 2/3 u⁰ + 1/3 u² + 1/6 Δt G(u²)
#   u⁴ = u³ + 1/2 Δt G(u³)
ssp_runge_kutta_weights(::Val{4}, FT) = (FT.((1, 1, 1//3, 1)), FT.((1//2, 1//2, 1//6, 1//2)))

"""
$(TYPEDSIGNATURES)

Return the Butcher abscissae reached after each stage of a Shu-Osher scheme with
weights `α` and `β`: ``cₘ = αₘ cₘ₋₁ + βₘ`` with ``c₀ = 0``.
"""
function ssp_runge_kutta_abscissae(α, β)
    c = map(zero, β)
    cₘ₋₁ = zero(eltype(β))
    for m in eachindex(β)
        cₘ = α[m] * cₘ₋₁ + β[m]
        c = Base.setindex(c, cₘ, m)
        cₘ₋₁ = cₘ
    end
    return c
end

"""
$(TYPEDSIGNATURES)

Construct an `N`-stage [`SSPRungeKutta`](@ref) time stepper on `grid` with `prognostic_fields`.

Keyword Arguments
=================

- `implicit_solver`: Optional implicit solver for diffusion. Default: `nothing`
- `Gⁿ`: Tendency fields at current stage. Default: similar to `prognostic_fields`
- `U⁰`: Storage for the state at the beginning of the step. Default: similar to
  `prognostic_fields`. Accepting it as a keyword lets callers (e.g. the adiabatic-balance
  twin) alias another stepper's tendency storage instead of allocating fresh fields.
"""
function SSPRungeKutta{N}(grid, prognostic_fields;
                          dynamics = nothing,
                          implicit_solver::TI = nothing,
                          cache_advecting_state = false,   # accepted for TimeStepper-call uniformity; unused
                          Gⁿ::TG = map(similar, prognostic_fields),
                          U⁰::U0 = map(similar, prognostic_fields)) where {N, TI, TG, U0}

    FT = eltype(grid)
    α, β = ssp_runge_kutta_weights(Val(N), FT)
    c = ssp_runge_kutta_abscissae(α, β)

    return SSPRungeKutta{N, FT, U0, TG, TI}(α, β, c, U⁰, Gⁿ, implicit_solver)
end

"""
$(TYPEDEF)

The three-stage, third-order SSP Runge-Kutta scheme of
[Shu and Osher (1988)](@cite Shu1988Efficient):

```math
\\begin{align*}
u^{(1)} &= u^{(0)} + Δt \\, G(u^{(0)}) \\\\
u^{(2)} &= \\frac{3}{4} u^{(0)} + \\frac{1}{4} u^{(1)} + \\frac{1}{4} Δt \\, G(u^{(1)}) \\\\
u^{(3)} &= \\frac{1}{3} u^{(0)} + \\frac{2}{3} u^{(2)} + \\frac{2}{3} Δt \\, G(u^{(2)})
\\end{align*}
```

with Butcher abscissae ``c = (0, 1, 1/2)`` and SSP coefficient ``𝒞 = 1``: the scheme is
total variation diminishing whenever forward Euler is, at the same time step.
It is the default time stepper for anelastic dynamics and for compressible dynamics with
explicit time stepping; select it with `timestepper = :SSPRungeKutta3`.

See [`SSPRungeKutta`](@ref) for the constructor keyword arguments and the general
Shu-Osher form.
"""
const SSPRungeKutta3 = SSPRungeKutta{3}

"""
$(TYPEDEF)

The four-stage, third-order SSP Runge-Kutta scheme of
[Kraaijevanger (1991)](@cite Kraaijevanger1991), often written SSPRK(4,3):

```math
\\begin{align*}
u^{(1)} &= u^{(0)} + \\frac{1}{2} Δt \\, G(u^{(0)}) \\\\
u^{(2)} &= u^{(1)} + \\frac{1}{2} Δt \\, G(u^{(1)}) \\\\
u^{(3)} &= \\frac{2}{3} u^{(0)} + \\frac{1}{3} u^{(2)} + \\frac{1}{6} Δt \\, G(u^{(2)}) \\\\
u^{(4)} &= u^{(3)} + \\frac{1}{2} Δt \\, G(u^{(3)})
\\end{align*}
```

with Butcher abscissae ``c = (0, 1/2, 1, 1/2)`` and SSP coefficient ``𝒞 = 2``. Compared
with [`SSPRungeKutta3`](@ref), the extra stage buys a doubling of the SSP time step limit,
so the scheme preserves strong stability at a time step per stage that is ``3/2`` times
larger. Its linear stability region also reaches about ``3/2`` times as far along the
negative real axis per stage (damped modes such as explicit diffusion), while the
imaginary axis extent per stage (pure advection) is about ten percent smaller: the scheme
pays for its strong stability with a slightly tighter linear advective limit per stage.
Schemes of this family are used to good effect in compressible fluid dynamics, for example by
[Ranocha et al. (2021)](@cite Ranocha2021Optimized) and
[Ranocha et al. (2025)](@cite Ranocha2025Error). Select it with
`timestepper = :SSPRungeKutta43`.

See [`SSPRungeKutta`](@ref) for the constructor keyword arguments and the general
Shu-Osher form.
"""
const SSPRungeKutta43 = SSPRungeKutta{4}

AtmosphereModels.timestepper_name(::SSPRungeKutta3) = "SSPRungeKutta3"
AtmosphereModels.timestepper_name(::SSPRungeKutta43) = "SSPRungeKutta43"

#####
##### Stage update kernel
#####

"""
$(TYPEDSIGNATURES)

Apply an SSP Runge-Kutta substep with weights ``α`` and ``β``:
```math
u^{(m)} = (1 - α) u^{(0)} + α u^{(m-1)} + β \\, Δt \\, G
```
where ``u^{(0)}`` is stored in the time stepper, ``u^{(m-1)}`` is the current field value,
and ``G`` is the current tendency. Vertically-implicit diffusion, when present, is applied
over the stage's tendency weight ``β Δt``.
"""
function ssp_runge_kutta_substep!(model, Δt, α, β)
    grid = model.grid
    arch = grid.architecture
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    kernel_Δt = kernel_time_step(arch, grid, Δt)

    prognostic = prognostic_fields(model)
    names = keys(prognostic)

    for (name, u, u⁰, G) in zip(names, prognostic, U⁰, Gⁿ)
        launch!(arch, grid, :xyz, _ssp_runge_kutta_substep!, u, u⁰, G, kernel_Δt, α, β)

        # Dynamics-specific prognostics (the compressible dry density, the kinematic driver's
        # density) are advanced explicitly and have no diffusivity to apply; momentum and every
        # scalar take the solve, momentum with `field_index = nothing` (viscosity) and the
        # scalars with their position in the closure's scalar names (see `closure_scalar_index`).
        skip_vertical_diffusion(model, name) && continue

        field_index = closure_scalar_index(model, name)
        advection = field_advection_scheme(model.advection, name)

        # The implicit solve must carry the reference density whenever it runs at all: the
        # diffusion half is mass-flux weighted for the z-Center prognostics, and adaptive implicit
        # vertical advection adds a density-weighted advection contribution on top.
        #
        # The guard is on the *solver*, not on `needs_implicit_solver(advection)`: that predicate
        # is false for `advection = nothing` and for ordinary WENO, so keying on it would drop the
        # density — and hence the mass-flux weighting — for every vertically-implicit closure
        # without adaptive-implicit advection, the single-column configuration included.
        if !isnothing(model.timestepper.implicit_solver)
            implicit_step!(u,
                           model.timestepper.implicit_solver,
                           model.closure,
                           model.closure_fields,
                           field_index,
                           model.clock,
                           fields(model),
                           β * Δt,
                           implicit_step_scheme(advection),
                           implicit_advection_velocities(model.dynamics, model.velocities, name),
                           implicit_advection_density(model.dynamics, model.formulation, name))
        end
    end

    return nothing
end

@kernel function _ssp_runge_kutta_substep!(u, u⁰, G, Δt, α, β)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # u^(m) = (1 - α) * u^(0) + α * u^(m-1) + β * Δt * G
        u[i, j, k] = (1 - α) * u⁰[i, j, k] + α * u[i, j, k] + β * Δt * G[i, j, k]
    end
end

"""
$(TYPEDSIGNATURES)

Copy prognostic fields to `U⁰` storage for use in later Runge-Kutta stages.
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

Step forward `model` one time step ``Δt`` with an `N`-stage SSP Runge-Kutta method.

Every stage ``m = 1, …, N`` computes the tendency of the current state, applies the
Shu-Osher update

```math
u^{(m)} = (1 - αₘ) u^{(0)} + αₘ u^{(m-1)} + βₘ \\, Δt \\, G(u^{(m-1)}) ,
```

projects the momentum onto the dynamics' constraint over the stage's tendency weight
``βₘ Δt``, and ticks the clock to the stage's Butcher abscissa ``cₘ`` so that the next
tendency is evaluated at the right time. The final stage lands on ``tⁿ + Δt`` exactly.
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:SSPRungeKutta{N}}, Δt; callbacks=[]) where N

    # Be paranoid and prepare at iteration 0, in case run! is not used:
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    ts = model.timestepper
    α = ts.α
    β = ts.β
    c = ts.c

    # Compute the next time step a priori to reduce floating point error accumulation
    tⁿ⁺¹ = model.clock.time + Δt

    # Store u^(0) for use in the later stages
    store_initial_state!(model)

    # The clock sits at cₘ₋₁ Δt on entering stage m. Advancing by (cₘ - cₘ₋₁) Δt puts the
    # tendency evaluation for stage m + 1 at its abscissa; for the last interior stage of
    # both schemes this is a step *back* by Δt/2. Evaluating the last tendency at tⁿ + Δt
    # instead breaks the order conditions (Σbᵢcᵢ ≠ 1/2), and leaves `clock.stage` stuck at
    # N - 1, so per-stage work keyed on `(iteration, stage)`, such as the filtered surface
    # state, skips the last stage.
    cₘ₋₁ = zero(eltype(c))

    for m in 1:N
        # u^(m) = (1 - αₘ) u^(0) + αₘ u^(m-1) + βₘ Δt G(u^(m-1))
        compute_flux_bc_tendencies!(model)
        compute_closure_tendencies!(model)
        ssp_runge_kutta_substep!(model, Δt, α[m], β[m])

        compute_pressure_correction!(model, β[m] * Δt)
        make_pressure_correction!(model, β[m] * Δt)

        # The final stage's tick is the full-step tick below, corrected for round-off
        # so that the step closes on tⁿ⁺¹ exactly.
        m == N && break

        tick_stage!(model.clock, (c[m] - cₘ₋₁) * Δt)
        update_state!(model, callbacks; compute_tendencies = true)
        cₘ₋₁ = c[m]
    end

    # Adjust final time-step to reduce floating point error accumulation
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)

    # Apply the operator-split microphysics update exactly once per step, on the post-RK
    # state just refreshed by `update_state!`. A no-op for tendency-interface schemes.
    microphysics_model_update!(model.microphysics, model)

    # Advect particles once per step, over the full Δt, with the velocity of the state
    # just refreshed to tⁿ⁺¹: Xⁿ⁺¹ = Xⁿ + Δt u(Xⁿ, tⁿ⁺¹) — consistent, but first order,
    # and so lower order than the dycore. A stage-wise update is possible in principle
    # (X obeys dX/dt = u like any prognostic, so the SSP combination applies to it too),
    # but would need Xⁿ stored alongside the current position, since every SSP stage
    # recombines with u⁰. Oceananigans' low-storage RK3 needs no such storage only
    # because its per-stage increments sum to Δt; the SSP tendency weights do not
    # (Σ βₘ = 23/12 for SSP RK3), so pushing with them stage by stage would be wrong.
    step_lagrangian_particles!(model, Δt)

    return nothing
end

Oceananigans.prognostic_state(::SSPRungeKutta) = nothing
Oceananigans.restore_prognostic_state!(timestepper::SSPRungeKutta, ::Nothing) = timestepper
