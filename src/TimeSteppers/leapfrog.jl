using KernelAbstractions: @kernel, @index

using Oceananigans: prognostic_fields, fields
using Oceananigans.TimeSteppers:
    AbstractTimeStepper,
    tick_stage!,
    update_state!,
    compute_flux_bc_tendencies!,
    step_lagrangian_particles!

using Breeze.AtmosphereModels: AtmosphereModel, compute_pressure_correction!, make_pressure_correction!,
                                microphysics_model_update!
using Oceananigans.Utils: launch!, time_difference_seconds

"""
$(TYPEDEF)

A three-level **leapfrog** (centered-in-time) time stepper:

```math
u^{n+1} = u^{n-1} + 2 Δt \\, G(u^{n})
```

The leapfrog is **time-symmetric**: `S(-Δt) ∘ S(Δt) = I` on the full tendency `G`, with **no
amplitude error** on oscillatory (acoustic) eigenvalues. This is exactly the reversibility the
adiabatic-balance (`na_init`) DFI excursion requires — the explicit RK3 schemes
(`SSPRungeKutta3`/`AcousticRungeKutta3`) have acoustic amplitude error that accumulates over the
±Δt excursion and drives density negative on energetic (e.g. ERA5-interpolated) initial conditions.
Intended as the balance-twin integrator, **not** for production runs (leapfrog's computational mode
is filtered by the DFI itself; a Robert–Asselin filter would re-introduce the dissipation we are
avoiding, so it is deliberately omitted).

Fields
======

- `Uᵐ`: previous-level state ``u^{n-1}``.
- `U⁰`: start-of-step storage (parity with the other steppers; may alias another stepper's storage).
- `Gⁿ`: tendency fields ``G(u^{n})``.
- `implicit_solver`: unused (kept for interface parity); must be `nothing` for a reversible excursion.
- `started`: bootstrap flag — the first step is a forward Euler half-step to seed ``u^{n-1}``.
"""
mutable struct Leapfrog{FT, UM, U0, TG, TI} <: AbstractTimeStepper
    Uᵐ :: UM
    U⁰ :: U0
    Gⁿ :: TG
    implicit_solver :: TI
    robert_asselin :: FT   # weak RA coefficient ν (0 = pure/reversible; ~0.02 tames the 2Δt mode)
    started :: Bool
end

"""
$(TYPEDSIGNATURES)

Construct a `Leapfrog` stepper. `Gⁿ`/`U⁰` may be supplied to alias another stepper's storage
(used by the adiabatic-balance twin); `Uᵐ` is always allocated fresh (it holds the extra level).
"""
function Leapfrog(grid, prognostic_fields;
                  dynamics = nothing,
                  implicit_solver::TI = nothing,
                  robert_asselin = 0,
                  Gⁿ::TG = map(similar, prognostic_fields),
                  U⁰::U0 = map(similar, prognostic_fields)) where {TI, TG, U0}
    Uᵐ = map(similar, prognostic_fields)
    UM = typeof(Uᵐ)
    ν = convert(eltype(grid), robert_asselin)
    return Leapfrog{typeof(ν), UM, U0, TG, TI}(Uᵐ, U⁰, Gⁿ, implicit_solver, ν, false)
end

@kernel function _leapfrog_step!(u, uᵐ, G, two_Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = uᵐ[i, j, k] + two_Δt * G[i, j, k]
end

@kernel function _euler_step!(u, G, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u[i, j, k] + Δt * G[i, j, k]
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with the leapfrog scheme. On the first step (or the first
after a `started = false` reset, e.g. after a nudge), a forward-Euler half-step seeds ``u^{n-1}``.
"""
function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:Leapfrog}, Δt; callbacks=[])
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    ts   = model.timestepper
    grid = model.grid
    arch = grid.architecture
    Gⁿ   = ts.Gⁿ
    Uᵐ   = ts.Uᵐ
    prog = prognostic_fields(model)
    Δt_FT   = kernel_time_step(arch, grid, Δt)
    twoΔt   = kernel_time_step(arch, grid, 2Δt)

    tⁿ⁺¹ = model.clock.time + Δt

    if !ts.started
        # Forward-Euler bootstrap: save uⁿ⁻¹ ← uⁿ, then uⁿ⁺¹ = uⁿ + Δt G(uⁿ).
        for (u, uᵐ) in zip(prog, Uᵐ); parent(uᵐ) .= parent(u); end
        for (u, G) in zip(prog, Gⁿ)
            launch!(arch, grid, :xyz, _euler_step!, u, G, Δt_FT)
        end
        ts.started = true
    else
        # Leapfrog: uⁿ⁺¹ = uⁿ⁻¹ + 2Δt G(uⁿ). Then cycle uⁿ⁻¹ ← the (optionally Robert–Asselin-filtered)
        # uⁿ: ū_n = uⁿ + ν(uⁿ⁻¹ - 2uⁿ + uⁿ⁺¹) tames the 2Δt computational mode. ν=0 ⇒ pure/reversible.
        ν = ts.robert_asselin
        for (u, uᵐ, G) in zip(prog, Uᵐ, Gⁿ)
            uⁿ    = interior_copy_parent(u)                 # uⁿ
            uⁿ⁻¹  = ν > 0 ? copy(parent(uᵐ)) : uⁿ           # uⁿ⁻¹ (still in Uᵐ; copy before overwrite)
            launch!(arch, grid, :xyz, _leapfrog_step!, u, uᵐ, G, twoΔt)   # u ← uⁿ⁺¹
            parent(uᵐ) .= ν > 0 ? (uⁿ .+ ν .* (uⁿ⁻¹ .- 2 .* uⁿ .+ parent(u))) : uⁿ
        end
    end

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    microphysics_model_update!(model.microphysics, model)
    step_lagrangian_particles!(model, Δt)

    return nothing
end

@inline interior_copy_parent(u) = copy(parent(u))

"""
$(TYPEDSIGNATURES)

Reset the leapfrog bootstrap so the next `time_step!` re-seeds `u^{n-1}` with a forward-Euler
half-step. Call after any out-of-band state modification (e.g. the DFI nudge) that invalidates the
`u^{n-1}` history.
"""
reset_leapfrog!(model) = (model.timestepper.started = false; nothing)

"""
$(TYPEDSIGNATURES)

Turn a leapfrog trajectory around for exact time reversal. The leapfrog map on pairs
`(uⁿ⁻¹, uⁿ) ↦ (uⁿ, uⁿ⁺¹)` is inverted by swapping the two stored levels (current ↔ `Uᵐ`) and
continuing the recurrence with `-Δt`: the subsequent `time_step!(model, -Δt)` then reconstructs
`uⁿ⁻¹ = uⁿ⁺¹ - 2Δt G(uⁿ)`. Refreshes the tendency for the swapped state so the next backward step
evaluates `G` at the correct level. Requires `started == true` (a trajectory to reverse).
"""
function turnaround!(model, callbacks=[])
    ts = model.timestepper
    prog = prognostic_fields(model)
    for (u, uᵐ) in zip(prog, ts.Uᵐ)
        tmp = copy(parent(u))
        parent(u)  .= parent(uᵐ)
        parent(uᵐ) .= tmp
    end
    update_state!(model, callbacks; compute_tendencies = true)
    return nothing
end
