#####
##### Adiabatic (FV3 `na_init`) initialization.
#####
##### Two layers:
#####   * `balance_adiabatically!(model; Δt, cycles, weight)` — the low-level, solver-agnostic
#####     spin-up. It requires a *stripped* model (no microphysics, closure, sponge, or forcing, and a
#####     reversible time stepper), since dissipative/irreversible terms corrupt the symmetric
#####     forward/backward excursion.
#####   * `balance_adiabatically!(model, balancer)` / `set!(model; balancer = …)` — the high-level
#####     entry point. An `AdiabaticBalancer` drives `adiabatic_balance_twin`, which builds a stripped
#####     twin that SHARES all field memory with the production model and steps it in place, so the
#####     balanced state lands directly in `model` with no graft and no second field set.
#####

using Oceananigans: time_step!, Centered
using Oceananigans.Grids: minimum_zspacing
using Oceananigans.TimeSteppers: reset!

"""
$(TYPEDSIGNATURES)

Spin up a balanced vertical momentum `ρw` (and the nonhydrostatic pressure balance) consistent with
`model`'s initial (analysis) state, via FV3 adiabatic initialization (`na_init`).

Analyses (ERA5, GFS, …) supply the density, momentum, and thermodynamic state but cold-start the
vertical velocity `w` at zero (hydrostatic), so the nonhydrostatic state is out of balance with the
rest. Each of `cycles` cycles entails two symmetric forward/backward dynamics excursions at the same
`Δt`. After each excursion — which lets `ρw` develop — the *initial fields* (every prognostic except
`ρw`) are nudged back toward their `t = 0` snapshot by the weighted mean

    x ← (x + weight·x₀) / (1 + weight)

(default `weight = 2` → ⅓ dynamics + ⅔ snapshot). `ρw` is never snapshotted or nudged, so the
balance the excursion imprints on it is exactly what is kept. `update_state!` after each nudge
rebuilds the diagnostics; the clock is reset to `t = 0` on exit.

`balance_adiabatically!` performs *adiabatic* dynamics only. The caller must pass a model built
without physics (`microphysics = nothing`), without an upper sponge, and without forcing — these run
inside `update_state!`/`time_step!` and would corrupt the spin-up. Boundary conditions are not
modified; pass a model whose boundaries are time-invariant so the symmetric excursion stays nearly
reversible. The two-argument [`balance_adiabatically!`](@ref)`(model, balancer)` constructs such a
model automatically.
"""
# Lynch–Huang digital-filter weights: a Lanczos-windowed ideal low-pass, symmetric in `k = 0…span`
# (h₋ₖ = hₖ), cutoff period `τc`. Modes with period < τc (the fast acoustic branch) are removed;
# slower (balanced) modes pass. Normalized so h₀ + 2 Σₖ hₖ = 1.
function dfi_lanczos_weights(span::Int, Δt, τc)
    θc = 2π * abs(Δt) / τc                       # digital cutoff (rad/step)
    h  = zeros(Float64, span + 1)
    h[1] = θc / π
    for k in 1:span
        lanczos = sin(k * π / (span + 1)) / (k * π / (span + 1))   # window
        h[k + 1] = sin(k * θc) / (k * π) * lanczos
    end
    h ./= (h[1] + 2 * sum(@view h[2:end]))
    return h
end

# Reset the leapfrog bootstrap so the next `time_step!` re-seeds uⁿ⁻¹ with a forward-Euler half-step.
# No-op for non-leapfrog steppers, which carry no bootstrap state (e.g. the anelastic twin's native
# projection RK3), so the low-level DFI stays solver-agnostic.
restart_bootstrap!(timestepper) = hasfield(typeof(timestepper), :started) && (timestepper.started = false)

function balance_adiabatically!(model::AtmosphereModel; Δt, span::Int = 16, cutoff_period = 2 * span * abs(Δt))
    prog = prognostic_fields(model)
    U⁰   = map(f -> copy(parent(f)), prog)
    h    = dfi_lanczos_weights(span, Δt, cutoff_period)

    # Symmetric Lynch–Huang filter over a reversible (leapfrog) trajectory: seed with h₀·u(0), run
    # `span` forward and `span` backward leapfrog steps from u(0), accumulating hₖ·u(±kΔt). The
    # leapfrog is exactly reversible (no acoustic amplitude error) so each run stays bounded; the
    # low-pass removes the fast acoustic modes, leaving the balanced state (incl. a consistent ρw).
    acc = map(u0 -> h[1] .* u0, U⁰)

    restart_bootstrap!(model.timestepper)        # (re)bootstrap the leapfrog for a fresh run
    for k in 1:span
        time_step!(model, +Δt)
        for (a, f) in zip(acc, prog); a .+= h[k + 1] .* parent(f); end
    end

    for (f, u0) in zip(prog, U⁰); parent(f) .= u0; end   # restore u(0)
    update_state!(model)
    restart_bootstrap!(model.timestepper)
    for k in 1:span
        time_step!(model, -Δt)
        for (a, f) in zip(acc, prog); a .+= h[k + 1] .* parent(f); end
    end

    for (f, a) in zip(prog, acc); parent(f) .= a; end     # install the filtered (balanced) state
    update_state!(model)

    # Full reset (time, iteration, stage, last_Δt, …): the excursion leaves these dirty, misfiring
    # e.g. the filtered surface state's `isinf(last_Δt)` guard.
    reset!(model.clock)

    return model
end

"""
$(TYPEDEF)

Sentinel for `AdiabaticBalancer`'s default `time_stepping`: the fully-explicit twin for
`CompressibleDynamics`, and the native scheme for solvers without a separable time discretization
(e.g. `AnelasticDynamics`). It lets the default avoid naming a concrete time discretization, whose
type lives in a submodule loaded after `AtmosphereModels`.
"""
struct DefaultTimeStepping end

"""
$(TYPEDEF)

Configuration for adiabatic (FV3 `na_init`) initialization, applied with
`balance_adiabatically!(model, balancer)` or `set!(model; balancer = AdiabaticBalancer(...))`.
Works for both `CompressibleDynamics` and `AnelasticDynamics`.

Keyword arguments
=================

  * `time_stepping`: the time discretization used for the balance excursion (the sponge is always
    stripped — it is irreversible). **`CompressibleDynamics` only**; ignored for `AnelasticDynamics`
    (which has a single projection-based scheme). Options:
      - default (`DefaultTimeStepping()`) — fully-explicit stepping. Memory-minimal (no acoustic
        substepper; only the aliased `Gⁿ`/`U⁰` tendency storage) and cleanly reversible, but `Δt` is
        bounded by the vertical acoustic CFL.
      - `nothing` — reuse the model's *native* scheme (e.g. split-explicit), at the cost of
        rebuilding the acoustic substepper's scratch fields.
      - any time-discretization object — swapped in as-is.
  * `Δt`: forward/backward step size. `nothing` (default) auto-derives the vertical-acoustic-CFL step
    `acoustic_cfl_safety · Δz_min / c` from the grid and analysis sound speed; pass a number to
    override.
  * `span`: one-sided digital-filter window `N` (default `16`) — the number of leapfrog steps run
    forward and backward from the analysis state.
  * `cutoff`: Lynch–Huang cutoff period `τc`. `nothing` (default) auto-derives `2·span·|Δt|` (modes
    faster than the excursion window are filtered); pass a number to override.
  * `with_moisture`: if `true` (default) the moisture density `ρqᵉ` is filtered with the other
    prognostics. If `false`, `ρqᵉ` is snapshotted before the balance and restored after, so it is
    preserved exactly — reproducing a graft that returns only `(ρ, ρu, ρv, ρw, ρθ)`.
"""
struct AdiabaticBalancer{T, C, S}
    Δt :: T
    span :: Int
    cutoff :: C
    robert_asselin :: Float64
    with_moisture :: Bool
    time_stepping :: S
end

AdiabaticBalancer(; Δt = nothing, span = 16, cutoff = nothing, robert_asselin = 0.02,
                  with_moisture = true, time_stepping = DefaultTimeStepping()) =
    AdiabaticBalancer(Δt, span, cutoff, Float64(robert_asselin), with_moisture, time_stepping)

# Fraction of the vertical acoustic CFL used for the auto-derived balance Δt. The leapfrog excursion
# is neutrally stable only for the acoustic Courant number ≤ 1 and — being reversible — provides NO
# damping, so any cell that exceeds the limit grows its acoustic/gravity oscillation over the span
# until a step drives ρ or θ transiently negative (crashing the Exner `θ^γ` mid-excursion). The
# nominal `minimum_zspacing` also understates the true spacing in terrain-metric-compressed columns,
# pushing the effective Courant number higher there. So this is deliberately well below 1 (≈2.5×
# headroom) to keep every cell — including steep-terrain ones — inside the leapfrog stability region.
# Override per-call with `AdiabaticBalancer(Δt = …)` (the balance-Δt sweep lever).
const acoustic_cfl_safety = 0.4

# Resolve the balance step size: honor an explicit `Δt`, else derive it from the vertical acoustic
# CFL on the smallest cell using the (warmest, hence fastest-sound) analysis temperature.
resolve_balance_Δt(balancer::AdiabaticBalancer, model) = resolve_balance_Δt(balancer.Δt, model)
resolve_balance_Δt(Δt, model) = Δt

function resolve_balance_Δt(::Nothing, model)
    grid      = model.grid
    constants = model.thermodynamic_constants
    Rᵈ  = Thermodynamics.dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵛᵈ = cᵖᵈ - Rᵈ
    γᵈ  = cᵖᵈ / cᵛᵈ
    T★  = maximum(model.temperature)   # warmest column → fastest sound → safest Δt
    c   = sqrt(γᵈ * Rᵈ * T★)
    return convert(eltype(grid), acoustic_cfl_safety * minimum_zspacing(grid) / c)
end

"""
$(TYPEDSIGNATURES)

Run adiabatic (FV3 `na_init`) initialization on `model` in place: spin the nonhydrostatic state
(`ρw` and the pressure balance) into balance with the analysis fields. `balancer` is an
[`AdiabaticBalancer`](@ref) (or `true` for the defaults / `false` for a no-op). Builds a stripped,
memory-sharing twin via [`adiabatic_balance_twin`](@ref) and runs the low-level
[`balance_adiabatically!`](@ref)`(model; Δt, cycles, weight)` on it, so the balanced state lands
directly in `model` — no graft, no second field set.
"""
function balance_adiabatically!(model::AtmosphereModel, balancer::AdiabaticBalancer)
    balancer.with_moisture || (ρqᵉ₀ = copy(parent(model.moisture_density)))

    twin = adiabatic_balance_twin(model, balancer)
    # The Robert–Asselin coefficient is a leapfrog concept (compressible twin); non-acoustic solvers
    # keep their native projection stepper, which carries no such field.
    if hasfield(typeof(twin.timestepper), :robert_asselin)
        twin.timestepper.robert_asselin = convert(eltype(twin.grid), balancer.robert_asselin)
    end
    update_state!(twin)
    Δt_b = resolve_balance_Δt(balancer, model)
    τc   = isnothing(balancer.cutoff) ? 2 * balancer.span * abs(Δt_b) : balancer.cutoff
    balance_adiabatically!(twin; Δt = Δt_b, span = balancer.span, cutoff_period = τc)

    balancer.with_moisture || (parent(model.moisture_density) .= ρqᵉ₀)
    update_state!(model)
    return model
end

balance_adiabatically!(model::AtmosphereModel, balancer::Bool) =
    balancer ? balance_adiabatically!(model, AdiabaticBalancer()) : model

"""
$(TYPEDSIGNATURES)

Return the dynamics for the adiabatic-balance twin, given the production `dynamics` and the
requested `time_stepping`. The generic fallback reuses `dynamics` unchanged — correct for solvers
without a separable time discretization (e.g. `AnelasticDynamics`) and for any future solver, keeping
the balance solver-agnostic. `CompressibleDynamics` extends this to swap the time discretization
(sponge always stripped, as it is irreversible).
"""
adiabatic_twin_dynamics(dynamics, time_stepping) = dynamics

"""
$(TYPEDSIGNATURES)

Return the time stepper for the adiabatic-balance twin, given the resolved `twin_dynamics`. The
generic fallback keeps the solver's native scheme (`default_timestepper`) — it carries the pressure
projection the excursion needs and is correct for solvers *without* acoustic modes (e.g.
`AnelasticDynamics`, whose projection must run every step). `CompressibleDynamics` overrides this
with a reversible **leapfrog**: the acoustic branch is exactly where the RK3 schemes' amplitude
error accumulates over the ±Δt digital-filter excursion and drives density negative, so acoustic
reversibility is the whole point. `Gⁿ`/`U⁰` alias the production stepper's tendency storage.
"""
adiabatic_twin_timestepper(twin_dynamics, grid, twin_prognostic; Gⁿ, U⁰) =
    TimeStepper(Val(adiabatic_twin_timestepper_symbol(twin_dynamics)), grid, twin_prognostic;
                dynamics = twin_dynamics, Gⁿ, U⁰, implicit_solver = nothing)

# Which registered stepper the twin uses. Native scheme by default (carries the projection needed by
# non-acoustic solvers); `CompressibleDynamics` overrides this to the reversible `:Leapfrog`.
adiabatic_twin_timestepper_symbol(twin_dynamics) = default_timestepper(twin_dynamics)

"""
$(TYPEDSIGNATURES)

Build a stripped adiabatic twin of `model` that SHARES all field memory (momentum, velocities,
densities, ρθ, moisture, temperature, pressure solver, dynamics fields) and steps it in place. The
twin's dynamics comes from [`adiabatic_twin_dynamics`](@ref) (per `balancer.time_stepping`);
microphysics, closure, the implicit diffusion solver, the sponge, and forcing — all
dissipative/irreversible — are removed; the time stepper's `Gⁿ`/`U⁰` tendency storage *aliases* the
production stepper's same-named arrays (moisture key re-mapped from the microphysics name, e.g.
`:ρqᵉ`, to the moistureless `:ρqᵛ`); and a fresh `Clock` is used so the balance's clock reset does
not touch the production clock.
"""
adiabatic_balance_twin(model::AtmosphereModel, balancer::AdiabaticBalancer = AdiabaticBalancer()) =
    assemble_adiabatic_twin(model, adiabatic_twin_dynamics(model.dynamics, balancer.time_stepping))

function assemble_adiabatic_twin(model::AtmosphereModel, twin_dynamics)
    grid        = model.grid
    arch        = model.architecture
    formulation = model.formulation
    constants   = model.thermodynamic_constants

    # microphysics = nothing on the twin: rebuild the thermodynamic-density field's surface-flux BCs to
    # match (else the shared BCs call the production scheme on the twin's stripped field set → error).
    twin_formulation = adiabatic_twin_formulation(formulation)

    twin_microphysics  = nothing
    qᵛ                 = specific_prognostic_moisture(model)
    twin_microphysical = (; qᵛ)

    # Re-map the production moisture key (scheme-dependent, e.g. :ρqᵉ) to the moistureless :ρqᵛ.
    prod_moisture_name = moisture_prognostic_name(model.microphysics)
    twin_moisture_name = moisture_prognostic_name(twin_microphysics)
    remap(name) = name === twin_moisture_name ? prod_moisture_name : name

    twin_prognostic = collect_prognostic_fields(twin_formulation, twin_dynamics, model.momentum,
                                                model.moisture_density, twin_moisture_name, (;), model.tracers)
    twin_names = keys(twin_prognostic)

    Gⁿ = NamedTuple{twin_names}(model.timestepper.Gⁿ[remap(n)] for n in twin_names)
    U⁰ = NamedTuple{twin_names}(model.timestepper.U⁰[remap(n)] for n in twin_names)
    # No implicit_solver: vertically-implicit diffusion is irreversible (it amplifies on the −Δt
    # step), and the closure is stripped below — the excursion must be adiabatic. The stepper is
    # dynamics-dependent (see `adiabatic_twin_timestepper`): a reversible leapfrog for the acoustic
    # `CompressibleDynamics`, the native projection scheme for `AnelasticDynamics`/others.
    twin_timestepper = adiabatic_twin_timestepper(twin_dynamics, grid, twin_prognostic; Gⁿ, U⁰)

    # Advection schemes are immutable and shared; just re-key the moisture scheme and drop precip.
    twin_scalar_names = (:ρθ, twin_moisture_name, keys(model.tracers)...)
    # Centered (non-dissipative ⇒ reversible) advection for momentum AND scalars: WENO's grid-scale
    # upwind anti-diffusion is the other irreversibility source (it breaks the DFI's reversibility on
    # the backward run). The balanced state is handed back to the production model, which keeps WENO.
    centered = Centered(order = 2)
    twin_advection = merge((; momentum = centered),
                           NamedTuple{twin_scalar_names}(centered for _ in twin_scalar_names))

    # Zeroed forcing, keyed exactly as the constructor would for this stripped prognostic set.
    density = dynamics_density(twin_dynamics)
    twin_model_fields = merge(twin_prognostic, fields(twin_formulation), model.velocities,
                              (; T = model.temperature), twin_microphysical)
    twin_forcing = atmosphere_model_forcing(NamedTuple(), twin_prognostic, twin_model_fields,
                                            grid, model.coriolis, density,
                                            model.velocities, twin_dynamics, formulation,
                                            twin_microphysics, qᵛ)

    # closure = nothing: turbulent diffusion is dissipative (irreversible), so it is excluded from the
    # adiabatic excursion. `closure_fields` is carried along but unused when closure is nothing.
    return AtmosphereModel(arch, grid, Clock(grid),
                           twin_dynamics, twin_formulation, constants,
                           model.momentum, model.moisture_density, model.temperature,
                           model.pressure_solver, model.velocities, model.tracers,
                           nothing, twin_advection, model.coriolis, twin_forcing,
                           twin_microphysics, twin_microphysical, twin_timestepper,
                           nothing, model.closure_fields, nothing)
end
