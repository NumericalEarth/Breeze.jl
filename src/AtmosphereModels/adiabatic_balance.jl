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

using Oceananigans: time_step!
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
function balance_adiabatically!(model::AtmosphereModel; Δt, cycles = 1, weight = 2)
    snapshot = snapshot_initial_fields(model)

    for _ in 1:cycles
        # Half-cycle A: 0 → +Δt → 0, then nudge.
        time_step!(model, +Δt)
        time_step!(model, -Δt)
        nudge_initial_fields!(model, snapshot, weight)
        update_state!(model)

        # Half-cycle B: 0 → -Δt → 0, then nudge.
        time_step!(model, -Δt)
        time_step!(model, +Δt)
        nudge_initial_fields!(model, snapshot, weight)
        update_state!(model)
    end

    # Full reset (time, iteration, stage, last_Δt, …): the excursion leaves these dirty, misfiring
    # e.g. the filtered surface state's `isinf(last_Δt)` guard.
    reset!(model.clock)

    return model
end

# The prognostic fields nudged back toward the analysis each cycle: every prognostic except the
# vertical momentum `ρw`, which is the free field that spins up balance with them. Solver-agnostic —
# for `CompressibleDynamics` this is `(ρ, ρu, ρv, ρθ, ρqᵉ)`; for `AnelasticDynamics`, `(ρu, ρv, ρθ,
# ρqᵉ)` (the reference density `ρᵣ` is not prognostic and so is absent from `prognostic_fields`).
initial_fields(model::AtmosphereModel) = values(Base.structdiff(prognostic_fields(model), NamedTuple{(:ρw,)}))

# Copy the initial fields' full (haloed) parent arrays at t = 0.
snapshot_initial_fields(model::AtmosphereModel) = map(f -> copy(parent(f)), initial_fields(model))

# In-place weighted blend of each initial field toward its snapshot: x ← (x + weight·x₀) / (1 + weight).
function nudge_initial_fields!(model::AtmosphereModel, snapshot, weight)
    w = convert(eltype(model.grid), weight)
    for (f, x₀) in zip(initial_fields(model), snapshot)
        p = parent(f)
        @. p = (p + w * x₀) / (1 + w)
        fill_halo_regions!(f)
    end
    return nothing
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
  * `cycles`: number of balance cycles (default `1`).
  * `weight`: nudging weight toward the analysis snapshot (default `2` → ⅓ dynamics + ⅔ analysis).
  * `with_moisture`: if `true` (default) the moisture density `ρqᵉ` relaxes with the other
    prognostics. If `false`, `ρqᵉ` is snapshotted before the balance and restored after, so it is
    preserved exactly — reproducing a graft that returns only `(ρ, ρu, ρv, ρw, ρθ)`.
"""
struct AdiabaticBalancer{T, S}
    Δt :: T
    cycles :: Int
    weight :: Float64
    with_moisture :: Bool
    time_stepping :: S
end

AdiabaticBalancer(; Δt = nothing, cycles = 1, weight = 2, with_moisture = true,
                  time_stepping = DefaultTimeStepping()) =
    AdiabaticBalancer(Δt, cycles, Float64(weight), with_moisture, time_stepping)

# Conservative fraction of the vertical acoustic CFL used for the auto-derived balance Δt.
const acoustic_cfl_safety = 0.85

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
    update_state!(twin)
    balance_adiabatically!(twin; Δt = resolve_balance_Δt(balancer, model),
                           cycles = balancer.cycles, weight = balancer.weight)

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

Build a stripped adiabatic twin of `model` that SHARES all field memory (momentum, velocities,
densities, ρθ, moisture, tracers, temperature, pressure solver, dynamics fields) and steps it in
place. Every prognostic scalar — momentum, ρθ/ρe, moisture, and tracers — is rewrapped with its
surface fluxes stripped to no-flux (see [`adiabatic_scalar_bcs`](@ref)), sharing the production data
so no memory is reallocated. The twin's dynamics comes from [`adiabatic_twin_dynamics`](@ref) (per
`balancer.time_stepping`); microphysics, closure, the implicit diffusion solver, the sponge, and
forcing — all dissipative/irreversible — are removed; the time stepper's `Gⁿ`/`U⁰` tendency storage
*aliases* the production stepper's same-named arrays (moisture key re-mapped from the microphysics
name, e.g. `:ρqᵉ`, to the moistureless `:ρqᵛ`); and a fresh `Clock` is used so the balance's clock
reset does not touch the production clock.
"""
adiabatic_balance_twin(model::AtmosphereModel, balancer::AdiabaticBalancer = AdiabaticBalancer()) =
    assemble_adiabatic_twin(model, adiabatic_twin_dynamics(model.dynamics, balancer.time_stepping))

# Rewrap a prognostic scalar `twin_field` with its surface fluxes stripped to no-flux (see
# [`adiabatic_scalar_bcs`](@ref)), sharing the production `data`, indices, and operand so no memory
# is reallocated and the balanced state still lands in `model`.
adiabatic_field(twin_field) =
    Oceananigans.Field(Oceananigans.instantiated_location(twin_field), twin_field.grid, twin_field.data,
                       adiabatic_scalar_bcs(twin_field.boundary_conditions),
                       twin_field.indices, twin_field.operand, twin_field.status)

function assemble_adiabatic_twin(model::AtmosphereModel, twin_dynamics)
    grid        = model.grid
    arch        = model.architecture
    constants   = model.thermodynamic_constants

    # Strip surface fluxes from every one of the twin's prognostic scalars. The excursion is
    # adiabatic — pure, reversible dynamics — so surface heating (bulk sensible-heat / energy / θ
    # flux), surface moisture sources (vapor flux), surface tracer fluxes, and surface momentum
    # drag (dissipative, like closure) must all be absent. Each field is rewrapped sharing the
    # production data (no reallocation), so the balanced state still lands in `model`.
    formulation = model.formulation
    ρᵡ = thermodynamic_density(formulation)
    formulation = with_thermodynamic_density(formulation, adiabatic_field(ρᵡ))

    twin_momentum         = NamedTuple{keys(model.momentum)}(adiabatic_field(ρu) for ρu in model.momentum)
    twin_moisture_density = adiabatic_field(model.moisture_density)
    twin_tracers          = NamedTuple{keys(model.tracers)}(adiabatic_field(c) for c in model.tracers)

    twin_microphysics  = nothing
    qᵛ                 = specific_prognostic_moisture(model)
    twin_microphysical = (; qᵛ)

    # Re-map the production moisture key (scheme-dependent, e.g. :ρqᵉ) to the moistureless :ρqᵛ.
    prod_moisture_name = moisture_prognostic_name(model.microphysics)
    twin_moisture_name = moisture_prognostic_name(twin_microphysics)
    remap(name) = name === twin_moisture_name ? prod_moisture_name : name

    twin_prognostic = collect_prognostic_fields(formulation, twin_dynamics, twin_momentum,
                                                twin_moisture_density, twin_moisture_name, (;), twin_tracers)
    twin_names = keys(twin_prognostic)

    Gⁿ = NamedTuple{twin_names}(model.timestepper.Gⁿ[remap(n)] for n in twin_names)
    U⁰ = NamedTuple{twin_names}(model.timestepper.U⁰[remap(n)] for n in twin_names)
    # No implicit_solver: vertically-implicit diffusion is irreversible (it amplifies on the −Δt
    # step), and the closure is stripped below — the excursion must be adiabatic.
    twin_timestepper = TimeStepper(default_timestepper(twin_dynamics), grid, twin_prognostic;
                                   dynamics = twin_dynamics, Gⁿ, U⁰, implicit_solver = nothing)

    # Advection schemes are immutable and shared; just re-key the moisture scheme and drop precip.
    twin_scalar_names = (:ρθ, twin_moisture_name, keys(twin_tracers)...)
    twin_advection = merge((; momentum = model.advection.momentum),
                           NamedTuple{twin_scalar_names}(model.advection[remap(n)] for n in twin_scalar_names))

    # Zeroed forcing, keyed exactly as the constructor would for this stripped prognostic set.
    density = dynamics_density(twin_dynamics)
    twin_model_fields = merge(twin_prognostic, fields(formulation), model.velocities,
                              (; T = model.temperature), twin_microphysical)
    twin_forcing = atmosphere_model_forcing(NamedTuple(), twin_prognostic, twin_model_fields,
                                            grid, model.coriolis, density,
                                            model.velocities, twin_dynamics, formulation,
                                            twin_microphysics, qᵛ)

    # closure = nothing: turbulent diffusion is dissipative (irreversible), so it is excluded from the
    # adiabatic excursion. `closure_fields` is carried along but unused when closure is nothing.
    return AtmosphereModel(arch, grid, Clock(grid),
                           twin_dynamics, formulation, constants,
                           twin_momentum, twin_moisture_density, model.temperature,
                           model.pressure_solver, model.velocities, twin_tracers,
                           nothing, twin_advection, model.coriolis, twin_forcing,
                           twin_microphysics, twin_microphysical, nothing, twin_timestepper,
                           nothing, model.closure_fields, nothing)
end
