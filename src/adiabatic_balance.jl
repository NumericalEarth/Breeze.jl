#####
##### `set!(model; balance = …)` — in-place adiabatic (FV3 `na_init`) initialization.
#####
##### `balance_adiabatically!` (included earlier) does the spin-up but requires a *stripped*
##### model: no microphysics, no upper sponge, no forcing, and a reversible time stepper. Rather
##### than make the caller hand-build that twin and graft fields back (see the DFI block in
##### NumericalEarth's breeze_downscaling_era5 example), `AdiabaticBalance` + `adiabatic_balance_twin`
##### construct a twin that SHARES all field memory with the production model and steps it in place,
##### so the balanced state lands directly in the production model with no graft and no second field
##### set.
#####

using Oceananigans: Clock, fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: minimum_zspacing
using Oceananigans.TimeSteppers: TimeStepper, update_state!

"""
$(TYPEDSIGNATURES)

Specification for the adiabatic (FV3 `na_init`) initialization run by
`set!(model; balance = AdiabaticBalance(...))`.

Keyword arguments
=================

  * `time_stepping`: the time discretization used for the balance excursion (the sponge is always
    stripped — it is irreversible). Options:
      - `ExplicitTimeStepping()` (default) — replace the model's scheme with fully-explicit
        stepping. Memory-minimal (no acoustic substepper; only the aliased `Gⁿ`/`U⁰` tendency
        storage) and cleanly reversible, but `Δt` is bounded by the vertical acoustic CFL.
      - `nothing` — reuse the model's *native* scheme (e.g. split-explicit). Production-consistent
        numerics, and the larger outer `Δt` the substepping permits, at the cost of rebuilding the
        acoustic substepper's scratch fields.
      - any other time-discretization object — swapped in as-is (sponge stripped).
  * `Δt`: forward/backward step size of the excursion. `nothing` (default) auto-derives the
    vertical-acoustic-CFL step `acoustic_cfl_safety · Δz_min / c` from the grid and analysis sound
    speed — appropriate for `ExplicitTimeStepping`, and a safe (if conservative) outer step for the
    native scheme; pass a number to override (recommended with `time_stepping = nothing`).
  * `cycles`: number of balance cycles (default `1`).
  * `weight`: nudging weight toward the analysis snapshot (default `2` → ⅓ dynamics + ⅔ analysis).
  * `with_moisture`: if `true` (default) the moisture density `ρqᵉ` relaxes with the other
    prognostics (more physically consistent). If `false`, `ρqᵉ` is snapshotted before the balance
    and restored after, so it is preserved exactly — reproducing the grafted DFI that returns only
    `(ρ, ρu, ρv, ρw, ρθ)`.
"""
struct AdiabaticBalance{T, S}
    Δt :: T
    cycles :: Int
    weight :: Float64
    with_moisture :: Bool
    time_stepping :: S
end

AdiabaticBalance(; Δt = nothing, cycles = 1, weight = 2, with_moisture = true,
                 time_stepping = ExplicitTimeStepping()) =
    AdiabaticBalance(Δt, cycles, Float64(weight), with_moisture, time_stepping)

# Conservative fraction of the vertical acoustic CFL used for the auto-derived balance Δt.
const acoustic_cfl_safety = 0.85

# Resolve the balance step size: honor an explicit `Δt`, else derive it from the vertical acoustic
# CFL on the smallest cell using the (warmest, hence fastest-sound) analysis temperature.
resolve_balance_Δt(spec::AdiabaticBalance, model) = resolve_balance_Δt(spec.Δt, model)
resolve_balance_Δt(Δt, model) = Δt

function resolve_balance_Δt(::Nothing, model)
    grid      = model.grid
    constants = model.thermodynamic_constants
    Rᵈ = Thermodynamics.dry_air_gas_constant(constants)
    cᵖ = constants.dry_air.heat_capacity
    γ  = cᵖ / (cᵖ - Rᵈ)
    T★ = maximum(interior(model.temperature))   # warmest column → fastest sound → safest Δt
    c  = sqrt(γ * Rᵈ * T★)
    return convert(eltype(grid), acoustic_cfl_safety * minimum_zspacing(grid) / c)
end

# Resolve the time discretization of the balance twin: `nothing` reuses the model's native scheme,
# anything else is swapped in. The sponge is always stripped (it breaks reversibility).
twin_time_discretization(::Nothing, model) = CompressibleEquations.without_sponge(model.dynamics.time_discretization)
twin_time_discretization(time_stepping, model) = CompressibleEquations.without_sponge(time_stepping)

"""
$(TYPEDSIGNATURES)

Build a stripped adiabatic twin of `model` that SHARES all field memory (momentum, velocities,
densities, ρθ, moisture, temperature, pressure solver, dynamics fields) and steps it in place.

`time_stepping` selects the twin's scheme (see [`AdiabaticBalance`](@ref)); the sponge is always
stripped. Only the pieces that must differ are rebuilt:

  * the dynamics' `time_discretization` is swapped (a fresh immutable wrapper around the *same*
    dynamics fields, via `with_time_discretization`);
  * microphysics is disabled and forcing zeroed;
  * a time stepper matching the twin's discretization is built, its `Gⁿ`/`U⁰` tendency storage
    *aliasing* the production stepper's same-named arrays (the production prognostics are a superset
    of the twin's, with the moisture key re-mapped from the microphysics name, e.g. `:ρqᵉ`, to the
    moistureless `:ρqᵛ`). With `ExplicitTimeStepping` this allocates nothing further; the native
    split-explicit scheme additionally rebuilds the acoustic substepper's scratch fields;
  * a fresh `Clock` (so `balance_adiabatically!`'s clock reset does not touch the production clock).
"""
function adiabatic_balance_twin(model::AtmosphereModel, time_stepping = ExplicitTimeStepping())
    grid        = model.grid
    arch        = model.architecture
    formulation = model.formulation
    constants   = model.thermodynamic_constants

    twin_dynamics      = CompressibleEquations.with_time_discretization(model.dynamics,
                                                                        twin_time_discretization(time_stepping, model))
    twin_microphysics  = nothing
    qᵛ                 = AtmosphereModels.specific_prognostic_moisture(model)
    twin_microphysical = (; qᵛ)

    # Re-map the production moisture key (scheme-dependent, e.g. :ρqᵉ) to the moistureless :ρqᵛ.
    prod_moisture_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    twin_moisture_name = AtmosphereModels.moisture_prognostic_name(twin_microphysics)
    remap(name) = name === twin_moisture_name ? prod_moisture_name : name

    twin_prognostic = AtmosphereModels.collect_prognostic_fields(formulation, twin_dynamics,
                                                                 model.momentum, model.moisture_density,
                                                                 twin_moisture_name, (;), model.tracers)
    twin_names = keys(twin_prognostic)

    Gⁿ = NamedTuple{twin_names}(model.timestepper.Gⁿ[remap(n)] for n in twin_names)
    U⁰ = NamedTuple{twin_names}(model.timestepper.U⁰[remap(n)] for n in twin_names)
    twin_timestepper = TimeStepper(AtmosphereModels.default_timestepper(twin_dynamics),
                                   grid, twin_prognostic;
                                   dynamics = twin_dynamics, Gⁿ, U⁰,
                                   implicit_solver = model.timestepper.implicit_solver)

    # Advection schemes are immutable and shared; just re-key the moisture scheme and drop precip.
    twin_scalar_names = (:ρθ, twin_moisture_name, keys(model.tracers)...)
    twin_advection = merge((; momentum = model.advection.momentum),
                           NamedTuple{twin_scalar_names}(model.advection[remap(n)] for n in twin_scalar_names))

    # Zeroed forcing, keyed exactly as the constructor would for this stripped prognostic set.
    density = AtmosphereModels.dynamics_density(twin_dynamics)
    twin_model_fields = merge(twin_prognostic, fields(formulation), model.velocities,
                              (; T = model.temperature), twin_microphysical)
    twin_forcing = AtmosphereModels.atmosphere_model_forcing(NamedTuple(), twin_prognostic, twin_model_fields,
                                                             grid, model.coriolis, density,
                                                             model.velocities, twin_dynamics, formulation,
                                                             twin_microphysics, qᵛ)

    return AtmosphereModel(arch, grid, Clock(grid),
                           twin_dynamics, formulation, constants,
                           model.momentum, model.moisture_density, model.temperature,
                           model.pressure_solver, model.velocities, model.tracers,
                           nothing, twin_advection, model.coriolis, twin_forcing,
                           twin_microphysics, twin_microphysical, twin_timestepper,
                           model.closure, model.closure_fields, nothing)
end

# `set!(model; balance = …)` hook. `false` → no-op; `true` → default `AdiabaticBalance`.
AtmosphereModels.balance_initial_state!(model, balance::Bool) =
    balance ? AtmosphereModels.balance_initial_state!(model, AdiabaticBalance()) : nothing

function AtmosphereModels.balance_initial_state!(model, spec::AdiabaticBalance)
    model.dynamics isa CompressibleDynamics || throw(ArgumentError(
        "set!(model; balance = …) currently supports only CompressibleDynamics. For other " *
        "dynamics, call balance_adiabatically!(model; Δt, cycles) directly on a model built " *
        "without microphysics, sponge, or forcing."))

    spec.with_moisture || (ρqᵉ₀ = copy(parent(model.moisture_density)))

    twin = adiabatic_balance_twin(model, spec.time_stepping)
    update_state!(twin)
    balance_adiabatically!(twin; Δt = resolve_balance_Δt(spec, model),
                           cycles = spec.cycles, weight = spec.weight)

    spec.with_moisture || (parent(model.moisture_density) .= ρqᵉ₀)
    update_state!(model)
    return nothing
end
