#####
##### `set!(model; balance = …)` — in-place adiabatic (FV3 `na_init`) initialization.
#####
##### `balance_adiabatically!` (included earlier) does the spin-up but requires a *stripped*
##### model: no microphysics, no upper sponge, no divergence damping, no forcing, and a
##### reversible (explicit) time stepper. Rather than make the caller hand-build that twin and
##### graft fields back (see the DFI block in NumericalEarth's breeze_downscaling_era5 example),
##### `AdiabaticBalance` + `adiabatic_balance_twin` construct a twin that SHARES all field memory
##### with the production model and steps it in place, so the balanced state lands directly in the
##### production model with no graft and no second field set.
#####

using Oceananigans: Clock, fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: minimum_zspacing
using Oceananigans.TimeSteppers: update_state!

"""
$(TYPEDSIGNATURES)

Specification for the adiabatic (FV3 `na_init`) initialization run by
`set!(model; balance = AdiabaticBalance(...))`.

Keyword arguments
=================

  * `Δt`: the explicit forward/backward step size of the balance excursion. The balance twin
    uses `ExplicitTimeStepping`, so `Δt` is bounded by the *vertical acoustic* CFL on the
    smallest cell. `nothing` (default) auto-derives it from the grid spacing and the analysis
    sound speed (`acoustic_cfl_safety · Δz_min / c`); pass a number to override.
  * `cycles`: number of balance cycles (default `1`).
  * `weight`: nudging weight toward the analysis snapshot (default `2` → ⅓ dynamics + ⅔ analysis).
  * `with_moisture`: if `true` (default) the moisture density `ρqᵉ` relaxes with the other
    prognostics (more physically consistent). If `false`, `ρqᵉ` is snapshotted before the balance
    and restored after, so it is preserved exactly — reproducing the grafted DFI that returns only
    `(ρ, ρu, ρv, ρw, ρθ)`.
"""
struct AdiabaticBalance{T}
    Δt :: T
    cycles :: Int
    weight :: Float64
    with_moisture :: Bool
end

AdiabaticBalance(; Δt = nothing, cycles = 1, weight = 2, with_moisture = true) =
    AdiabaticBalance(Δt, cycles, Float64(weight), with_moisture)

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

"""
$(TYPEDSIGNATURES)

Build a stripped adiabatic twin of `model` that SHARES all field memory (momentum, velocities,
densities, ρθ, moisture, temperature, pressure solver, dynamics fields) and steps it in place.

Only the pieces that must differ for a reversible adiabatic excursion are rebuilt, and none
allocates a field array:

  * the dynamics' `time_discretization` is swapped to `ExplicitTimeStepping` (a fresh immutable
    wrapper around the *same* dynamics fields), which carries no acoustic substepper and therefore
    no baked-in sponge or divergence damping;
  * microphysics is disabled and forcing is zeroed;
  * a fresh `SSPRungeKutta3` is built whose `Gⁿ`/`U⁰` tendency storage *aliases* the production
    stepper's same-named arrays (the production prognostics are a superset of the twin's, with the
    moisture key re-mapped from the microphysics name, e.g. `:ρqᵉ`, to the moistureless `:ρqᵛ`);
  * a fresh `Clock` (so `balance_adiabatically!`'s clock reset does not touch the production clock).
"""
function adiabatic_balance_twin(model::AtmosphereModel)
    grid        = model.grid
    arch        = model.architecture
    formulation = model.formulation
    constants   = model.thermodynamic_constants

    twin_dynamics      = CompressibleEquations.with_time_discretization(model.dynamics, ExplicitTimeStepping())
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
    twin_timestepper = TimeSteppers.SSPRungeKutta3(grid, twin_prognostic;
                                                   Gⁿ, U⁰, implicit_solver = model.timestepper.implicit_solver)

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

    twin = adiabatic_balance_twin(model)
    update_state!(twin)
    balance_adiabatically!(twin; Δt = resolve_balance_Δt(spec, model),
                           cycles = spec.cycles, weight = spec.weight)

    spec.with_moisture || (parent(model.moisture_density) .= ρqᵉ₀)
    update_state!(model)
    return nothing
end
