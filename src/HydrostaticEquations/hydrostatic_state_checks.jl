#####
##### Checks of the state of HydrostaticDynamics
#####
##### `hydrostatic_state_check` reduces the state to the extrema of every prognostic field, of
##### the pressure thickness of the layers and of the temperatures, and to three verdicts: every
##### prognostic value is a number, every layer has mass, and the temperatures are physical.
##### `accumulate_state_checks` combines the checks of successive checkpoints, so that a claim
##### that a run stayed bounded rests on every checkpoint rather than on the last one.
#####

@inline layer_pressure_thickness(i, j, k, grid, A, B, pˢ) = layer_pressure_thickness(i, j, k, A, B, pˢ)

function field_extrema(field)
    lowest, highest = minimum(field), maximum(field)
    return (convert(Float64, lowest), convert(Float64, highest))
end

within((lowest, highest), (lower_bound, upper_bound)) = lower_bound ≤ lowest && highest ≤ upper_bound

# Potential temperatures of the layer pressures at the bounds of the temperature
function potential_temperature_bounds(model, temperature_bounds)
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    pˢᵗ = model.dynamics.standard_pressure
    p⁻, p⁺ = field_extrema(model.dynamics.pressure)
    T⁻, T⁺ = temperature_bounds
    return (T⁻ * (pˢᵗ / p⁺)^κ, T⁺ * (pˢᵗ / p⁻)^κ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS_NO_RETURN)

Check the state of an `AtmosphereModel` with [`HydrostaticDynamics`](@ref) and return a
`NamedTuple` with

- `time` and `iteration` of the model;
- `extrema`: the smallest and largest value of every prognostic field, of the pressure
  thickness `Δp` of the layers (in Pa), of the potential temperature `θ` and of the
  temperature `T`;
- `finite`: whether every value of every prognostic field is a number;
- `positive_layers`: whether every layer of every column has a positive pressure thickness;
- `physical_temperatures`: whether `T` lies within `temperature_bounds` and `θ` within
  `potential_temperature_bounds`;
- `valid`: whether all three hold.

The diagnostics `θ` and `T` are those of the last `update_state!`, which every time step ends
with. Nothing is thrown: an integration is stopped by the checks of `update_state!` itself, and
this function reports.

Keyword arguments
=================

- `temperature_bounds`: bounds of `T` in K. Default `(150, 350)`, which no dry air of the
  Earth's troposphere and stratosphere leaves.
- `potential_temperature_bounds`: bounds of `θ` in K. By default, the potential temperatures
  that `temperature_bounds` give at the highest and lowest layer pressures of the model.

```jldoctest
using Oceananigans, Breeze
using Breeze.HydrostaticEquations: hydrostatic_state_check

grid = RectilinearGrid(size=(8, 4), x=(0, 8e5), z=(0, 1), topology=(Periodic, Flat, Bounded))
model = AtmosphereModel(grid; dynamics=HydrostaticDynamics(top_pressure=200), momentum_advection=VectorInvariant())
set!(model; θ=300)
check = hydrostatic_state_check(model)
(check.valid, check.extrema.pˢ, check.extrema.θ)

# output
(true, (101325.0, 101325.0), (300.0, 300.0))
```
"""
function hydrostatic_state_check(model::HydrostaticModel;
                                 temperature_bounds = (150, 350),
                                 potential_temperature_bounds = potential_temperature_bounds(model, temperature_bounds))
    grid = model.grid
    dynamics = model.dynamics
    prognostic = Oceananigans.prognostic_fields(model)

    Δp = KernelFunctionOperation{Center, Center, Center}(layer_pressure_thickness, grid,
                                                         dynamics.pressure_coefficient,
                                                         dynamics.surface_pressure_coefficient,
                                                         dynamics.surface_pressure)

    prognostic_extrema = map(field_extrema, prognostic)
    diagnostic_extrema = (Δp = field_extrema(Δp),
                          θ = field_extrema(model.formulation.potential_temperature),
                          T = field_extrema(model.temperature))

    finite = all(e -> isfinite(e[1]) && isfinite(e[2]), prognostic_extrema)
    positive_layers = diagnostic_extrema.Δp[1] > 0
    physical_temperatures = within(diagnostic_extrema.T, temperature_bounds) &&
                            within(diagnostic_extrema.θ, potential_temperature_bounds)

    return (time = model.clock.time,
            iteration = model.clock.iteration,
            extrema = merge(prognostic_extrema, diagnostic_extrema),
            finite = finite,
            positive_layers = positive_layers,
            physical_temperatures = physical_temperatures,
            valid = finite && positive_layers && physical_temperatures)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS_NO_RETURN)

Combine the [`hydrostatic_state_check`](@ref)s of successive checkpoints of one integration:
the extrema are those over all the checkpoints, and a verdict holds only if it holds at every
one of them. `time` and `iteration` are those of the last checkpoint.
"""
function accumulate_state_checks(checks)
    latest = last(checks)
    names = keys(latest.extrema)
    combined = NamedTuple{names}(map(name -> (minimum(c -> c.extrema[name][1], checks),
                                              maximum(c -> c.extrema[name][2], checks)), names))

    return (time = latest.time,
            iteration = latest.iteration,
            extrema = combined,
            finite = all(c -> c.finite, checks),
            positive_layers = all(c -> c.positive_layers, checks),
            physical_temperatures = all(c -> c.physical_temperatures, checks),
            valid = all(c -> c.valid, checks))
end
