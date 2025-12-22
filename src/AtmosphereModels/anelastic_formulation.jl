"""
$(TYPEDSIGNATURES)

Construct an anelastic formulation for `AtmosphereModel` using the given `reference_state`.

Keyword arguments
=================

- `thermodynamics`: The thermodynamic formulation to use. Can be:
  - `:LiquidIcePotentialTemperature` or `:θ` - uses liquid-ice potential temperature density `ρθ` (default)
  - `:StaticEnergy` or `:e` - uses static energy density `ρe` as prognostic variable

Returns a `NamedTuple` with `dynamics` and `thermodynamic_formulation` that can be passed
to `AtmosphereModel` via the `formulation` keyword argument.

Example
=======

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 2, 3));

julia> constants = ThermodynamicConstants();

julia> reference_state = ReferenceState(grid, constants);

julia> formulation = AnelasticFormulation(reference_state);

julia> model = AtmosphereModel(grid; formulation)
AtmosphereModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── dynamics: AnelasticDynamics(p₀=101325.0, θ₀=288.0)
├── formulation: LiquidIcePotentialTemperatureFormulation
├── timestepper: RungeKutta3TimeStepper
├── advection scheme:
│   ├── momentum: Centered(order=2)
│   ├── ρθ: Centered(order=2)
│   └── ρqᵗ: Centered(order=2)
├── tracers: ()
├── coriolis: Nothing
└── microphysics: Nothing
```
"""
function AnelasticFormulation(reference_state; thermodynamics = :LiquidIcePotentialTemperature)
    dynamics = AnelasticDynamics(reference_state)
    thermodynamic_formulation = convert_thermodynamics_symbol(thermodynamics)
    return (; dynamics, thermodynamic_formulation)
end

# Convert thermodynamics symbols to formulation stubs
# Note: these are stubs with `nothing` fields that get materialized during AtmosphereModel construction
function convert_thermodynamics_symbol(thermodynamics::Symbol)
    if thermodynamics ∈ (:StaticEnergy, :e, :ρe)
        return StaticEnergyFormulation(nothing, nothing)
    elseif thermodynamics ∈ (:LiquidIcePotentialTemperature, :θ, :ρθ, :PotentialTemperature)
        return LiquidIcePotentialTemperatureFormulation(nothing, nothing)
    else
        throw(ArgumentError("Unknown thermodynamics formulation: $thermodynamics. " *
                            "Valid options are :StaticEnergy, :LiquidIcePotentialTemperature."))
    end
end

# Allow passing a formulation type directly
convert_thermodynamics_symbol(formulation::StaticEnergyFormulation) = formulation
convert_thermodynamics_symbol(formulation::LiquidIcePotentialTemperatureFormulation) = formulation

