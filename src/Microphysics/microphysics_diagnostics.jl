using Adapt: Adapt, adapt
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, Center

using Breeze.Thermodynamics:
    saturation_specific_humidity,
    dry_air_gas_constant,
    vapor_gas_constant,
    density,
    saturation_vapor_pressure

struct SaturationSpecificHumidityKernelFunction{μ, FL, M, MF, T, R, TH}
    flavor :: FL
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.flavor),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_moisture),
                                             adapt(to, k.temperature),
                                             adapt(to, k.reference_state),
                                             adapt(to, k.thermodynamic_constants))

const C = Center
const SaturationSpecificHumidity = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:SaturationSpecificHumidityKernelFunction}
const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidity}

struct Prognostic end
struct Equilibrium end
struct TotalMoisture end

"""
$(TYPEDSIGNATURES)

Return a field computing the specified flavor of *saturation specific humidity* ``qᵛ⁺``.

## Flavor options

### `:prognostic`

Returns the *saturation specific humidity* corresponding to the `model`'s prognostic state.
This is the same as the equilibrium saturation specific humidity for saturated conditions
and a model that uses saturation adjustment microphysics.

### `:equilibrium`

Returns the *saturation specific humidity* in saturated conditions, using the
`model.specific_moisture`. This is equivalent to the `:total_moisture` flavor
under saturated conditions with no condensate; or in other words, if `model.specific_moisture` happens
to be equal to the saturation specific humidity.

### `:total_moisture`

Returns *saturation specific humidity* in the case that the total specific moisture is
equal to the saturation specific humidity and there is no condensate.
This is useful for manufacturing perfectly saturated initial conditions.

## Examples

```@example ssh
using Breeze
grid = RectilinearGrid(size=(4, 4, 4), extent=(500, 500, 1000))
model = AtmosphereModel(grid)
set!(model, θ=300)
qᵛ⁺ = SaturationSpecificHumidity(model, :prognostic)
```

Equilibrium flavor

```@example ssh
qᵛ⁺ₑ = SaturationSpecificHumidity(model, :equilibrium)
```

Equilibrium flavor

```@example ssh
qᵛ = SaturationSpecificHumidity(model, :total_moisture)
```
"""
function SaturationSpecificHumidity(model, flavor_symbol=:prognostic)

    flavor = if flavor_symbol == :prognostic
        Prognostic()
    elseif flavor_symbol == :equilibrium
        Equilibrium()
    elseif flavor_symbol == :total_moisture
        TotalMoisture()
    else
        valid_flavors = (:prognostic, :equilibrium, :total_moisture)
        throw(ArgumentError("Flavor $flavor_symbol is not one of the valid flavors $valid_flavors"))
    end

    func = SaturationSpecificHumidityKernelFunction(flavor,
                                                    model.microphysics,
                                                    model.microphysical_fields,
                                                    model.specific_moisture,
                                                    model.temperature,
                                                    model.formulation.reference_state,
                                                    model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

@inline function saturation_total_specific_moisture(T, pᵣ, constants, equil)
    surface = equilibrated_surface(equil, T)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    δᵈᵛ = Rᵈ / Rᵛ - 1
    return pᵛ⁺ / (pᵣ + δᵈᵛ * pᵛ⁺)
end

const AdjustmentSH = SaturationSpecificHumidityKernelFunction{<:SaturationAdjustment}

function (d::AdjustmentSH)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    equil = d.microphysics.equilibrium

    if d.flavor isa Prognostic
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
        ρ = density(pᵣ, T, q, constants)
        surface = equilibrated_surface(equil, T)
        return saturation_specific_humidity(T, ρ, constants, surface)

    elseif d.flavor isa Equilibrium
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        return equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equil)

    elseif d.flavor isa TotalMoisture
        return saturation_total_specific_moisture(T, pᵣ, constants, equil)

    end
end

SaturationSpecificHumidityField(model, flavor_symbol=:prognostic) = Field(SaturationSpecificHumidity(model, flavor_symbol))
