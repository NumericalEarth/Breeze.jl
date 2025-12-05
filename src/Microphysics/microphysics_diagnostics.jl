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
    pressure :: R
    thermodynamics :: TH
end

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.flavor),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_moisture),
                                             adapt(to, k.temperature),
                                             adapt(to, k.pressure),
                                             adapt(to, k.thermodynamics))

const C = Center
const SaturationSpecificHumidity = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:SaturationSpecificHumidityKernelFunction}

struct Prognostic end
struct Equilibrium end
struct TotalMoisture end

"""
$(TYPEDSIGNATURES)
Return a field for the saturation specific humidity.
"""
function SaturationSpecificHumidity(model, flavor_symbol=:prognostic)

    flavor = if flavor_symbol == :prognostic
        Prognostic()
    elseif flavor_symbol == :equilibrium
        Equilibrium()
    elseif flavor_symbol == :total_moisture
        TotalMoisture()
    else
        throw(ArgumentError("Invalid flavor: $flavor_symbol"))
    end

    pressure = model.formulation.reference_state.pressure

    func = SaturationSpecificHumidityKernelFunction(flavor,
                                                    model.microphysics,
                                                    model.microphysical_fields,
                                                    model.specific_moisture,
                                                    model.temperature,
                                                    pressure,
                                                    model.thermodynamics)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

@inline function saturation_total_specific_moisture(T, p, thermo, equil)
    surface = equilibrated_surface(equil, T)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, surface)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    δᵈᵛ = Rᵈ / Rᵛ - 1
    return pᵛ⁺ / (p + δᵈᵛ * pᵛ⁺)
end

const AdjustmentSH = SaturationSpecificHumidityKernelFunction{<:SaturationAdjustment}

function (d::AdjustmentSH)(i, j, k, grid)
    @inbounds begin
        p = d.pressure[i, j, k]
        T = d.temperature[i, j, k]
    end

    thermo = d.thermodynamics
    equil = d.microphysics.equilibrium

    if d.flavor isa Prognostic
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρ, qᵗ, d.microphysical_fields)
        ρ = density(p, T, q, thermo)
        surface = equilibrated_surface(equil, T)
        return saturation_specific_humidity(T, ρ, thermo, surface)

    elseif d.flavor isa Equilibrium
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        return equilibrium_saturation_specific_humidity(T, p, qᵗ, thermo, equil)

    elseif d.flavor isa TotalMoisture
        return saturation_total_specific_moisture(T, p, thermo, equil)

    end
end

const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidity}
SaturationSpecificHumidityField(model, flavor_symbol=:prognostic) = Field(SaturationSpecificHumidity(model, flavor_symbol))
