using ..Thermodynamics: Thermodynamics
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

"""
    SaturationSpecificHumidityKernel(temperature)

A callable object for diagnosing the saturation specific humidity field,
given a temperature field, designed for use as a `KernelFunctionOperation`
in Oceananigans. Parameters are captured within the kernel struct for GPU friendliness.
"""
struct SaturationSpecificHumidityKernelFunction{R, T, μ, M, MF, TH}
    reference_state :: R
    temperature :: T
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    thermodynamics :: TH
end

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.reference_state),
                                             adapt(to, k.temperature),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_moisture),
                                             adapt(to, k.thermodynamics))

const C = Center

const SaturationSpecificHumidityOperation = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:SaturationSpecificHumidityKernelFunction}
const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidityOperation}

"""
$(TYPEDSIGNATURES)

Return a field for the saturation specific humidity.
"""
function SaturationSpecificHumidityField(model)
    func = SaturationSpecificHumidityKernelFunction(model.formulation.reference_state,
                                                    model.temperature,
                                                    model.microphysics,
                                                    model.microphysical_fields,
                                                    model.specific_moisture,
                                                    model.thermodynamics)

    op = KernelFunctionOperation{Center, Center, Center}(func, model.grid)

    return Field(op)
end

function (d::SaturationSpecificHumidityKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        T = d.temperature[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
    end
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    ρ = Thermodynamics.density(pᵣ, T, q, d.thermodynamics)
    return Thermodynamics.saturation_specific_humidity(T, ρ, d.thermodynamics, d.thermodynamics.liquid)
end

"""
    PotentialTemperatureKernel(temperature)

A callable object for diagnosing the potential temperature field,
given a temperature field, designed for use as a `KernelFunctionOperation`
in Oceananigans. Follows the pattern used for saturation diagnostics.
"""
struct PotentialTemperatureKernelFunction{R, μ, M, MF, TMP, TH}
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamics :: TH
end

Adapt.adapt_structure(to, k::PotentialTemperatureKernelFunction) =
    PotentialTemperatureKernelFunction(adapt(to, k.reference_state),
                                       adapt(to, k.microphysics),
                                       adapt(to, k.microphysical_fields),
                                       adapt(to, k.specific_moisture),
                                       adapt(to, k.temperature),
                                       adapt(to, k.thermodynamics))

const PotentialTemperatureOperation = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:PotentialTemperatureKernelFunction}
const PotentialTemperatureField = Field{Center, Center, Center, <:PotentialTemperatureOperation}

"""
    $(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing potential temperature.
"""
function PotentialTemperature(model)
    grid = model.grid
    func = PotentialTemperatureKernelFunction(model.formulation.reference_state,
                                              model.microphysics,
                                              model.microphysical_fields,
                                              model.specific_moisture,
                                              model.temperature,
                                              model.thermodynamics)
    return KernelFunctionOperation{Center, Center, Center}(func, grid)
end

"""
    $(TYPEDSIGNATURES)

Return a `Field` representing potential temperature.
"""
PotentialTemperatureField(model) = Field(PotentialTemperature(model))

function (d::PotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    Rᵐ = Thermodynamics.mixture_gas_constant(q, d.thermodynamics)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamics)
    Π = (pᵣ / p₀)^(Rᵐ / cᵖᵐ)

    ℒˡᵣ = d.thermodynamics.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamics.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    return (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π
end
