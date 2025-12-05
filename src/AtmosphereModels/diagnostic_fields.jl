using ..Thermodynamics: Thermodynamics
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

#=
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
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.reference_state),
                                             adapt(to, k.temperature),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_moisture),
                                             adapt(to, k.thermodynamic_constants))

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
                                                    model.thermodynamic_constants)

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
    ρ = Thermodynamics.density(pᵣ, T, q, d.thermodynamic_constants)
    return Thermodynamics.saturation_specific_humidity(T, ρ, d.thermodynamic_constants, d.thermodynamic_constants.liquid)
end
=#

"""
    PotentialTemperatureKernel(temperature)

A callable object for diagnosing the potential temperature field,
given a temperature field, designed for use as a `KernelFunctionOperation`
in Oceananigans. Follows the pattern used for saturation diagnostics.
"""
struct LiquidIcePotentialTemperatureKernelFunction{R, μ, M, MF, TMP, TH}
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::LiquidIcePotentialTemperatureKernelFunction) =
    LiquidIcePotentialTemperatureKernelFunction(adapt(to, k.reference_state),
                                       adapt(to, k.microphysics),
                                       adapt(to, k.microphysical_fields),
                                       adapt(to, k.specific_moisture),
                                       adapt(to, k.temperature),
                                       adapt(to, k.thermodynamic_constants))

const LiquidIcePotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:LiquidIcePotentialTemperatureKernelFunction}
const LiquidIcePotentialTemperatureField = Field{Center, Center, Center, <:LiquidIcePotentialTemperature}

"""
    $(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing potential temperature.
"""
function LiquidIcePotentialTemperature(model)
    grid = model.grid
    func = LiquidIcePotentialTemperatureKernelFunction(model.formulation.reference_state,
                                              model.microphysics,
                                              model.microphysical_fields,
                                              model.specific_moisture,
                                              model.temperature,
                                              model.thermodynamic_constants)
    return KernelFunctionOperation{Center, Center, Center}(func, grid)
end

"""
    $(TYPEDSIGNATURES)

Return a `Field` representing potential temperature.
"""
LiquidIcePotentialTemperatureField(model) = Field(LiquidIcePotentialTemperature(model))

function (d::LiquidIcePotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    Rᵐ = Thermodynamics.mixture_gas_constant(q, d.thermodynamic_constants)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamic_constants)
    Π = (pᵣ / p₀)^(Rᵐ / cᵖᵐ)

    ℒˡᵣ = d.thermodynamic_constants.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamic_constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    return (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π
end
