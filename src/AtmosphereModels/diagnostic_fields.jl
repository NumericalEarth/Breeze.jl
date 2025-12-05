using ..Thermodynamics: Thermodynamics
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

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

Return a `KernelFunctionOperation` representing liquid-ice potential temperature.
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

#####
##### Static energy
#####

struct StaticEnergyKernelFunction{R, μ, M, MF, TMP, TH}
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::StaticEnergyKernelFunction) =
    StaticEnergyKernelFunction(adapt(to, k.reference_state),
                               adapt(to, k.microphysics),
                               adapt(to, k.microphysical_fields),
                               adapt(to, k.specific_moisture),
                               adapt(to, k.temperature),
                               adapt(to, k.thermodynamic_constants))

const StaticEnergy = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:StaticEnergyKernelFunction}
const StaticEnergyField = Field{Center, Center, Center, <:StaticEnergy}

"""
    $(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing potential temperature.
"""
function StaticEnergy(model)
    grid = model.grid
    func = StaticEnergyKernelFunction(model.formulation.reference_state,
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
StaticEnergyField(model) = Field(StaticEnergy(model))

function (d::StaticEnergyKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamic_constants)

    z = znode(i, j, k, grid, c, c, c)
    g = d.thermodynamic_constants.gravitational_acceleration
    ℒˡᵣ = d.thermodynamic_constants.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamic_constants.ice.reference_latent_heat
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    qˡ = q.liquid
    qⁱ = q.ice

    return cᵖᵐ * T + g * z - ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ
end
