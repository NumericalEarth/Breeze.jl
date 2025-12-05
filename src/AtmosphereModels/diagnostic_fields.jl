using ..Thermodynamics: Thermodynamics
using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

struct Specific end
struct Density end

struct LiquidIcePotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::LiquidIcePotentialTemperatureKernelFunction) =
    LiquidIcePotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                                adapt(to, k.reference_state),
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
function LiquidIcePotentialTemperature(model, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = LiquidIcePotentialTemperatureKernelFunction(flavor, 
                                                       model.formulation.reference_state,
                                                       model.microphysics,
                                                       model.microphysical_fields,
                                                       model.specific_moisture,
                                                       model.temperature,
                                                       model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    $(TYPEDSIGNATURES)

Return a `Field` representing potential temperature.
"""
LiquidIcePotentialTemperatureField(model, flavor_symbol=:specific) =
    LiquidIcePotentialTemperature(model, flavor_symbol) |> Field

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

    θ = (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π

    if d.flavor isa Specific()
        return θ
    elseif d.flavor isa Density()
        return ρᵣ * θ
    end
end

#####
##### Static energy
#####

struct StaticEnergyKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end
 
Adapt.adapt_structure(to, k::StaticEnergyKernelFunction) =
    StaticEnergyKernelFunction(adapt(to, k.flavor),
                               adapt(to, k.reference_state),
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
function StaticEnergy(model, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = StaticEnergyKernelFunction(flavor,
                                      model.formulation.reference_state,
                                      model.microphysics,
                                      model.microphysical_fields,
                                      model.specific_moisture,
                                      model.temperature,
                                      model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    $(TYPEDSIGNATURES)

Return a `Field` representing potential temperature.
"""
StaticEnergyField(model, flavor_symbol=:specific) =
    StaticEnergy(model, flavor_symbol) |> Field

function (d::StaticEnergyKernelFunction)(i, j, k, grid)
    @inbounds begin
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamic_constants)

    g = d.thermodynamic_constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)

    ℒˡᵣ = d.thermodynamic_constants.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamic_constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # Moist static energy
    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ

    if d.flavor isa Specific
        return e
    elseif d.flavor isa Density
        return ρᵣ * e
    end
end
