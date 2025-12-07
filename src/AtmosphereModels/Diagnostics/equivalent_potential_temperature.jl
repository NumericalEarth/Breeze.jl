#####
##### Equivalent potential temperature
#####

struct EquivalentPotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::EquivalentPotentialTemperatureKernelFunction) =
    EquivalentPotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                                 adapt(to, k.reference_state),
                                                 adapt(to, k.microphysics),
                                                 adapt(to, k.microphysical_fields),
                                                 adapt(to, k.specific_moisture),
                                                 adapt(to, k.temperature),
                                                 adapt(to, k.thermodynamic_constants))

const EquivalentPotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:EquivalentPotentialTemperatureKernelFunction}

"""
    EquivalentPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing equivalent potential temperature ``θ_e``.

Equivalent potential temperature is conserved during moist adiabatic processes
(including condensation and evaporation) and is useful for identifying air masses
and diagnosing moist instabilities. It is the temperature that a parcel would have
if all its moisture were condensed out and the resulting latent heat used to warm
the parcel, followed by adiabatic expansion to a reference pressure.

We use the formulation from [BryanFritsch2002](@citet), which provides an accurate
approximation:

```math
θ_e = T \\left( \\frac{p_0}{p_d} \\right)^{R^d / c_{pm}}
      \\exp \\left( \\frac{ℒ_v q^v}{c_{pm} T} \\right)
```

where ``T`` is temperature, ``p_d`` is dry air pressure, ``p_0`` is the reference pressure,
``ℒ_v`` is the latent heat of vaporization, ``q^v`` is the vapor specific humidity,
and ``c_{pm}`` is the heat capacity of the moist air mixture.

The formulation follows equation (39) of [BryanFritsch2002](@cite), which was
adapted from the derivation in [DurranKlemp1982](@citet).

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θ_e``, or `:density` to return ``ρ θ_e``.

# Examples

Equivalent potential temperature is larger than dry potential temperature when moisture
is present (due to the latent heat of condensation):

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θₑ = EquivalentPotentialTemperature(model)
θ = DryPotentialTemperature(model)
θₑ_field = compute!(Field(θₑ))
θ_field = compute!(Field(θ))
minimum(θₑ_field) > minimum(θ_field)

# output
true
```

# References

* [BryanFritsch2002](@cite)
* [DurranKlemp1982](@cite)
"""
function EquivalentPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = EquivalentPotentialTemperatureKernelFunction(flavor,
                                                        model.formulation.reference_state,
                                                        model.microphysics,
                                                        model.microphysical_fields,
                                                        model.specific_moisture,
                                                        model.temperature,
                                                        model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

function (d::EquivalentPotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    thermo = d.thermodynamic_constants
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)

    # Heat capacity and gas constants
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, thermo)

    # Dry air pressure: pᵈ = pᵣ - pᵛ, where pᵛ = ρ qᵛ Rᵛ T
    # Using the approximation pᵈ ≈ pᵣ * (1 - qᵛ * Rᵛ / Rᵐ)
    qᵛ = q.vapor
    Rᵐ = Thermodynamics.mixture_gas_constant(q, thermo)
    pᵈ = pᵣ * (1 - qᵛ * Rᵛ / Rᵐ)

    # Latent heat of vaporization at temperature T
    # ℒᵛ = ℒᵛᵣ + (cᵖᵛ - cˡ)(T - Tᵣ)
    ℒᵛᵣ = thermo.liquid.reference_latent_heat
    cᵖᵛ = thermo.vapor.heat_capacity
    cˡ = thermo.liquid.heat_capacity
    Tᵣ = thermo.energy_reference_temperature
    ℒᵛ = ℒᵛᵣ + (cᵖᵛ - cˡ) * (T - Tᵣ)

    # Equivalent potential temperature following Bryan & Fritsch (2002) Eq. (39)
    # θₑ = T * (p₀ / pᵈ)^(Rᵈ / cᵖᵐ) * exp(ℒᵛ * qᵛ / (cᵖᵐ * T))
    θₑ = T * (p₀ / pᵈ)^(Rᵈ / cᵖᵐ) * exp(ℒᵛ * qᵛ / (cᵖᵐ * T))

    if d.flavor isa Specific
        return θₑ
    elseif d.flavor isa Density
        return ρᵣ * θₑ
    end
end

