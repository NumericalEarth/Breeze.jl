# Imports are provided by the Diagnostics module

struct SaturationSpecificHumidityKernelFunction{μ, FL, M, MF, T, R, TH}
    flavor :: FL
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

Oceananigans.Utils.prettysummary(kf::SaturationSpecificHumidityKernelFunction) = "$(kf.flavor) SaturationSpecificHumidityKernelFunction"

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

struct PrognosticFlavor end
struct EquilibriumFlavor end
struct TotalMoistureFlavor end

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing the specified flavor
of *saturation specific humidity* ``qᵛ⁺``.

## Flavor options

* `:prognostic`

  Return the *saturation specific humidity* corresponding to the `model`'s prognostic state.
  This is the same as the equilibrium saturation specific humidity for saturated conditions
  and a model that uses saturation adjustment microphysics.

* `:equilibrium`

  Return the *saturation specific humidity* in potentially-saturated conditions, using the
  `model.specific_moisture`. This is equivalent to the `:total_moisture` flavor
  under saturated conditions with no condensate; or in other words, if `model.specific_moisture` happens
  to be equal to the saturation specific humidity.

* `:total_moisture`

  Return *saturation specific humidity* in the case that the total specific moisture is
  equal to the saturation specific humidity and there is no condensate.
  This is useful for manufacturing perfectly saturated initial conditions.
"""
function SaturationSpecificHumidity(model, flavor_symbol=:prognostic)

    flavor = if flavor_symbol == :prognostic
        PrognosticFlavor()
    elseif flavor_symbol == :equilibrium
        EquilibriumFlavor()
    elseif flavor_symbol == :total_moisture
        TotalMoistureFlavor()
    else
        valid_flavors = (:prognostic, :equilibrium, :total_moisture)
        throw(ArgumentError("Flavor $flavor_symbol is not one of the valid flavors $valid_flavors"))
    end

    func = SaturationSpecificHumidityKernelFunction(flavor,
                                                    model.microphysics,
                                                    model.microphysical_fields,
                                                    model.specific_moisture,
                                                    model.temperature,
                                                    model.dynamics.reference_state,
                                                    model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

#####
##### Helper functions for computing saturation specific humidity
#####

# Get phase equilibrium from microphysics
# For microphysics without a specific equilibrium model, default to warm phase (liquid only)
# SaturationAdjustment extends this in Microphysics to return μ.equilibrium
@inline microphysics_phase_equilibrium(μ) = WarmPhaseEquilibrium()

# Compute saturation specific humidity for :total_moisture flavor
# This is the saturation specific humidity when qᵗ = qᵛ⁺ (all moisture is vapor)
@inline function saturation_total_specific_moisture(T, pᵣ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    δᵈᵛ = Rᵈ / Rᵛ - 1
    return pᵛ⁺ / (pᵣ + δᵈᵛ * pᵛ⁺)
end

#####
##### Kernel function implementation
#####

function (d::SaturationSpecificHumidityKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    equilibrium = microphysics_phase_equilibrium(d.microphysics)
    surface = equilibrated_surface(equilibrium, T)

    if d.flavor isa PrognosticFlavor
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
        ρ = density(T, pᵣ, q, constants)
        return saturation_specific_humidity(T, ρ, constants, surface)

    elseif d.flavor isa EquilibriumFlavor
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        return equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, surface)

    elseif d.flavor isa TotalMoistureFlavor
        return saturation_total_specific_moisture(T, pᵣ, constants, surface)

    end
end

const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidity}
SaturationSpecificHumidityField(model, flavor_symbol=:prognostic) = Field(SaturationSpecificHumidity(model, flavor_symbol))

