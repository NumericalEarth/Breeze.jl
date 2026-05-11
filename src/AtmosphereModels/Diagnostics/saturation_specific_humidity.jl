# Imports are provided by the Diagnostics module

struct SaturationSpecificHumidityKernelFunction{μ, FL, M, MF, T, R, TH}
    flavor :: FL
    microphysics :: μ
    microphysical_fields :: M
    specific_prognostic_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

Oceananigans.Utils.prettysummary(kf::SaturationSpecificHumidityKernelFunction) = "$(kf.flavor) SaturationSpecificHumidityKernelFunction"

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.flavor),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_prognostic_moisture),
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
  model's specific moisture field. This is equivalent to the `:total_moisture` flavor
  under saturated conditions with no condensate; or in other words, if the specific moisture
  happens to be equal to the saturation specific humidity.

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
                                                    specific_prognostic_moisture(model),
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

"""
$(TYPEDSIGNATURES)

Compute the *saturation total specific moisture* under the assumption that all moisture is vapor at saturation, 
``qᵗ = qᵛ⁺``. With this assumption, the equation of state for moist air can be solved in closed form, yielding an
expression for the saturation specific humidity in terms of temperature `T` and reference pressure `pᵣ` alone:

```math
qᵛ⁺ = \\frac{ϵᵈᵛ \\, pᵛ⁺(T)}{pᵣ + δᵈᵛ \\, pᵛ⁺(T)} ,
```

where ``ϵᵈᵛ ≡ Rᵈ / Rᵛ ≈ 0.622`` and ``δᵈᵛ ≡ ϵᵈᵛ - 1 ≈ -0.378``.

The resulting expression coincides with the saturation specific humidity used in the COARE 3.6 [Edson (2013)](@cite Edson2013) 
and Large-Yeager [(2004)](@cite LargeYeager2004) ocean bulk-flux algorithms, where the air-side specific humidity at the surface 
is unknown a priori and [`saturation_specific_humidity`](@ref Breeze.Thermodynamics.saturation_specific_humidity) cannot be 
evaluated directly.

See the [Atmosphere Thermodynamics](@ref Thermodynamics-section) section of the documentation for a derivation.
"""
@inline function saturation_total_specific_moisture(T, pᵣ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ϵᵈᵛ = Rᵈ / Rᵛ
    δᵈᵛ = ϵᵈᵛ - 1
    return ϵᵈᵛ * pᵛ⁺ / (pᵣ + δᵈᵛ * pᵛ⁺)
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
        qᵛᵉ = @inbounds d.specific_prognostic_moisture[i, j, k]
        q = grid_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵛᵉ, d.microphysical_fields)
        ρ = density(T, pᵣ, q, constants)
        return saturation_specific_humidity(T, ρ, constants, surface)

    elseif d.flavor isa EquilibriumFlavor
        qᵛᵉ = @inbounds d.specific_prognostic_moisture[i, j, k]
        return equilibrium_saturation_specific_humidity(T, pᵣ, qᵛᵉ, constants, surface)

    elseif d.flavor isa TotalMoistureFlavor
        return saturation_total_specific_moisture(T, pᵣ, constants, surface)

    end
end

const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidity}
SaturationSpecificHumidityField(model, flavor_symbol=:prognostic) = Field(SaturationSpecificHumidity(model, flavor_symbol))
