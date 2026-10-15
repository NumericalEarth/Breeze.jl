using Oceananigans.AbstractOperations: ∂z
using Oceananigans.Fields: FractionalIndices, Center
using Breeze.ParcelModels: ParcelModels
using Breeze.Thermodynamics: Thermodynamics as TD

@inline function parcel_temperature_tendency(state::TD.StaticEnergyState, constants, w, pressure_tendency)
    return -constants.gravitational_acceleration * w / TD.mixture_heat_capacity(state.moisture_mass_fractions, constants)
end

@inline function parcel_temperature_tendency(state::TD.LiquidIcePotentialTemperatureState, constants, w, pressure_tendency)
    q = state.moisture_mass_fractions
    κ = TD.mixture_gas_constant(q, constants) / TD.mixture_heat_capacity(q, constants)
    return state.potential_temperature * TD.exner_function(state, constants) * κ * pressure_tendency / state.reference_pressure
end

# Thermodynamic derivatives follow the same conserved variable as ParcelModel.
@inline function parcel_phase_temperature_tendency(state::TD.LiquidIcePotentialTemperatureState,
                                                   constants, vapor, liquid, ice)
    q = state.moisture_mass_fractions
    cp = TD.mixture_heat_capacity(q, constants)
    R = TD.mixture_gas_constant(q, constants)
    cp_tendency = parcel_heat_capacity_tendency(constants, vapor, liquid, ice)
    R_tendency = parcel_gas_constant_tendency(constants, vapor, liquid, ice)
    latent = constants.liquid.reference_latent_heat * q.liquid + constants.ice.reference_latent_heat * q.ice
    latent_tendency = constants.liquid.reference_latent_heat * liquid + constants.ice.reference_latent_heat * ice
    dry_temperature = state.potential_temperature * TD.exner_function(state, constants)
    return dry_temperature * log(state.reference_pressure / state.standard_pressure) *
           (R_tendency / cp - R * cp_tendency / cp^2) + latent_tendency / cp - latent * cp_tendency / cp^2
end

@inline function parcel_phase_temperature_tendency(state::TD.StaticEnergyState, constants, vapor, liquid, ice)
    cp = TD.mixture_heat_capacity(state.moisture_mass_fractions, constants)
    heating = TD.absolute_zero_latent_heat(constants, constants.liquid) * liquid +
              TD.absolute_zero_latent_heat(constants, constants.ice) * ice
    return (heating - TD.temperature(state, constants) * parcel_heat_capacity_tendency(constants, vapor, liquid, ice)) / cp
end

@inline parcel_heat_capacity_tendency(constants, vapor, liquid, ice) =
    (constants.vapor.heat_capacity - constants.dry_air.heat_capacity) * vapor +
    (constants.liquid.heat_capacity - constants.dry_air.heat_capacity) * liquid +
    (constants.ice.heat_capacity - constants.dry_air.heat_capacity) * ice

@inline parcel_gas_constant_tendency(constants, vapor, liquid, ice) =
    (TD.vapor_gas_constant(constants) - TD.dry_air_gas_constant(constants)) * vapor -
    TD.dry_air_gas_constant(constants) * (liquid + ice)

@inline parcel_tendency_component(result, name::Val, supersaturation) = p3_tendency_component(result, name)
@inline parcel_tendency_component(result, ::Val{:ρsᵛ⁺ˡ}, supersaturation) = supersaturation

function ParcelModels.compute_microphysics_prognostic_tendencies(p3::P3, ρ, μ, ℳ, 𝒰, constants, model)
    state = model.dynamics.state
    # Differentiate the same piecewise-linear profile used to update pressure.
    # Interpolating the face gradients instead would smooth the derivative and
    # introduce a spurious source near a boundary or a change in profile slope.
    # TODO: expose derivatives of interpolated fields in Oceananigans.
    index = FractionalIndices((state.z,), model.grid, Center(), Center(), Center()).k
    k = floor(Int, index)
    pressure_tendency = ℳ.w * ∂z(model.dynamics.pressure)[1, 1, k+1]
    temperature_tendency = parcel_temperature_tendency(𝒰, constants, ℳ.w, pressure_tendency)
    T = TD.temperature(𝒰, constants)
    properties = p3_process_properties(p3, ρ, ℳ)
    result = p3_tendency_compute(p3, ρ, ℳ, 𝒰, constants, properties, T, temperature_tendency, zero(ρ))

    # Integrate s = q_v - e_sat(T) R_m / (p R_v) with the chain rule.
    # A finite-τ estimate of an end-of-step state is not its instantaneous RHS.
    # Including both resolved expansion and phase-change heating preserves this
    # identity to the accuracy of the existing RK scheme, without resetting s.
    vapor = result.tendency_ρqᵛ / ρ
    liquid = (result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqʷⁱ) / ρ
    ice = result.tendency_ρqⁱ / ρ
    total_temperature_tendency = temperature_tendency + parcel_phase_temperature_tendency(𝒰, constants, vapor, liquid, ice)
    R = TD.mixture_gas_constant(𝒰.moisture_mass_fractions, constants)
    R_tendency = parcel_gas_constant_tendency(constants, vapor, liquid, ice)
    saturation = TD.saturation_specific_humidity(T, ρ, constants, TD.PlanarLiquidSurface())
    derivative = TD.liquid_saturation_vapor_pressure_derivative(T, constants)
    saturation_tendency = derivative * total_temperature_tendency / (ρ * TD.vapor_gas_constant(constants) * T) +
                          saturation * (R_tendency / R - pressure_tendency / 𝒰.reference_pressure)
    supersaturation_tendency = ρ * (vapor - saturation_tendency)
    tendencies = map(keys(μ)) do name
        parcel_tendency_component(result, Val(name), supersaturation_tendency)
    end
    return NamedTuple{keys(μ)}(tendencies)
end
