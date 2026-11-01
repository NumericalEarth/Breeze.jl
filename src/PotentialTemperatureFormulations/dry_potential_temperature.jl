#####
##### The closure of the dry potential temperature θ = T / Π, Π = (p / pˢᵗ)^κᵈ
#####
#
# The first law at fixed volume and composition gives cᵛᵐ dT = − Rᵐ T ∇⋅𝐮 dt + Q dt, and d ln p =
# d ln T − ∇⋅𝐮 dt, so that with θ = T / Π
#
#   Dθ/Dt = − θ (Rᵐ − κᵈ cᵖᵐ) / cᵛᵐ ∇⋅𝐮 + (1 − κᵈ) Q / (cᵛᵐ Π)        (`IsochoricPressureWork`)
#
# the form of CM1's `eqtset = 2` (`solve2.F:410`, `solve1.F:945`, `radiation_driver.F:794`). The other way
# to write the same equation holds the pressure fixed through the heating,
#
#   Dθ/Dt = θ (κᵐ − κᵈ) D ln p / Dt + Q / (cᵖᵐ Π)                     (`IsobaricPressureWork`)
#
# with the total pressure tendency, as for θˡⁱ. The two agree exactly with the exact
# D ln p / Dt = (cᵖᵐ / cᵛᵐ) (− ∇⋅𝐮 + Q / (cᵖᵐ T) + D ln Rᵐ / Dt), so the pairs must not be mixed: the
# divergence alone leaves out the expansion that the heating drives, and the isobaric coefficient
# already includes it.

@inline dry_kappa(constants) = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity

# A state at absolute zero (before `set!`) has no response.
@inline heating_response(::Dry{<:IsochoricPressureWork}, θ, T, q, exponent, constants) =
    (1 - dry_kappa(constants)) * θ / ((mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)) * ifelse(T == 0, one(T), T))

@inline heating_response(::Dry{<:IsobaricPressureWork}, θ, T, q, exponent, constants) =
    θ / (mixture_heat_capacity(q, constants) * ifelse(T == 0, one(T), T))

"""
$(TYPEDSIGNATURES)

Return ``S = θ (Rᵐ - κᵈ cᵖᵐ) / cᵛᵐ``, the coefficient of the velocity divergence in the dry
potential temperature equation, ``Dθ/Dt ∋ - S ∇ ⋅ 𝐮`` (`thterm` in CM1).
"""
@inline function divergence_expansion_coefficient(θ, q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return θ * (Rᵐ - dry_kappa(constants) * cᵖᵐ) / (cᵖᵐ - Rᵐ)
end

# The divergence the slow tendency can see: `CompressibleEquations` makes this zero on the
# split-explicit core, whose expansion term is applied after the acoustic loop.
@inline explicit_velocity_divergence(i, j, k, grid, dynamics, velocities) =
    divᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

@inline expansion_tendency(i, j, k, grid, ::Dry{<:IsochoricPressureWork}, dynamics, ρᵈ, θ, T, q, exponent, constants, velocities, model_fields) =
    - ρᵈ * divergence_expansion_coefficient(θ, q, constants) * explicit_velocity_divergence(i, j, k, grid, dynamics, velocities)

@inline function expansion_tendency(i, j, k, grid, ::Dry{<:IsobaricPressureWork}, dynamics, ρᵈ, θ, T, q, exponent, constants, velocities, model_fields)
    κᵐ = mixture_gas_constant(q, constants) / mixture_heat_capacity(q, constants)
    Dt_ln_p = pressure_logarithm_tendency(i, j, k, grid, dynamics, constants, model_fields.w)
    return ρᵈ * θ * (κᵐ - dry_kappa(constants)) * Dt_ln_p
end

#####
##### Phase change
#####
#
# Water changes phase at fixed density and internal energy, ρ𝓊 = (ρᵈ cᵛᵈ + ρqᵛ cᵛᵛ + ρqˡ cˡ + ρqⁱ cⁱ) T + ℒˡ₀ ρqᵛ + (ℒˡ₀ − ℒⁱ₀) ρqⁱ,
# so ρ cᵛᵐ Ṫ = − Σ eˣ Ṡˣ with eᵛ = cᵛᵛ T + ℒˡ₀, eˡ = cˡ T, eⁱ = cⁱ T + ℒˡ₀ − ℒⁱ₀. With the equation of state
# p = (ρᵈ Rᵈ + ρqᵛ Rᵛ) T, d ln θ = (1 − κᵈ) d ln T − κᵈ d ln(ρ Rᵐ), which gives
#
#   ρ θ̇ = − Θ_ρ [ (1 − κᵈ) Σ eˣ Ṡˣ / (cᵛᵐ T) + κᵈ (Rᵛ / Rᵐ) Ṡᵛ ],     Θ_ρ = ρᵈ θ / ρ ,
#
# the tendency form of CM1's `kessler.F:95-97`. At fixed pressure instead, ρ cᵖᵐ Ṫ = − Σ hˣ Ṡˣ with
# hᵛ = cᵖᵛ T + ℒˡ₀, hˡ = cˡ T, hⁱ = cⁱ T + ℒˡ₀ − ℒⁱ₀, and ρ θ̇ = − Θ_ρ Σ hˣ Ṡˣ / (cᵖᵐ T).

"""
$(TYPEDSIGNATURES)

Return the weights `(Wᵛ, Wˡ, Wⁱ)` of the phase-change source of the dry potential temperature density,
``ρθ̇ = Wᵛ Ṡᵛ + Wˡ Ṡˡ + Wⁱ Ṡⁱ``, given the temperature `T`, the mass fractions `q` and the ratio
`Θ_ρ = ρᵈ θ / ρ`. The microphysical tendencies ``Ṡ`` of the water densities are assumed to conserve water.
"""
@inline function phase_change_weights(::Dry{<:IsochoricPressureWork}, T, q, ρ, Θ_ρ, pˢᵗ, exponent, constants)
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    κᵈ = dry_kappa(constants)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵛᵐ = mixture_heat_capacity(q, constants) - Rᵐ

    a = (1 - κᵈ) / (cᵛᵐ * T)
    Wᵛ = - Θ_ρ * (a * ((cᵖᵛ - Rᵛ) * T + ℒˡ₀) + κᵈ * Rᵛ / Rᵐ)
    Wˡ = - Θ_ρ * a * cˡ * T
    Wⁱ = - Θ_ρ * a * (cⁱ * T + ℒˡ₀ - ℒⁱ₀)
    return Wᵛ, Wˡ, Wⁱ
end

@inline function phase_change_weights(::Dry{<:IsobaricPressureWork}, T, q, ρ, Θ_ρ, pˢᵗ, exponent, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    a = Θ_ρ / (mixture_heat_capacity(q, constants) * T)
    Wᵛ = - a * (constants.vapor.heat_capacity * T + ℒˡ₀)
    Wˡ = - a * constants.liquid.heat_capacity * T
    Wⁱ = - a * (constants.ice.heat_capacity * T + ℒˡ₀ - ℒⁱ₀)
    return Wᵛ, Wˡ, Wⁱ
end

#####
##### Negative moisture correction
#####
#
# Borrowing between water species is a phase change that θˡⁱ does not notice but θ must pay for.

struct DryMoistureCorrectionHeating{L, B, R, D, T, Θ, Q, M, F, MP, C}
    flavor :: L
    potential_temperature_density :: B
    total_density :: R
    dry_density :: D
    temperature :: T
    potential_temperature :: Θ
    specific_moisture :: Q
    microphysical_fields :: M
    liquid_fractions :: F
    microphysics :: MP
    constants :: C
end

Adapt.adapt_structure(to, h::DryMoistureCorrectionHeating) =
    DryMoistureCorrectionHeating(h.flavor, adapt(to, h.potential_temperature_density), adapt(to, h.total_density),
                                 adapt(to, h.dry_density), adapt(to, h.temperature), adapt(to, h.potential_temperature),
                                 adapt(to, h.specific_moisture), adapt(to, h.microphysical_fields), h.liquid_fractions,
                                 adapt(to, h.microphysics), adapt(to, h.constants))

AtmosphereModels.moisture_correction_heating(formulation::DryPotentialTemperatureFormulation, model) =
    DryMoistureCorrectionHeating(formulation.flavor, formulation.potential_temperature_density,
                                 total_density(model.dynamics), dynamics_density(model.dynamics), model.temperature,
                                 formulation.potential_temperature, specific_prognostic_moisture(model),
                                 model.microphysical_fields, AtmosphereModels.correction_liquid_fractions(model.microphysics),
                                 model.microphysics, model.thermodynamic_constants)

@inline correction_phase_heat(::Tuple{}, ::Tuple{}, ::Tuple{}, Wˡ, Wⁱ) = 0

@inline function correction_phase_heat(before::Tuple, after::Tuple, fractions::Tuple, Wˡ, Wⁱ)
    f = first(fractions)
    heat = (f * Wˡ + (1 - f) * Wⁱ) * (first(after) - first(before))
    return heat + correction_phase_heat(Base.tail(before), Base.tail(after), Base.tail(fractions), Wˡ, Wⁱ)
end

@inline function AtmosphereModels.apply_moisture_correction_heating!(h::DryMoistureCorrectionHeating, i, j, k, before,
                                                                     moisture_fields, ρqᵛᵉ)
    after = AtmosphereModels.correction_water_densities(i, j, k, moisture_fields, ρqᵛᵉ)
    @inbounds begin
        ρ = h.total_density[i, j, k]
        ρᵈ = h.dry_density[i, j, k]
        T = h.temperature[i, j, k]
        θ = h.potential_temperature[i, j, k]
        qᵛ = h.specific_moisture[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, nothing, h.microphysics, ρ, qᵛ, h.microphysical_fields)
    Wᵛ, Wˡ, Wⁱ = phase_change_weights(h.flavor, T, q, ρ, ρᵈ / ρ * θ, nothing, nothing, h.constants)
    ΔΘ = Wᵛ * (first(after) - first(before)) +
         correction_phase_heat(Base.tail(before), Base.tail(after), h.liquid_fractions, Wˡ, Wⁱ)
    # A state with no temperature yet (the first `update_state!` of `set!`) has no response.
    @inbounds h.potential_temperature_density[i, j, k] += ifelse(T > 0, ΔΘ, zero(ΔΘ))
    return nothing
end

#####
##### Sedimentation
#####
#
# θ depends on the composition only through the pressure, and condensate leaving a cell at fixed T
# does not change p = (ρᵈ Rᵈ + ρqᵛ Rᵛ) T, so the content per unit falling mass is zero. Falling
# condensate can carry its enthalpy (CM1's `efall = 1`): the transported hˣ is what the
# other formulations use, with the response of the heating path. A flavor with
# `condensate_enthalpy_flux = false` transports nothing (`efall = 0`).

@inline transported_condensate_enthalpies(::Val{false}, T, constants) = (zero(T), zero(T))

@inline function transported_condensate_enthalpies(::Val{true}, T, constants)
    hˡ = constants.liquid.heat_capacity * T - absolute_zero_latent_heat(constants, constants.liquid)
    hⁱ = constants.ice.heat_capacity * T - absolute_zero_latent_heat(constants, constants.ice)
    return (hˡ, hⁱ)
end

@inline function dry_condensate_content(flavor, θ, T, q, constants)
    h = transported_condensate_enthalpies(flavor.condensate_enthalpy_flux, T, constants)
    return (; χ = (zero(T), zero(T)), h, ∂φ∂h = heating_response(flavor, θ, T, q, DryExponent(), constants))
end

@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::DryPotentialTemperatureFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, temperature_field)
    @inbounds ρ = total_density(dynamics)[i, j, k]
    @inbounds qᵛᵉ = specific_prognostic_moisture[i, j, k]
    @inbounds θ = formulation.potential_temperature[i, j, k]
    @inbounds T = temperature_field[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    return dry_condensate_content(formulation.flavor, θ, T, q, constants)
end

@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::DryPotentialTemperatureFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, ::Nothing)
    𝒰 = grid_thermodynamic_state(i, j, k, grid, formulation, dynamics,
                                 microphysics, microphysical_fields, specific_prognostic_moisture)
    T = temperature(𝒰, constants)
    return dry_condensate_content(formulation.flavor, 𝒰.potential_temperature, T, 𝒰.moisture_mass_fractions, constants)
end

# As for θˡⁱ, `implicit_sedimentation_step!` rediagnoses T from the post-solve state.
AtmosphereModels.content_temperature(::DryPotentialTemperatureFormulation, temperature_field) = nothing

#####
##### Setting from temperature
#####

@inline potential_temperature_from_state(::Dry, T, q, pˢᵗ, p, κ̂, constants) = T * (pˢᵗ / p)^dry_kappa(constants)

#####
##### Wall fluxes
#####

AtmosphereModels.wall_potential_temperature_counts_condensate(::DryPotentialTemperatureFormulation) = false
AtmosphereModels.wall_potential_temperature_counts_condensate(::Val{:DryPotentialTemperature}) = false
AtmosphereModels.liquid_ice_exner_exponent(::Val{:DryPotentialTemperature}) = DryExponent()

#####
##### The isochoric closure of θˡⁱ
#####
#
# At fixed composition θˡⁱ changes only through its expansion source, Dθˡⁱ/Dt = ℰ D ln p/Dt with
# ℰ = (θˡⁱ / H)(Rᵐ T − κ H), and adiabatic compression gives D ln p/Dt = −(cᵖᵐ / cᵛᵐ) ∇⋅𝐮, so
#
#   Dθˡⁱ/Dt = − S ∇⋅𝐮 ,   S = ℰ cᵖᵐ / cᵛᵐ ,
#
# which reduces to the dry S = θ (Rᵐ − κᵈ cᵖᵐ) / cᵛᵐ for the dry exponent in unsaturated air (where H = cᵖᵐ T,
# θ = T / Π). Heating at fixed volume adds the expansion it drives: the response is ∂θˡⁱ/∂h + ℰ / (cᵛᵐ T),
# and phase change at fixed volume is the fixed-volume phase-change source (ℰ times the phase-change
# part of D ln p / Dt, which is otherwise zero for θˡⁱ).

@inline function liquid_ice_divergence_coefficient(θ, T, q, exponent, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return expansion_work_source(θ, T, q, exponent, constants) * cᵖᵐ / (cᵖᵐ - Rᵐ)
end

"""
$(TYPEDSIGNATURES)

Return the coefficient ``S`` of the velocity divergence, ``Dθ/Dt ∋ - S ∇ ⋅ 𝐮``, for the isochoric closure of `flavor`.
"""
@inline post_loop_divergence_coefficient(::Dry, θ, T, q, exponent, constants) = divergence_expansion_coefficient(θ, q, constants)
@inline post_loop_divergence_coefficient(::LiquidIce, θ, T, q, exponent, constants) =
    liquid_ice_divergence_coefficient(θ, T, q, exponent, constants)

@inline heating_response(flavor::LiquidIce{<:IsochoricPressureWork}, θ, T, q, exponent, constants) =
    potential_temperature_heating_response(θ, liquid_ice_enthalpy(T, q, constants)) +
    isochoric_heating_correction(θ, T, q, exponent, constants)

@inline function isochoric_heating_correction(θ, T, q, exponent, constants)
    T̂ = ifelse(T == 0, one(T), T)
    cᵛᵐ = mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)
    return expansion_work_source(θ, T, q, exponent, constants) / (cᵛᵐ * T̂)
end

@inline expansion_tendency(i, j, k, grid, flavor::LiquidIce{<:IsochoricPressureWork}, dynamics, ρᵈ, θ, T, q, exponent, constants, velocities, model_fields) =
    - ρᵈ * liquid_ice_divergence_coefficient(θ, T, q, exponent, constants) * explicit_velocity_divergence(i, j, k, grid, dynamics, velocities)
