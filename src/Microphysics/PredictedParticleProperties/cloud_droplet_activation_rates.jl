#####
##### Cloud-droplet activation rates
#####
##### Prescribed droplet number uses a seed-mass source. Aerosol activation
##### predicts droplet number, with an optional prognostic aerosol reservoir.
#####

#####
##### Activation with prescribed droplet number
#####

"""
$(TYPEDSIGNATURES)

Return the vapor-to-cloud mass rate [kg/kg/s] for prescribed droplet concentration `Nᶜˡ`.

In supersaturated air, supply seed mass up to
``N^{cl} / ρ × m_{\\text{drop}}`` where ``m_{\\text{drop}} = (4π/3) ρ_w r^3``
and the default seed radius is 1 μm. The supersaturation cap uses
``ξˡ = 1 + ℒˡ² q^{v+ℓ} / (c_p^d R_v T²)`` with the dry-air heat capacity,
consistent with `limit_vapor_rates` and `predicted_supersaturation_adjustment`.
"""
@inline function prescribed_cloud_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, ρ, Nᶜˡ, constants)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates

    # Mass of a newly formed cloud droplet
    droplet_mass = activated_droplet_mass(parameters, FT)

    # Target cloud mass for prescribed droplet concentration
    qᶜˡ_target = Nᶜˡ / ρ * droplet_mass

    # Deficit: how much mass is needed to reach the minimum
    deficit = max(0, qᶜˡ_target - max(0, qᶜˡ))

    # Psychrometric correction (liquid saturation)
    ℒˡ = vaporization_latent_heat(constants, T)
    Rᵛ = FT(vapor_gas_constant(constants))
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

    # Limit by the available supersaturation
    max_from_ss = max(0, (qᵛ - qᵛ⁺ˡ) / ξˡ)
    rate = min(deficit, max_from_ss) / parameters.sink_limiting_timescale

    # Only activate when supersaturated
    floors = parameters.floors
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, floors.saturation_mass_fraction)
    ℂᶠᵒʳᵐ₃ = parameters.activation_supersaturation_threshold
    is_supersaturated = S > ℂᶠᵒʳᵐ₃
    return ifelse(is_supersaturated, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Return cloud droplet activation rates `(; mass, number)` in [kg/kg/s] and [kg⁻¹ s⁻¹].

With `nothing`, droplet number is prescribed and `number` is zero.
With [`AerosolActivation`](@ref), both rates follow the aerosol distribution;
`prognostic=true` also limits activation to the remaining reservoir.
"""
@inline function compute_cloud_droplet_activation(::Nothing, p3, qᶜˡ, nᶜˡ, nᵃ,
                                                  qᵛ, qᵛ⁺ˡ, T, ρ, constants)
    FT = typeof(qᶜˡ)
    # `prescribed_cloud_activation_rate` takes Nᶜˡ per volume and divides by ρ itself,
    # so the target is the scheme parameter directly. The per-mass `nᶜˡ` is unused.
    Nᶜˡ_target = p3.cloud.number_concentration
    mass = prescribed_cloud_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, ρ,
                                            Nᶜˡ_target, constants)
    return (; mass, number = zero(FT))
end

# Fixed populations use the full distribution; the unused nᵃ argument is zero.
@inline function compute_cloud_droplet_activation(aerosol::AerosolActivation{<:Any, false},
                                                  p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, ρ, constants)
    result = aerosol_activation_rate(aerosol, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end

# Prognostic populations are limited by the remaining nᵃ.
@inline function compute_cloud_droplet_activation(aerosol::AerosolActivation{<:Any, true},
                                                  p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, ρ, constants)
    result = aerosol_activation_rate(aerosol, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end
