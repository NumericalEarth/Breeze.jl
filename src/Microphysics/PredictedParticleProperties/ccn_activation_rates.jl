#####
##### CCN activation rates
#####
##### Cloud-droplet activation for the prescribed-Nᶜˡ and prognostic-CCN paths.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

#####
##### CCN activation
#####

"""
$(TYPEDSIGNATURES)

Compute CCN activation rate for the 1-moment (prescribed Nᶜˡ) case.

When the air is supersaturated
and the cloud mass is below the minimum threshold for the prescribed droplet
concentration, a seed mass is created. The target cloud mass is
``N^{cl} / ρ × m_{\\text{drop}}`` where ``m_{\\text{drop}} = (4π/3) ρ_w r^3``
for ``r = 1`` μm. The rate is limited by the available supersaturation.

The supersaturation limit divides by the same liquid psychrometric factor
``ξˡ = 1 + ℒˡ² q^{v+ℓ} / (c_p^d R_v T²)`` that `limit_vapor_rates` uses to build
`qcon_cap` and that the Grabowski-Morrison alignment uses in
`predicted_supersaturation_adjustment`. Sizing the rate with the moist mixture
heat capacity and then capping it with the dry-air one would mix two conventions inside
one cell's vapor budget, so the dry-air form is used throughout.

# Returns
- Rate of vapor → cloud liquid conversion from CCN activation [kg/kg/s]
"""
@inline function ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜˡ, constants)
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
    is_supersaturated = S > parameters.activation_supersaturation_threshold
    return ifelse(is_supersaturated, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Dispatch CCN activation: prescribed (Nothing) or prognostic (AerosolActivation).
Returns `(; mass, number)` named tuple.
"""
@inline function compute_ccn_activation(::Nothing, p3, qᶜˡ, nᶜˡ, nᵃ,
                                        qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜˡ, constants)
    FT = typeof(qᶜˡ)
    # Prescribed-Nᶜˡ path: the activation target is the scheme parameter, not the
    # DSD-diagnosed `Nᶜˡ`.
    # When `qᶜˡ` is below the mass threshold, `diagnose_cloud_dsd` clamps the
    # returned `Nᶜˡ` toward zero — using that value would collapse `qᶜˡ_target`
    # and block any seed mass from forming in a warm-bubble parcel.
    Nᶜˡ_target = p3.cloud.number_concentration
    mass = ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ,
                               Nᶜˡ_target, constants)
    return (; mass, number = zero(FT))
end

@inline function compute_ccn_activation(aerosol::AerosolActivation, p3, qᶜˡ, nᶜˡ, nᵃ,
                                        qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜˡ, constants)
    result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end
