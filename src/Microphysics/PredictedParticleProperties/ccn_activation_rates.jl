#####
##### CCN activation rates
#####
##### Cloud-droplet activation for the prescribed-Nᶜ and prognostic-CCN paths.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature,
                             adjustment_saturation_specific_humidity,
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             density,
                             liquid_latent_heat,
                             ice_latent_heat,
                             vapor_gas_constant,
                             MoistureMassFractions,
                             ThermodynamicConstants
using DocStringExtensions: TYPEDSIGNATURES

#####
##### CCN activation
#####

"""
$(TYPEDSIGNATURES)

Compute CCN activation rate for the 1-moment (prescribed Nᶜ) case.

Following Fortran P3 v5.5.0 (lines 3953-3963): when the air is supersaturated
and the cloud mass is below the minimum threshold for the prescribed droplet
concentration, a seed mass is created. The target cloud mass is
``N_c / ρ × m_{\\text{drop}}`` where ``m_{\\text{drop}} = (4π/3) ρ_w r^3``
for ``r = 1`` μm. The rate is limited by the available supersaturation.

The supersaturation limit divides by the same liquid psychrometric factor
``ξˡ = 1 + ℒˡ² q^{v+ℓ} / (c_p^d R_v T²)`` that `limit_vapor_rates` uses to build
`qcon_cap` and that the Grabowski-Morrison alignment uses in
`predicted_supersaturation_adjustment`. Sizing the rate with the moist mixture
heat capacity and then capping it with the dry-air one would mix two conventions inside
one cell's vapor budget; Fortran's `ab` is the dry-air form, applied once.

# Returns
- Rate of vapor → cloud liquid conversion from CCN activation [kg/kg/s]
"""
@inline function ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    # Mass of a newly formed cloud droplet (Fortran cons7)
    cons7 = activated_droplet_mass(prp, FT)

    # Target cloud mass for prescribed droplet concentration
    target_qc = Nᶜ / ρ * cons7

    # Deficit: how much mass is needed to reach the minimum
    deficit = clamp_positive(target_qc - clamp_positive(qᶜˡ))

    # Psychrometric correction (liquid saturation, Fortran `ab`)
    ℒˡ = vaporization_latent_heat(constants, T)
    Rᵛ = FT(vapor_gas_constant(constants))
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

    # Limit by available supersaturation (Fortran: min(tmp1, (Qv_cld-dumqvs)/ab))
    max_from_ss = clamp_positive((qᵛ - qᵛ⁺ˡ) / ξˡ)
    rate = min(deficit, max_from_ss) / prp.sink_limiting_timescale

    # Only activate when supersaturated (Fortran threshold: sup_cld > 1e-6)
    floors = prp.floors
    S = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, floors.saturation_mass_fraction)
    is_supersaturated = S > prp.activation_supersaturation_threshold
    return ifelse(is_supersaturated, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Dispatch CCN activation: prescribed (Nothing) or prognostic (AerosolActivation).
Returns `(; mass, number)` named tuple.
"""
@inline function compute_ccn_activation(::Nothing, p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    FT = typeof(qᶜˡ)
    # Prescribed-Nᶜ path (Fortran `log_predictNc = .false.`, `nc = nccnst_2`):
    # the activation target is the scheme parameter, not the DSD-diagnosed `Nᶜ`.
    # When `qᶜˡ` is below the mass threshold, `diagnose_cloud_dsd` clamps the
    # returned `Nᶜ` toward zero — using that value would collapse `target_qc`
    # and block any seed mass from forming in a warm-bubble parcel.
    target_Nᶜ = p3.cloud.number_concentration
    mass = ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, target_Nᶜ, constants)
    return (; mass, number = zero(FT))
end

@inline function compute_ccn_activation(aerosol::AerosolActivation, p3, qᶜˡ, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, nᵃ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end
