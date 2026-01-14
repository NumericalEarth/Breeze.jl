#####
##### Parcel microphysics stepping for CloudMicrophysics schemes
#####
#
# This file implements step_microphysics for the CloudMicrophysics one-moment
# and two-moment schemes, enabling parcel model simulations.
#####

using Breeze.ParcelDynamics: ParcelDynamics, step_microphysics

#####
##### One-moment warm-phase microphysics stepping
#####

"""
Step the warm-phase one-moment microphysical state forward by Δt.

Uses Forward Euler to integrate the tendencies for cloud liquid and rain.
"""
function ParcelDynamics.step_microphysics(
    microphysics::WarmPhase1M,
    ℳ::WarmPhaseOneMomentState{FT},
    ρ,
    𝒰,
    constants,
    Δt
) where FT

    # Compute tendencies (per unit mass, so divide by ρ)
    dρqᶜˡ_dt = microphysical_tendency(microphysics, Val(:ρqᶜˡ), ρ, ℳ, 𝒰, constants)
    dρqʳ_dt = microphysical_tendency(microphysics, Val(:ρqʳ), ρ, ℳ, 𝒰, constants)

    # Convert to mixing ratio tendencies
    dqᶜˡ_dt = dρqᶜˡ_dt / ρ
    dqʳ_dt = dρqʳ_dt / ρ

    # Forward Euler update
    qᶜˡ_new = max(0, ℳ.qᶜˡ + dqᶜˡ_dt * Δt)
    qʳ_new = max(0, ℳ.qʳ + dqʳ_dt * Δt)

    return WarmPhaseOneMomentState{FT}(qᶜˡ_new, qʳ_new)
end

# For saturation adjustment schemes (WP1M), cloud liquid is diagnosed
function ParcelDynamics.step_microphysics(
    microphysics::WP1M,
    ℳ::WarmPhaseOneMomentState{FT},
    ρ,
    𝒰,
    constants,
    Δt
) where FT

    # Rain evolves via tendencies
    dρqʳ_dt = microphysical_tendency(microphysics, Val(:ρqʳ), ρ, ℳ, 𝒰, constants)
    dqʳ_dt = dρqʳ_dt / ρ
    qʳ_new = max(0, ℳ.qʳ + dqʳ_dt * Δt)

    # Cloud liquid is diagnosed from saturation adjustment
    # (will be computed when thermodynamic state is adjusted)
    qᶜˡ_new = ℳ.qᶜˡ  # Keep current value; saturation adjustment handles this

    return WarmPhaseOneMomentState{FT}(qᶜˡ_new, qʳ_new)
end

#####
##### One-moment mixed-phase microphysics stepping
#####

function ParcelDynamics.step_microphysics(
    microphysics::MPNE1M,
    ℳ::MixedPhaseOneMomentState{FT},
    ρ,
    𝒰,
    constants,
    Δt
) where FT

    # Compute tendencies
    dρqᶜˡ_dt = microphysical_tendency(microphysics, Val(:ρqᶜˡ), ρ, ℳ, 𝒰, constants)
    dρqᶜⁱ_dt = microphysical_tendency(microphysics, Val(:ρqᶜⁱ), ρ, ℳ, 𝒰, constants)
    dρqʳ_dt = microphysical_tendency(microphysics, Val(:ρqʳ), ρ, ℳ, 𝒰, constants)
    # TODO: Add snow tendency when implemented

    # Convert to mixing ratio tendencies
    dqᶜˡ_dt = dρqᶜˡ_dt / ρ
    dqᶜⁱ_dt = dρqᶜⁱ_dt / ρ
    dqʳ_dt = dρqʳ_dt / ρ

    # Forward Euler update
    qᶜˡ_new = max(0, ℳ.qᶜˡ + dqᶜˡ_dt * Δt)
    qᶜⁱ_new = max(0, ℳ.qᶜⁱ + dqᶜⁱ_dt * Δt)
    qʳ_new = max(0, ℳ.qʳ + dqʳ_dt * Δt)
    qˢ_new = ℳ.qˢ  # Snow not yet implemented

    return MixedPhaseOneMomentState{FT}(qᶜˡ_new, qᶜⁱ_new, qʳ_new, qˢ_new)
end

#####
##### Two-moment warm-phase microphysics stepping
#####

function ParcelDynamics.step_microphysics(
    microphysics::WPNE2M,
    ℳ::WarmPhaseTwoMomentState{FT},
    ρ,
    𝒰,
    constants,
    Δt
) where FT

    # Compute tendencies for all four prognostic variables
    dρqᶜˡ_dt = microphysical_tendency(microphysics, Val(:ρqᶜˡ), ρ, ℳ, 𝒰, constants)
    dρnᶜˡ_dt = microphysical_tendency(microphysics, Val(:ρnᶜˡ), ρ, ℳ, 𝒰, constants)
    dρqʳ_dt = microphysical_tendency(microphysics, Val(:ρqʳ), ρ, ℳ, 𝒰, constants)
    dρnʳ_dt = microphysical_tendency(microphysics, Val(:ρnʳ), ρ, ℳ, 𝒰, constants)

    # Convert to per-mass tendencies
    dqᶜˡ_dt = dρqᶜˡ_dt / ρ
    dnᶜˡ_dt = dρnᶜˡ_dt / ρ
    dqʳ_dt = dρqʳ_dt / ρ
    dnʳ_dt = dρnʳ_dt / ρ

    # Forward Euler update (with positivity constraints)
    qᶜˡ_new = max(0, ℳ.qᶜˡ + dqᶜˡ_dt * Δt)
    nᶜˡ_new = max(0, ℳ.nᶜˡ + dnᶜˡ_dt * Δt)
    qʳ_new = max(0, ℳ.qʳ + dqʳ_dt * Δt)
    nʳ_new = max(0, ℳ.nʳ + dnʳ_dt * Δt)

    return WarmPhaseTwoMomentState{FT}(qᶜˡ_new, nᶜˡ_new, qʳ_new, nʳ_new)
end

#####
##### Compute moisture fractions from microphysical states
#####

# Warm-phase one-moment: qˡ = qᶜˡ + qʳ
@inline function ParcelDynamics.compute_moisture_fractions(
    ℳ::WarmPhaseOneMomentState,
    qᵗ
)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Mixed-phase one-moment: qˡ = qᶜˡ + qʳ, qⁱ = qᶜⁱ + qˢ
@inline function ParcelDynamics.compute_moisture_fractions(
    ℳ::MixedPhaseOneMomentState,
    qᵗ
)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qⁱ = ℳ.qᶜⁱ + ℳ.qˢ
    qᵛ = qᵗ - qˡ - qⁱ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

# Two-moment warm-phase: same as one-moment
@inline function ParcelDynamics.compute_moisture_fractions(
    ℳ::WarmPhaseTwoMomentState,
    qᵗ
)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end
