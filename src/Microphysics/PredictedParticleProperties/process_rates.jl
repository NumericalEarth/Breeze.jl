#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### Phase 1: Rain processes, ice deposition/sublimation, melting.
##### Phase 2: Aggregation, riming, shedding, refreezing, ice nucleation.
##### Phase 3: Terminal velocities (sedimentation).
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature

#####
##### Physical constants (to be replaced with thermodynamic constants interface)
#####

const ρʷ = 1000.0   # Liquid water density [kg/m³]
const ρⁱ = 917.0    # Pure ice density [kg/m³]
const Dᵛ_ref = 2.21e-5  # Reference water vapor diffusivity [m²/s]
const Kᵗʰ_ref = 0.024   # Reference thermal conductivity [W/(m·K)]
const mᵢ₀ = 1e-12   # Mass of nucleated ice crystal [kg] (10 μm diameter sphere at 917 kg/m³)
const T_freeze = 273.15  # Freezing temperature [K]

#####
##### Utility functions
#####

"""
    clamp_positive(x)

Return max(0, x) for numerical stability.
"""
@inline clamp_positive(x) = max(0, x)

"""
    safe_divide(a, b, default=zero(a))

Safe division returning `default` when b ≈ 0.
"""
@inline function safe_divide(a, b, default=zero(a))
    FT = typeof(a)
    ε = eps(FT)
    return ifelse(abs(b) < ε, default, a / b)
end

#####
##### Rain processes
#####

"""
    rain_autoconversion_rate(qᶜˡ, ρ, Nc; k₁=2.47e-2, q_threshold=1e-4)

Compute rain autoconversion rate following Khairoutdinov and Kogan (2000).

Cloud droplets larger than a threshold undergo collision-coalescence to form rain.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `ρ`: Air density [kg/m³]
- `Nc`: Cloud droplet number concentration [1/m³]
- `k₁`: Autoconversion rate coefficient [s⁻¹], default 2.47e-2
- `q_threshold`: Minimum cloud water for autoconversion [kg/kg], default 1e-4

# Returns
- Rate of cloud → rain conversion [kg/kg/s]

# Reference
Khairoutdinov, M. and Kogan, Y. (2000). A new cloud physics parameterization
in a large-eddy simulation model of marine stratocumulus. Mon. Wea. Rev.
"""
@inline function rain_autoconversion_rate(qᶜˡ, ρ, Nc;
                                           k₁ = 2.47e-2,
                                           q_threshold = 1e-4)
    FT = typeof(qᶜˡ)

    # No autoconversion below threshold
    qᶜˡ_eff = clamp_positive(qᶜˡ - q_threshold)

    # Khairoutdinov-Kogan (2000) autoconversion: ∂qʳ/∂t = k₁ * qᶜˡ^α * Nc^β
    # With α ≈ 2.47, β ≈ -1.79, simplified here to:
    # ∂qʳ/∂t = k₁ * qᶜˡ^2.47 * (Nc/1e8)^(-1.79)
    Nc_scaled = Nc / FT(1e8)  # Reference concentration 100/cm³

    # Avoid division by zero
    Nc_scaled = max(Nc_scaled, FT(0.01))

    α = FT(2.47)
    β = FT(-1.79)

    return k₁ * qᶜˡ_eff^α * Nc_scaled^β
end

"""
    rain_accretion_rate(qᶜˡ, qʳ, ρ; k₂=67.0)

Compute rain accretion rate following Khairoutdinov and Kogan (2000).

Falling rain drops collect cloud droplets via gravitational sweep-out.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qʳ`: Rain mass fraction [kg/kg]
- `ρ`: Air density [kg/m³]
- `k₂`: Accretion rate coefficient [s⁻¹], default 67.0

# Returns
- Rate of cloud → rain conversion [kg/kg/s]

# Reference
Khairoutdinov, M. and Kogan, Y. (2000). Mon. Wea. Rev.
"""
@inline function rain_accretion_rate(qᶜˡ, qʳ, ρ;
                                      k₂ = 67.0)
    FT = typeof(qᶜˡ)

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qʳ_eff = clamp_positive(qʳ)

    # KK2000: ∂qʳ/∂t = k₂ * (qᶜˡ * qʳ)^1.15
    α = FT(1.15)

    return k₂ * (qᶜˡ_eff * qʳ_eff)^α
end

"""
    rain_self_collection_rate(qʳ, nʳ, ρ)

Compute rain self-collection rate (number tendency only).

Large rain drops collect smaller ones, reducing number but conserving mass.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Rate of rain number reduction [1/kg/s]
"""
@inline function rain_self_collection_rate(qʳ, nʳ, ρ)
    FT = typeof(qʳ)

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)

    # Seifert & Beheng (2001) self-collection
    k_rr = FT(4.33)  # Collection kernel coefficient

    # ∂nʳ/∂t = -k_rr * ρ * qʳ * nʳ
    return -k_rr * ρ * qʳ_eff * nʳ_eff
end

"""
    rain_evaporation_rate(qʳ, qᵛ, qᵛ⁺, T, ρ, nʳ; τ_evap=10.0)

Compute rain evaporation rate for subsaturated conditions.

Rain drops evaporate when the ambient air is subsaturated (qᵛ < qᵛ⁺).

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺`: Saturation vapor mass fraction [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `nʳ`: Rain number concentration [1/kg]
- `τ_evap`: Evaporation timescale [s], default 10

# Returns
- Rate of rain → vapor conversion [kg/kg/s] (negative = evaporation)
"""
@inline function rain_evaporation_rate(qʳ, qᵛ, qᵛ⁺, T, ρ, nʳ;
                                        τ_evap = 10.0)
    FT = typeof(qʳ)

    qʳ_eff = clamp_positive(qʳ)

    # Subsaturation
    S = qᵛ - qᵛ⁺

    # Only evaporate in subsaturated conditions
    S_sub = min(S, zero(FT))

    # Simplified relaxation: ∂qʳ/∂t = S / τ
    # Limited by available rain
    evap_rate = S_sub / τ_evap

    # Cannot evaporate more than available
    max_evap = -qʳ_eff / τ_evap

    return max(evap_rate, max_evap)
end

#####
##### Ice deposition and sublimation
#####

"""
    ice_deposition_rate(qⁱ, qᵛ, qᵛ⁺ⁱ, T, ρ, nⁱ; τ_dep=10.0)

Compute ice deposition/sublimation rate.

Ice grows by vapor deposition when supersaturated with respect to ice,
and sublimates when subsaturated.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `nⁱ`: Ice number concentration [1/kg]
- `τ_dep`: Deposition/sublimation timescale [s], default 10

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
@inline function ice_deposition_rate(qⁱ, qᵛ, qᵛ⁺ⁱ, T, ρ, nⁱ;
                                      τ_dep = 10.0)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)

    # Supersaturation with respect to ice
    Sⁱ = qᵛ - qᵛ⁺ⁱ

    # Relaxation toward saturation
    dep_rate = Sⁱ / τ_dep

    # Limit sublimation to available ice
    is_sublimation = Sⁱ < 0
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

"""
    ventilation_enhanced_deposition(qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, T, ρ, Fᶠ, ρᶠ;
                                     Dᵛ=Dᵛ_ref, Kᵗʰ=Kᵗʰ_ref)

Compute ventilation-enhanced ice deposition/sublimation rate.

Large falling ice particles enhance vapor diffusion through ventilation.
This uses the full capacitance formulation with ventilation factors.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `Dᵛ`: Vapor diffusivity [m²/s]
- `Kᵗʰ`: Thermal conductivity [W/(m·K)]

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)

# Notes
This is a simplified version. The full P3 implementation uses quadrature
integrals over the size distribution with regime-dependent ventilation.
"""
@inline function ventilation_enhanced_deposition(qⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, T, ρ, Fᶠ, ρᶠ;
                                                  Dᵛ = Dᵛ_ref,
                                                  Kᵗʰ = Kᵗʰ_ref,
                                                  ℒⁱ = 2.834e6)  # Latent heat [J/kg]
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Mean mass and diameter (simplified)
    m_mean = safe_divide(qⁱ_eff, nⁱ_eff, FT(1e-12))

    # Estimate mean diameter from mass assuming ρ_eff
    ρ_eff = (1 - Fᶠ) * FT(ρⁱ) * FT(0.1) + Fᶠ * ρᶠ  # Effective density
    D_mean = cbrt(6 * m_mean / (FT(π) * ρ_eff))

    # Capacitance (sphere for small, 0.48*D for large)
    D_threshold = FT(100e-6)
    C = ifelse(D_mean < D_threshold, D_mean / 2, FT(0.48) * D_mean)

    # Supersaturation with respect to ice
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, FT(1e-10))

    # Vapor diffusion coefficient (simplified)
    G = 4 * FT(π) * C * Dᵛ * ρ

    # Ventilation factor (simplified average)
    fᵛ = FT(1.0) + FT(0.5) * sqrt(D_mean / FT(100e-6))

    # Deposition rate per particle
    dm_dt = G * fᵛ * Sⁱ * qᵛ⁺ⁱ

    # Total rate
    dep_rate = nⁱ_eff * dm_dt

    # Limit sublimation
    is_sublimation = Sⁱ < 0
    τ_sub = FT(10.0)
    max_sublim = -qⁱ_eff / τ_sub

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

#####
##### Melting
#####

"""
    ice_melting_rate(qⁱ, nⁱ, T, ρ, T_freeze; τ_melt=60.0)

Compute ice melting rate when temperature exceeds freezing.

Ice particles melt to rain when the ambient temperature is above freezing.
The melting rate depends on the temperature excess and particle surface area.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `T_freeze`: Freezing temperature [K], default 273.15
- `τ_melt`: Melting timescale at ΔT=1K [s], default 60

# Returns
- Rate of ice → rain conversion [kg/kg/s]
"""
@inline function ice_melting_rate(qⁱ, nⁱ, T, ρ;
                                   T_freeze = 273.15,
                                   τ_melt = 60.0)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)

    # Temperature excess above freezing
    ΔT = T - FT(T_freeze)
    ΔT_pos = clamp_positive(ΔT)

    # Melting rate proportional to temperature excess
    # Faster melting for larger ΔT
    rate_factor = ΔT_pos / FT(1.0)  # Normalize to 1K

    # Melt rate
    melt_rate = qⁱ_eff * rate_factor / τ_melt

    return melt_rate
end

"""
    ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)

Compute ice number tendency from melting.

Number of melted particles equals number of rain drops produced.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qⁱ_melt_rate`: Ice mass melting rate [kg/kg/s]

# Returns
- Rate of ice number reduction [1/kg/s]
"""
@inline function ice_melting_number_rate(qⁱ, nⁱ, qⁱ_melt_rate)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Number rate proportional to mass rate
    # ∂nⁱ/∂t = (nⁱ/qⁱ) * ∂qⁱ_melt/∂t
    ratio = safe_divide(nⁱ_eff, qⁱ_eff, zero(FT))

    return -ratio * qⁱ_melt_rate
end

#####
##### Ice nucleation (deposition and immersion freezing)
#####

"""
    deposition_nucleation_rate(T, qᵛ, qᵛ⁺ⁱ, nⁱ_current, ρ;
                               T_threshold=258.15, Sⁱ_threshold=0.05)

Compute ice nucleation rate from deposition/condensation freezing.

New ice crystals nucleate when temperature is below -15°C and the air
is supersaturated with respect to ice. Uses Cooper (1986) parameterization.

# Arguments
- `T`: Temperature [K]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `nⁱ_current`: Current ice number concentration [1/kg]
- `ρ`: Air density [kg/m³]
- `T_threshold`: Maximum temperature for nucleation [K] (default -15°C = 258.15 K)
- `Sⁱ_threshold`: Ice supersaturation threshold for nucleation (default 5%)

# Returns
- Tuple (Q_nuc, N_nuc): mass rate [kg/kg/s] and number rate [1/kg/s]

# Reference
Cooper, W. A. (1986). Ice initiation in natural clouds. Precipitation 
Enhancement—A Scientific Challenge. AMS Meteor. Monogr.
"""
@inline function deposition_nucleation_rate(T, qᵛ, qᵛ⁺ⁱ, nⁱ_current, ρ;
                                             T_threshold = 258.15,
                                             Sⁱ_threshold = 0.05,
                                             N_max = 100e3,
                                             τ_nuc = 60.0)  # Nucleation relaxation timescale [s]
    FT = typeof(T)
    
    # Ice supersaturation
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, FT(1e-10))
    
    # Conditions for nucleation
    nucleation_active = (T < FT(T_threshold)) && (Sⁱ > FT(Sⁱ_threshold))
    
    # Cooper (1986): N_ice = 0.005 × exp(0.304 × (T₀ - T))
    # where T₀ = 273.15 K
    ΔT = FT(T_freeze) - T
    N_cooper = FT(0.005) * exp(FT(0.304) * ΔT) * FT(1000) / ρ  # Convert L⁻¹ to kg⁻¹
    
    # Limit to maximum and subtract existing ice
    N_equilibrium = min(N_cooper, FT(N_max) / ρ)
    
    # Nucleation rate: relaxation toward equilibrium with timescale τ_nuc
    N_nuc = clamp_positive(N_equilibrium - nⁱ_current) / FT(τ_nuc)
    
    # Mass nucleation rate (each crystal has initial mass mᵢ₀)
    Q_nuc = N_nuc * FT(mᵢ₀)
    
    # Zero out if conditions not met
    N_nuc = ifelse(nucleation_active && N_nuc > FT(1e-20), N_nuc, zero(FT))
    Q_nuc = ifelse(nucleation_active && Q_nuc > FT(1e-30), Q_nuc, zero(FT))
    
    return Q_nuc, N_nuc
end

"""
    immersion_freezing_cloud_rate(qᶜˡ, Nc, T, ρ)

Compute immersion freezing rate of cloud droplets.

Cloud droplets freeze when temperature is below -4°C. Uses Bigg (1953)
stochastic freezing parameterization with Gamma distribution integration.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nc`: Cloud droplet number concentration [1/m³ or 1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]

# Reference
Bigg, E. K. (1953). The formation of atmospheric ice crystals by the
freezing of droplets. Quart. J. Roy. Meteor. Soc.
"""
@inline function immersion_freezing_cloud_rate(qᶜˡ, Nc, T, ρ;
                                                T_max = 269.15,  # -4°C
                                                aimm = 0.66)      # Bigg parameter
    FT = typeof(qᶜˡ)
    
    qᶜˡ_eff = clamp_positive(qᶜˡ)
    
    # Conditions for freezing
    freezing_active = (T < FT(T_max)) && (qᶜˡ_eff > FT(1e-8))
    
    # Bigg (1953) freezing rate coefficient
    # J = exp(aimm × (T₀ - T))
    ΔT = FT(T_freeze) - T
    J = exp(FT(aimm) * ΔT)
    
    # Simplified: fraction frozen per timestep depends on temperature
    # Use characteristic freezing timescale that decreases with T
    τ_frz = FT(1000) / max(J, FT(1))  # Timescale decreases as J increases
    
    # Freezing rate
    N_frz = ifelse(freezing_active, Nc / τ_frz, zero(FT))
    Q_frz = ifelse(freezing_active, qᶜˡ_eff / τ_frz, zero(FT))
    
    return Q_frz, N_frz
end

"""
    immersion_freezing_rain_rate(qʳ, nʳ, T, ρ)

Compute immersion freezing rate of rain drops.

Rain drops freeze when temperature is below -4°C. Uses Bigg (1953)
stochastic freezing parameterization.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_frz, N_frz): mass rate [kg/kg/s] and number rate [1/kg/s]
"""
@inline function immersion_freezing_rain_rate(qʳ, nʳ, T, ρ;
                                               T_max = 269.15,  # -4°C
                                               aimm = 0.66)
    FT = typeof(qʳ)
    
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    
    # Conditions for freezing
    freezing_active = (T < FT(T_max)) && (qʳ_eff > FT(1e-8))
    
    # Bigg (1953) freezing rate coefficient
    ΔT = FT(T_freeze) - T
    J = exp(FT(aimm) * ΔT)
    
    # Rain freezes faster due to larger volume (stochastic freezing ∝ V × J)
    # Characteristic time decreases with drop size and supercooling
    τ_frz = FT(300) / max(J, FT(1))
    
    # Freezing rate
    N_frz = ifelse(freezing_active, nʳ_eff / τ_frz, zero(FT))
    Q_frz = ifelse(freezing_active, qʳ_eff / τ_frz, zero(FT))
    
    return Q_frz, N_frz
end

#####
##### Rime splintering (Hallett-Mossop secondary ice production)
#####

"""
    rime_splintering_rate(qʳ, nⁱ, cloud_riming, rain_riming, T, ρ)

Compute secondary ice production from rime splintering (Hallett-Mossop).

When rimed ice particles accrete supercooled drops, ice splinters are 
ejected. This occurs only in the temperature range -8°C to -3°C.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `cloud_riming`: Cloud droplet riming rate [kg/kg/s]
- `rain_riming`: Rain riming rate [kg/kg/s]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]

# Returns
- Tuple (Q_spl, N_spl): ice mass rate [kg/kg/s] and number rate [1/kg/s]

# Reference
Hallett, J. and Mossop, S. C. (1974). Production of secondary ice
particles during the riming process. Nature.
"""
@inline function rime_splintering_rate(cloud_riming, rain_riming, T, ρ;
                                        T_low = 265.15,   # -8°C
                                        T_high = 270.15,  # -3°C
                                        c_splinter = 3.5e8)  # Splinters per kg of rime
    FT = typeof(T)
    
    # Hallett-Mossop temperature window: -8°C to -3°C
    in_HM_window = (T > FT(T_low)) && (T < FT(T_high))
    
    # Efficiency peaks at -5°C, tapers to zero at boundaries
    T_peak = FT(268.15)  # -5°C
    T_width = FT(2.5)     # Half-width of efficiency curve
    efficiency = exp(-((T - T_peak) / T_width)^2)
    
    # Total riming rate
    total_riming = clamp_positive(cloud_riming + rain_riming)
    
    # Number of splinters produced (Hallett-Mossop rate ~350 per mg of rime)
    # c_splinter = 3.5e8 splinters per kg of rime
    N_spl = ifelse(in_HM_window,
                    efficiency * FT(c_splinter) * total_riming,
                    zero(FT))
    
    # Mass of splinters (each splinter has initial mass mᵢ₀)
    Q_spl = N_spl * FT(mᵢ₀)
    
    return Q_spl, N_spl
end

#####
##### Phase 2: Ice aggregation
#####

"""
    ice_aggregation_rate(qⁱ, nⁱ, T, ρ; Eᵢᵢ_max=1.0, τ_agg=600.0)

Compute ice self-collection (aggregation) rate.

Ice particles collide and stick together, reducing number concentration
without changing total mass. The sticking efficiency increases with temperature.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `Eᵢᵢ_max`: Maximum ice-ice collection efficiency
- `τ_agg`: Aggregation timescale at maximum efficiency [s]

# Returns
- Rate of ice number reduction [1/kg/s]

# Reference
Morrison & Milbrandt (2015). Self-collection computed using lookup table
integrals over the size distribution. Here we use a simplified relaxation form.
"""
@inline function ice_aggregation_rate(qⁱ, nⁱ, T, ρ;
                                       Eᵢᵢ_max = 1.0,
                                       τ_agg = 600.0)
    FT = typeof(qⁱ)
    T_freeze = FT(273.15)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # No aggregation for small ice content
    qⁱ_threshold = FT(1e-8)
    nⁱ_threshold = FT(1e2)  # per kg

    # Temperature-dependent sticking efficiency (P3 uses linear ramp)
    # E_ii = 0.1 at T < 253 K, linear ramp to 1.0 at T > 268 K
    T_low = FT(253.15)
    T_high = FT(268.15)

    Eᵢᵢ = ifelse(T < T_low,
                  FT(0.1),
                  ifelse(T > T_high,
                         Eᵢᵢ_max,
                         FT(0.1) + (T - T_low) * FT(0.9) / (T_high - T_low)))

    # Aggregation rate: collision kernel ∝ n² × collection efficiency
    # Simplified: ∂n/∂t = -E_ii × n² / (τ × n_ref)
    # The rate scales with n² because it's a binary collision process
    n_ref = FT(1e4)  # Reference number concentration [1/kg]

    # Only aggregate above thresholds
    rate = ifelse(qⁱ_eff > qⁱ_threshold && nⁱ_eff > nⁱ_threshold,
                   -Eᵢᵢ * nⁱ_eff^2 / (τ_agg * n_ref),
                   zero(FT))

    return rate
end

#####
##### Phase 2: Riming (cloud and rain collection by ice)
#####

"""
    cloud_riming_rate(qᶜˡ, qⁱ, nⁱ, T, ρ; Eᶜⁱ=1.0, τ_rim=300.0)

Compute cloud droplet collection (riming) by ice particles.

Cloud droplets are swept up by falling ice particles and freeze onto them.
This increases ice mass and rime mass.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `Eᶜⁱ`: Cloud-ice collection efficiency
- `τ_rim`: Riming timescale [s]

# Returns
- Rate of cloud → ice conversion [kg/kg/s] (also equals rime mass gain rate)

# Reference
P3 uses lookup table integrals. Here we use simplified continuous collection.
"""
@inline function cloud_riming_rate(qᶜˡ, qⁱ, nⁱ, T, ρ;
                                    Eᶜⁱ = 1.0,
                                    τ_rim = 300.0)
    FT = typeof(qᶜˡ)
    T_freeze = FT(273.15)

    qᶜˡ_eff = clamp_positive(qᶜˡ)
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)

    # Thresholds
    q_threshold = FT(1e-8)

    # Only rime below freezing
    below_freezing = T < T_freeze

    # Simplified riming rate: ∂qᶜˡ/∂t = -E × qᶜˡ × qⁱ / τ
    # Rate increases with both cloud and ice content
    rate = ifelse(below_freezing && qᶜˡ_eff > q_threshold && qⁱ_eff > q_threshold,
                   Eᶜⁱ * qᶜˡ_eff * qⁱ_eff / τ_rim,
                   zero(FT))

    return rate
end

"""
    cloud_riming_number_rate(qᶜˡ, Nc, riming_rate)

Compute cloud droplet number sink from riming.

# Arguments
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `Nc`: Cloud droplet number concentration [1/kg]
- `riming_rate`: Cloud riming mass rate [kg/kg/s]

# Returns
- Rate of cloud number reduction [1/kg/s]
"""
@inline function cloud_riming_number_rate(qᶜˡ, Nc, riming_rate)
    FT = typeof(qᶜˡ)

    # Number rate proportional to mass rate
    ratio = safe_divide(Nc, qᶜˡ, zero(FT))

    return -ratio * riming_rate
end

"""
    rain_riming_rate(qʳ, qⁱ, nⁱ, T, ρ; Eʳⁱ=1.0, τ_rim=200.0)

Compute rain collection (riming) by ice particles.

Rain drops are swept up by falling ice particles and freeze onto them.
This increases ice mass and rime mass.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `Eʳⁱ`: Rain-ice collection efficiency
- `τ_rim`: Riming timescale [s]

# Returns
- Rate of rain → ice conversion [kg/kg/s] (also equals rime mass gain rate)
"""
@inline function rain_riming_rate(qʳ, qⁱ, nⁱ, T, ρ;
                                   Eʳⁱ = 1.0,
                                   τ_rim = 200.0)
    FT = typeof(qʳ)
    T_freeze = FT(273.15)

    qʳ_eff = clamp_positive(qʳ)
    qⁱ_eff = clamp_positive(qⁱ)

    # Thresholds
    q_threshold = FT(1e-8)

    # Only rime below freezing
    below_freezing = T < T_freeze

    # Simplified riming rate
    rate = ifelse(below_freezing && qʳ_eff > q_threshold && qⁱ_eff > q_threshold,
                   Eʳⁱ * qʳ_eff * qⁱ_eff / τ_rim,
                   zero(FT))

    return rate
end

"""
    rain_riming_number_rate(qʳ, nʳ, riming_rate)

Compute rain number sink from riming.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `riming_rate`: Rain riming mass rate [kg/kg/s]

# Returns
- Rate of rain number reduction [1/kg/s]
"""
@inline function rain_riming_number_rate(qʳ, nʳ, riming_rate)
    FT = typeof(qʳ)

    # Number rate proportional to mass rate
    ratio = safe_divide(nʳ, qʳ, zero(FT))

    return -ratio * riming_rate
end

"""
    rime_density(T, vᵢ; ρ_rim_min=50.0, ρ_rim_max=900.0)

Compute rime density based on temperature and ice fall speed.

Rime density depends on the degree of riming and temperature.
Denser rime forms at warmer temperatures and higher impact velocities.

# Arguments
- `T`: Temperature [K]
- `vᵢ`: Ice particle fall speed [m/s]
- `ρ_rim_min`: Minimum rime density [kg/m³]
- `ρ_rim_max`: Maximum rime density [kg/m³]

# Returns
- Rime density [kg/m³]

# Reference
P3 uses empirical relations from Cober & List (1993).
"""
@inline function rime_density(T, vᵢ;
                               ρ_rim_min = 50.0,
                               ρ_rim_max = 900.0)
    FT = typeof(T)
    T_freeze = FT(273.15)

    # Temperature factor: denser rime at warmer T
    Tc = T - T_freeze  # Celsius
    Tc_clamped = clamp(Tc, FT(-40), FT(0))

    # Linear interpolation: 100 kg/m³ at -40°C, 400 kg/m³ at 0°C
    ρ_T = FT(100) + (FT(400) - FT(100)) * (Tc_clamped + FT(40)) / FT(40)

    # Velocity factor: denser rime at higher fall speeds
    vᵢ_clamped = clamp(vᵢ, FT(0.1), FT(5))
    ρ_v = FT(1) + FT(0.5) * (vᵢ_clamped - FT(0.1))

    ρ_rim = ρ_T * ρ_v

    return clamp(ρ_rim, ρ_rim_min, ρ_rim_max)
end

#####
##### Phase 2: Shedding and Refreezing (liquid fraction dynamics)
#####

"""
    shedding_rate(qʷⁱ, qⁱ, T, ρ; τ_shed=60.0, qʷⁱ_max_frac=0.3)

Compute liquid shedding rate from ice particles.

When ice particles carry too much liquid coating (from partial melting
or warm riming), excess liquid is shed as rain drops.

# Arguments
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `qⁱ`: Ice mass fraction [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `τ_shed`: Shedding timescale [s]
- `qʷⁱ_max_frac`: Maximum liquid fraction before shedding

# Returns
- Rate of liquid → rain shedding [kg/kg/s]

# Reference
Milbrandt et al. (2025). Liquid shedding above a threshold fraction.
"""
@inline function shedding_rate(qʷⁱ, qⁱ, T, ρ;
                                τ_shed = 60.0,
                                qʷⁱ_max_frac = 0.3)
    FT = typeof(qʷⁱ)
    T_freeze = FT(273.15)

    qʷⁱ_eff = clamp_positive(qʷⁱ)
    qⁱ_eff = clamp_positive(qⁱ)

    # Total particle mass
    qᵗᵒᵗ = qⁱ_eff + qʷⁱ_eff

    # Maximum liquid that can be retained
    qʷⁱ_max = qʷⁱ_max_frac * qᵗᵒᵗ

    # Excess liquid sheds
    qʷⁱ_excess = clamp_positive(qʷⁱ_eff - qʷⁱ_max)

    # Enhanced shedding above freezing
    T_factor = ifelse(T > T_freeze, FT(3), FT(1))

    rate = T_factor * qʷⁱ_excess / τ_shed

    return rate
end

"""
    shedding_number_rate(shed_rate; m_shed=5.2e-7)

Compute rain number source from shedding.

Shed liquid forms rain drops of approximately 1 mm diameter.

# Arguments
- `shed_rate`: Liquid shedding mass rate [kg/kg/s]
- `m_shed`: Mass of shed drops [kg], default corresponds to 1 mm drop

# Returns
- Rate of rain number increase [1/kg/s]
"""
@inline function shedding_number_rate(shed_rate; m_shed = 5.2e-7)
    FT = typeof(shed_rate)

    # Number of drops formed
    return shed_rate / m_shed
end

"""
    refreezing_rate(qʷⁱ, T, ρ; τ_frz=30.0)

Compute refreezing rate of liquid on ice particles.

Below freezing, liquid coating on ice particles refreezes,
transferring mass from liquid-on-ice to ice+rime.

# Arguments
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `τ_frz`: Refreezing timescale [s]

# Returns
- Rate of liquid → ice refreezing [kg/kg/s]

# Reference
Milbrandt et al. (2025). Refreezing in the liquid fraction scheme.
"""
@inline function refreezing_rate(qʷⁱ, T, ρ;
                                  τ_frz = 30.0)
    FT = typeof(qʷⁱ)
    T_freeze = FT(273.15)

    qʷⁱ_eff = clamp_positive(qʷⁱ)

    # Only refreeze below freezing
    below_freezing = T < T_freeze

    # Faster refreezing at colder temperatures
    ΔT = clamp_positive(T_freeze - T)
    T_factor = FT(1) + FT(0.1) * ΔT  # Faster at colder T

    rate = ifelse(below_freezing && qʷⁱ_eff > FT(1e-10),
                   T_factor * qʷⁱ_eff / τ_frz,
                   zero(FT))

    return rate
end

#####
##### Combined P3 tendency calculation
#####

"""
    P3ProcessRates

Container for computed P3 process rates.
Includes Phase 1 (rain, deposition, melting), Phase 2 (aggregation, riming, shedding, nucleation).
"""
struct P3ProcessRates{FT}
    # Phase 1: Rain tendencies
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    rain_evaporation :: FT         # Rain → vapor mass [kg/kg/s]
    rain_self_collection :: FT     # Rain number reduction [1/kg/s]

    # Phase 1: Ice tendencies
    deposition :: FT               # Vapor → ice mass [kg/kg/s]
    melting :: FT                  # Ice → rain mass [kg/kg/s]
    melting_number :: FT           # Ice number reduction from melting [1/kg/s]

    # Phase 2: Ice aggregation
    aggregation :: FT              # Ice number reduction from self-collection [1/kg/s]

    # Phase 2: Riming
    cloud_riming :: FT             # Cloud → ice via riming [kg/kg/s]
    cloud_riming_number :: FT      # Cloud number reduction [1/kg/s]
    rain_riming :: FT              # Rain → ice via riming [kg/kg/s]
    rain_riming_number :: FT       # Rain number reduction [1/kg/s]
    rime_density_new :: FT         # Density of new rime [kg/m³]

    # Phase 2: Shedding and refreezing
    shedding :: FT                 # Liquid on ice → rain [kg/kg/s]
    shedding_number :: FT          # Rain number from shedding [1/kg/s]
    refreezing :: FT               # Liquid on ice → rime [kg/kg/s]
    
    # Ice nucleation (deposition + immersion freezing)
    nucleation_mass :: FT          # New ice mass from deposition nucleation [kg/kg/s]
    nucleation_number :: FT        # New ice number from deposition nucleation [1/kg/s]
    cloud_freezing_mass :: FT      # Cloud → ice mass from immersion freezing [kg/kg/s]
    cloud_freezing_number :: FT    # Cloud number to ice number [1/kg/s]
    rain_freezing_mass :: FT       # Rain → ice mass from immersion freezing [kg/kg/s]
    rain_freezing_number :: FT     # Rain number to ice number [1/kg/s]
    
    # Rime splintering (Hallett-Mossop)
    splintering_mass :: FT         # New ice mass from splintering [kg/kg/s]
    splintering_number :: FT       # New ice number from splintering [1/kg/s]
end

"""
    compute_p3_process_rates(p3, μ, ρ, 𝒰, constants)

Compute all P3 process rates (Phase 1 and Phase 2).

# Arguments
- `p3`: P3 microphysics scheme
- `μ`: Microphysical fields (prognostic and diagnostic)
- `ρ`: Air density [kg/m³]
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
- `P3ProcessRates` containing all computed rates
"""
@inline function compute_p3_process_rates(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    FT = eltype(grid)

    # Extract fields (density-weighted → specific)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    nʳ = @inbounds μ.ρnʳ[i, j, k] / ρ
    qⁱ = @inbounds μ.ρqⁱ[i, j, k] / ρ
    nⁱ = @inbounds μ.ρnⁱ[i, j, k] / ρ
    qᶠ = @inbounds μ.ρqᶠ[i, j, k] / ρ
    bᶠ = @inbounds μ.ρbᶠ[i, j, k] / ρ
    qʷⁱ = @inbounds μ.ρqʷⁱ[i, j, k] / ρ

    # Rime properties
    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))  # Rime fraction
    ρᶠ_current = safe_divide(qᶠ, bᶠ, FT(400))  # Current rime density

    # Thermodynamic state - temperature is computed from the state
    T = temperature(𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor

    # Saturation vapor mixing ratios (from thermodynamic state or compute)
    # For now, use simple approximations - will be replaced with proper thermo interface
    T_freeze = FT(273.15)

    # Clausius-Clapeyron approximation for saturation
    eₛ_liquid = FT(611.2) * exp(FT(17.67) * (T - T_freeze) / (T - FT(29.65)))
    eₛ_ice = FT(611.2) * exp(FT(21.87) * (T - T_freeze) / (T - FT(7.66)))

    # Convert to mass fractions (approximate)
    Rᵈ = FT(287.0)
    Rᵛ = FT(461.5)
    ε = Rᵈ / Rᵛ
    p = ρ * Rᵈ * T  # Approximate pressure
    qᵛ⁺ = ε * eₛ_liquid / (p - (1 - ε) * eₛ_liquid)
    qᵛ⁺ⁱ = ε * eₛ_ice / (p - (1 - ε) * eₛ_ice)

    # Cloud droplet properties
    Nc = p3.cloud.number_concentration
    
    # =========================================================================
    # Phase 1: Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(qᶜˡ, ρ, Nc)
    accr = rain_accretion_rate(qᶜˡ, qʳ, ρ)
    rain_evap = rain_evaporation_rate(qʳ, qᵛ, qᵛ⁺, T, ρ, nʳ)
    rain_self = rain_self_collection_rate(qʳ, nʳ, ρ)

    # =========================================================================
    # Phase 1: Ice deposition/sublimation and melting
    # =========================================================================
    dep = ice_deposition_rate(qⁱ, qᵛ, qᵛ⁺ⁱ, T, ρ, nⁱ)
    melt = ice_melting_rate(qⁱ, nⁱ, T, ρ)
    melt_n = ice_melting_number_rate(qⁱ, nⁱ, melt)

    # =========================================================================
    # Phase 2: Ice aggregation
    # =========================================================================
    agg = ice_aggregation_rate(qⁱ, nⁱ, T, ρ)

    # =========================================================================
    # Phase 2: Riming
    # =========================================================================
    # Cloud droplet collection by ice
    cloud_rim = cloud_riming_rate(qᶜˡ, qⁱ, nⁱ, T, ρ)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nc, cloud_rim)

    # Rain collection by ice
    rain_rim = rain_riming_rate(qʳ, qⁱ, nⁱ, T, ρ)
    rain_rim_n = rain_riming_number_rate(qʳ, nʳ, rain_rim)

    # Rime density for new rime (simplified: use terminal velocity proxy)
    vᵢ = FT(1.0)  # Placeholder fall speed [m/s], will use lookup table later
    ρ_rim_new = rime_density(T, vᵢ)

    # =========================================================================
    # Phase 2: Shedding and refreezing
    # =========================================================================
    shed = shedding_rate(qʷⁱ, qⁱ, T, ρ)
    shed_n = shedding_number_rate(shed)
    refrz = refreezing_rate(qʷⁱ, T, ρ)

    # =========================================================================
    # Ice nucleation (deposition nucleation and immersion freezing)
    # =========================================================================
    nuc_q, nuc_n = deposition_nucleation_rate(T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(qᶜˡ, Nc, T, ρ)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(qʳ, nʳ, T, ρ)

    # =========================================================================
    # Rime splintering (Hallett-Mossop secondary ice production)
    # =========================================================================
    spl_q, spl_n = rime_splintering_rate(cloud_rim, rain_rim, T, ρ)
    
    return P3ProcessRates(
        # Phase 1: Rain
        autoconv, accr, rain_evap, rain_self,
        # Phase 1: Ice
        dep, melt, melt_n,
        # Phase 2: Aggregation
        agg,
        # Phase 2: Riming
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρ_rim_new,
        # Phase 2: Shedding and refreezing
        shed, shed_n, refrz,
        # Ice nucleation
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        # Rime splintering
        spl_q, spl_n
    )
end

#####
##### Individual field tendencies
#####
##### These functions combine process rates into tendencies for each prognostic field.
##### Phase 1 processes: autoconversion, accretion, evaporation, deposition, melting
##### Phase 2 processes: aggregation, riming, shedding, refreezing
#####

"""
    tendency_ρqᶜˡ(rates)

Compute cloud liquid mass tendency from P3 process rates.

Cloud liquid is consumed by:
- Autoconversion (Phase 1)
- Accretion by rain (Phase 1)
- Riming by ice (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqᶜˡ(rates::P3ProcessRates, ρ)
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming + rates.cloud_freezing_mass
    return -ρ * loss
end

"""
    tendency_ρqʳ(rates)

Compute rain mass tendency from P3 process rates.

Rain gains from:
- Autoconversion (Phase 1)
- Accretion (Phase 1)
- Melting (Phase 1)
- Shedding (Phase 2)

Rain loses from:
- Evaporation (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ)
    # Phase 1: gains from autoconv, accr, melt; loses from evap
    # Phase 2: gains from shedding; loses from riming and freezing
    gain = rates.autoconversion + rates.accretion + rates.melting + rates.shedding
    loss = -rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass  # evap is negative
    return ρ * (gain - loss)
end

"""
    tendency_ρnʳ(rates, ρ, qᶜˡ, Nc, m_drop)

Compute rain number tendency from P3 process rates.

Rain number gains from:
- Autoconversion (Phase 1)
- Melting (Phase 1)
- Shedding (Phase 2)

Rain number loses from:
- Self-collection (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ;
                               m_rain_init = 5e-10)  # Initial rain drop mass [kg]
    FT = typeof(ρ)

    # Phase 1: New drops from autoconversion
    n_from_autoconv = rates.autoconversion / m_rain_init

    # Phase 1: New drops from melting (conserve number)
    n_from_melt = safe_divide(nⁱ * rates.melting, qⁱ, zero(FT))

    # Phase 1: Self-collection reduces number (already negative)
    # Phase 2: Shedding creates new drops
    # Phase 2: Riming removes rain drops (already negative)
    
    return ρ * (n_from_autoconv + n_from_melt + 
                rates.rain_self_collection + 
                rates.shedding_number + 
                rates.rain_riming_number)
end

"""
    tendency_ρqⁱ(rates)

Compute ice mass tendency from P3 process rates.

Ice gains from:
- Deposition (Phase 1)
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)

Ice loses from:
- Melting (Phase 1)
"""
@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ)
    # Phase 1: deposition, melting
    # Phase 2: riming (cloud + rain), refreezing, nucleation, freezing, splintering
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.splintering_mass
    loss = rates.melting
    return ρ * (gain - loss)
end

"""
    tendency_ρnⁱ(rates)

Compute ice number tendency from P3 process rates.

Ice number gains from:
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)

Ice number loses from:
- Melting (Phase 1)
- Aggregation (Phase 2)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering
    gain = rates.nucleation_number + rates.cloud_freezing_number + 
           rates.rain_freezing_number + rates.splintering_number
    # melting_number and aggregation are already negative (represent losses)
    loss_rates = rates.melting_number + rates.aggregation
    return ρ * (gain + loss_rates)
end

"""
    tendency_ρqᶠ(rates)

Compute rime mass tendency from P3 process rates.

Rime mass gains from:
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Immersion freezing (frozen cloud/rain becomes rimed ice) (Phase 2)

Rime mass loses from:
- Melting (proportional to rime fraction) (Phase 1)
"""
@inline function tendency_ρqᶠ(rates::P3ProcessRates, ρ, Fᶠ)
    # Phase 2: gains from riming, refreezing, and freezing
    # Frozen cloud/rain becomes fully rimed ice (100% rime fraction for new frozen particles)
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass
    # Phase 1: melts proportionally with ice mass
    loss = Fᶠ * rates.melting
    return ρ * (gain - loss)
end

"""
    tendency_ρbᶠ(rates, Fᶠ, ρᶠ)

Compute rime volume tendency from P3 process rates.

Rime volume changes with rime mass: ∂bᶠ/∂t = ∂qᶠ/∂t / ρ_rime
"""
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ)
    FT = typeof(ρ)

    ρᶠ_safe = max(ρᶠ, FT(100))
    ρ_rim_new_safe = max(rates.rime_density_new, FT(100))
    
    # Phase 2: Volume gain from new rime (cloud + rain riming + refreezing)
    # Use density of new rime for fresh rime, current density for refreezing
    volume_gain = (rates.cloud_riming + rates.rain_riming) / ρ_rim_new_safe + 
                   rates.refreezing / ρᶠ_safe
    
    # Phase 1: Volume loss from melting (proportional to rime fraction)
    volume_loss = Fᶠ * rates.melting / ρᶠ_safe

    return ρ * (volume_gain - volume_loss)
end

"""
    tendency_ρzⁱ(rates, μ, λ)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

For P3 3-moment, Z tendencies are computed more accurately using
size distribution integrals. This simplified version uses proportional scaling.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)
    
    # Simplified: Z changes proportionally to mass changes
    # More accurate version would use full integral formulation
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))
    
    # Net mass change for ice
    mass_change = rates.deposition - rates.melting + 
                  rates.cloud_riming + rates.rain_riming + rates.refreezing
    
    return ρ * ratio * mass_change
end

"""
    tendency_ρqʷⁱ(rates)

Compute liquid on ice tendency from P3 process rates.

Liquid on ice:
- Gains from partial melting above freezing (currently in melting rate)
- Loses from shedding (Phase 2)
- Loses from refreezing (Phase 2)
"""
@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ)
    # Phase 2: loses from shedding and refreezing
    # Gains: In full P3, partial melting above freezing adds to qʷⁱ
    # For now, melting goes directly to rain; this is a placeholder
    return -ρ * (rates.shedding + rates.refreezing)
end

#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Uses power-law relationships with air density correction.
#####

"""
    rain_terminal_velocity_mass_weighted(qʳ, nʳ, ρ; a=842.0, b=0.8, ρ₀=1.225)

Compute mass-weighted terminal velocity for rain.

Uses the power-law relationship from Klemp & Wilhelmson (1978) and
Seifert & Beheng (2006):

    v(D) = a × D^b × √(ρ₀/ρ)

The mass-weighted velocity is computed assuming a gamma size distribution:

    Vₘ = a × D̄ₘ^b × √(ρ₀/ρ)

where D̄ₘ is the mass-weighted mean diameter.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]
- `a`: Velocity coefficient [m^(1-b)/s]
- `b`: Velocity exponent
- `ρ₀`: Reference air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)

# Reference
Seifert, A. and Beheng, K. D. (2006). A two-moment cloud microphysics
parameterization for mixed-phase clouds. Meteor. Atmos. Phys.
"""
@inline function rain_terminal_velocity_mass_weighted(qʳ, nʳ, ρ;
                                                       a = 842.0,
                                                       b = 0.8,
                                                       ρ₀ = 1.225)
    FT = typeof(qʳ)

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1))  # Avoid division by zero

    # Mean rain drop mass
    m̄ = qʳ_eff / nʳ_eff

    # Mass-weighted mean diameter (assuming spherical drops)
    # m = (π/6) ρʷ D³ → D = (6m / (π ρʷ))^(1/3)
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * FT(ρʷ)))

    # Density correction factor
    ρ_correction = sqrt(FT(ρ₀) / ρ)

    # Clamp diameter to physical range [0.1 mm, 5 mm]
    D̄ₘ_clamped = clamp(D̄ₘ, FT(1e-4), FT(5e-3))

    # Terminal velocity
    vₜ = a * D̄ₘ_clamped^b * ρ_correction

    # Clamp to reasonable range [0.1, 15] m/s
    return clamp(vₜ, FT(0.1), FT(15))
end

"""
    rain_terminal_velocity_number_weighted(qʳ, nʳ, ρ; a=842.0, b=0.8, ρ₀=1.225)

Compute number-weighted terminal velocity for rain.

Similar to mass-weighted but uses number-weighted mean diameter.

# Arguments
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function rain_terminal_velocity_number_weighted(qʳ, nʳ, ρ;
                                                         a = 842.0,
                                                         b = 0.8,
                                                         ρ₀ = 1.225)
    FT = typeof(qʳ)

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1))

    # Mean rain drop mass
    m̄ = qʳ_eff / nʳ_eff

    # Number-weighted mean diameter is smaller than mass-weighted
    # For gamma distribution: D̄ₙ ≈ D̄ₘ × (μ+1)/(μ+4) where μ is shape parameter
    # Simplified: use D̄ₘ with factor ~0.6
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * FT(ρʷ)))
    D̄ₙ = FT(0.6) * D̄ₘ

    ρ_correction = sqrt(FT(ρ₀) / ρ)
    D̄ₙ_clamped = clamp(D̄ₙ, FT(1e-4), FT(5e-3))

    vₜ = a * D̄ₙ_clamped^b * ρ_correction

    return clamp(vₜ, FT(0.1), FT(15))
end

"""
    ice_terminal_velocity_mass_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; ρ₀=1.225)

Compute mass-weighted terminal velocity for ice.

Uses regime-dependent fall speeds following Mitchell (1996) and
the P3 particle property model.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `ρ₀`: Reference air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)

# Reference
Morrison, H. and Milbrandt, J. A. (2015). Parameterization of cloud
microphysics based on the prediction of bulk ice particle properties.
Part I: Scheme description and idealized tests. J. Atmos. Sci.
"""
@inline function ice_terminal_velocity_mass_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ;
                                                      ρ₀ = 1.225)
    FT = typeof(qⁱ)

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Effective ice density depends on riming
    # Unrimed: ρ_eff ≈ 100-200 kg/m³ (aggregates/dendrites)
    # Heavily rimed: ρ_eff ≈ ρᶠ ≈ 400-900 kg/m³ (graupel)
    Fᶠ_clamped = clamp(Fᶠ, FT(0), FT(1))
    ρᶠ_clamped = clamp(ρᶠ, FT(50), FT(900))
    ρ_eff_unrimed = FT(100)  # Aggregate effective density
    ρ_eff = ρ_eff_unrimed + Fᶠ_clamped * (ρᶠ_clamped - ρ_eff_unrimed)

    # Effective diameter assuming spherical with effective density
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρ_eff))

    # Fall speed depends on particle type:
    # - Small ice (D < 100 μm): v ≈ 700 D² (Stokes regime)
    # - Large unrimed (D > 100 μm): v ≈ 11.7 D^0.41 (Mitchell 1996)
    # - Rimed/graupel: v ≈ 19.3 D^0.37

    D_clamped = clamp(D̄ₘ, FT(1e-5), FT(0.02))  # 10 μm to 20 mm
    D_threshold = FT(100e-6)  # 100 μm

    # Coefficients interpolated based on riming
    # Unrimed: a=11.7, b=0.41 (aggregates)
    # Rimed: a=19.3, b=0.37 (graupel-like)
    a_unrimed = FT(11.7)
    b_unrimed = FT(0.41)
    a_rimed = FT(19.3)
    b_rimed = FT(0.37)

    a = a_unrimed + Fᶠ_clamped * (a_rimed - a_unrimed)
    b = b_unrimed + Fᶠ_clamped * (b_rimed - b_unrimed)

    # Density correction
    ρ_correction = sqrt(FT(ρ₀) / ρ)

    # Terminal velocity (large particle regime)
    vₜ_large = a * D_clamped^b * ρ_correction

    # Small particle (Stokes) regime
    vₜ_small = FT(700) * D_clamped^2 * ρ_correction

    # Blend between regimes
    vₜ = ifelse(D_clamped < D_threshold, vₜ_small, vₜ_large)

    # Clamp to reasonable range [0.01, 8] m/s
    return clamp(vₜ, FT(0.01), FT(8))
end

"""
    ice_terminal_velocity_number_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; ρ₀=1.225)

Compute number-weighted terminal velocity for ice.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_number_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ;
                                                        ρ₀ = 1.225)
    FT = typeof(qⁱ)

    # Number-weighted velocity is smaller than mass-weighted
    # Approximate ratio: Vₙ/Vₘ ≈ 0.6 for typical distributions
    vₘ = ice_terminal_velocity_mass_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; ρ₀)

    return FT(0.6) * vₘ
end

"""
    ice_terminal_velocity_reflectivity_weighted(qⁱ, nⁱ, zⁱ, Fᶠ, ρᶠ, ρ; ρ₀=1.225)

Compute reflectivity-weighted (Z-weighted) terminal velocity for ice.

Needed for the sixth moment (reflectivity) sedimentation in 3-moment P3.

# Arguments
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `zⁱ`: Ice sixth moment (reflectivity proxy) [m⁶/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Reflectivity-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_reflectivity_weighted(qⁱ, nⁱ, zⁱ, Fᶠ, ρᶠ, ρ;
                                                              ρ₀ = 1.225)
    FT = typeof(qⁱ)

    # Z-weighted velocity is larger than mass-weighted (biased toward large particles)
    # Approximate ratio: Vᵤ/Vₘ ≈ 1.2 for typical distributions
    vₘ = ice_terminal_velocity_mass_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; ρ₀)

    return FT(1.2) * vₘ
end
