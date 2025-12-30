"""
Kessler warm-rain bulk microphysics scheme.

A "warm-rain" (Kessler-type) bulk microphysics scheme with water vapor, cloud liquid, and rain.

Breeze uses mass fractions (q = mass_species / mass_total), while Kessler formulas use
mixing ratios (r = mass_species / mass_dry_air). Conversion:
- r = q / (1 - qᵗ)  where qᵗ is total moisture mass fraction
- q = r * (1 - qᵗ)

Prognostic variables (in Breeze mass fraction form):
- qᶜˡ: cloud liquid water mass fraction
- qʳ: rain water mass fraction

Diagnostic variable:
- qᵛ: water vapor mass fraction = qᵗ - qᶜˡ - qʳ (from Breeze's total moisture qᵗ)

Reference: Kessler (1969), "On the Distribution and Continuity of Water Substance in Atmospheric Circulations"
"""

using Oceananigans: Oceananigans, CenterField, Field, Center, Face, Nothing as ONothing
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: ZFaceField, ZeroField
using Oceananigans.Operators: Δzᶜᶜᶜ
using DocStringExtensions: TYPEDSIGNATURES

import ..AtmosphereModels:
    prognostic_field_names,
    materialize_microphysical_fields,
    microphysical_velocities,
    compute_moisture_fractions,
    microphysical_tendency,
    update_microphysical_fields!,
    precipitation_rate,
    surface_precipitation_flux,
    maybe_adjust_thermodynamic_state

using ..Thermodynamics:
    MoistureMassFractions,
    PlanarLiquidSurface,
    saturation_specific_humidity,
    temperature,
    density,
    liquid_latent_heat,
    mixture_heat_capacity,
    total_specific_moisture,
    exner_function

#####
##### Kessler microphysics struct
#####

"""
$(TYPEDSIGNATURES)

Kessler warm-rain microphysics scheme with cloud liquid and rain.

# Fields
- `autoconversion_rate`: Rate constant for autoconversion (cloud → rain), k₁ [s⁻¹]. Default: 0.001 s⁻¹
- `autoconversion_threshold`: Cloud water threshold for autoconversion, a [kg kg⁻¹]. Default: 0.001 kg kg⁻¹  
- `accretion_rate`: Rate constant for accretion (collection of cloud by rain), k₂ [s⁻¹]. Default: 2.2 s⁻¹

Note: The reference density ρ₀ for terminal velocity is obtained from Breeze's reference state
(ρᵣ[i,j,1]) rather than being stored as a parameter.
"""
struct KesslerMicrophysics{FT}
    autoconversion_rate :: FT       # k₁ [s⁻¹]
    autoconversion_threshold :: FT  # a [kg kg⁻¹]
    accretion_rate :: FT            # k₂ [s⁻¹]
end

Base.summary(::KesslerMicrophysics) = "KesslerMicrophysics"

function Base.show(io::IO, km::KesslerMicrophysics{FT}) where FT
    print(io, "KesslerMicrophysics{$FT}:\n",
              "├── autoconversion_rate: ", km.autoconversion_rate, " s⁻¹\n",
              "├── autoconversion_threshold: ", km.autoconversion_threshold, " kg kg⁻¹\n",
              "└── accretion_rate: ", km.accretion_rate, " s⁻¹")
end

"""
$(TYPEDSIGNATURES)

Construct a `KesslerMicrophysics` scheme with default parameters from Kessler (1969).

# Arguments
- `FT`: Float type to use (defaults to `Oceananigans.defaults.FloatType`)

# Keyword Arguments
- `autoconversion_rate`: Rate constant k₁ [s⁻¹]. Default: 0.001 s⁻¹
- `autoconversion_threshold`: Cloud water threshold a [kg kg⁻¹]. Default: 0.001 kg kg⁻¹
- `accretion_rate`: Rate constant k₂ [s⁻¹]. Default: 2.2 s⁻¹
"""
function KesslerMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                             autoconversion_rate = 0.001,
                             autoconversion_threshold = 0.001,
                             accretion_rate = 2.2)

    return KesslerMicrophysics{FT}(convert(FT, autoconversion_rate),
                                   convert(FT, autoconversion_threshold),
                                   convert(FT, accretion_rate))
end

const KM = KesslerMicrophysics

#####
##### Mass fraction ↔ mixing ratio conversion
#####

"""
Convert mass fraction q to mixing ratio r.

r = q / (1 - qᵗ)

where qᵗ is total moisture mass fraction and (1 - qᵗ) is dry air mass fraction.
"""
@inline function mass_fraction_to_mixing_ratio(q, qᵗ)
    qᵈ = 1 - qᵗ  # dry air mass fraction
    return q / qᵈ
end

"""
Convert mixing ratio r to mass fraction q.

q = r * (1 - qᵗ)

where qᵗ is total moisture mass fraction and (1 - qᵗ) is dry air mass fraction.
Also used to convert mixing ratio tendencies to mass fraction tendencies.
"""
@inline function mixing_ratio_to_mass_fraction(r, qᵗ)
    qᵈ = 1 - qᵗ  # dry air mass fraction
    return r * qᵈ
end

#####
##### Microphysics interface implementation
#####

# Only cloud liquid and rain are prognostic; vapor is diagnosed from qᵗ
prognostic_field_names(::KM) = (:ρqᶜˡ, :ρqʳ)

function materialize_microphysical_fields(km::KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted mass fractions)
    ρqᶜˡ = CenterField(grid; boundary_conditions=boundary_conditions.ρqᶜˡ)
    ρqʳ = CenterField(grid; boundary_conditions=boundary_conditions.ρqʳ)

    # Diagnostic fields (mass fractions)
    qᵛ = CenterField(grid)
    qᶜˡ = CenterField(grid)
    qʳ = CenterField(grid)

    # Cached microphysics rates (computed once per timestep in update_microphysical_fields!)
    # These are tendencies in mixing ratio space [kg kg⁻¹ s⁻¹]
    Cₖ = CenterField(grid)  # Condensation rate
    Eₖ = CenterField(grid)  # Cloud evaporation rate
    Aₖ = CenterField(grid)  # Autoconversion rate
    Kₖ = CenterField(grid)  # Accretion rate
    Eʳ = CenterField(grid)  # Rain evaporation rate

    # Rain terminal velocity (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    wʳ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=wʳ_bcs)

    # Surface precipitation rate (2D field, m/s)
    # This is the volume flux of rain at the surface: wʳ * qʳ (positive = precipitation out of domain)
    precipitation_rate = Field{Center, Center, ONothing}(grid)

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, Cₖ, Eₖ, Aₖ, Kₖ, Eʳ, wʳ, precipitation_rate)
end

@inline function update_microphysical_fields!(μ, km::KM, i, j, k, grid, ρ, 𝒰, constants)
    FT = eltype(grid)
    @inbounds begin
        # Get total moisture from thermodynamic state
        # In the moisture_mass_fractions, vapor contains qᵛ and liquid contains total condensate (qᶜˡ + qʳ)
        # But we need to separate qᶜˡ and qʳ from prognostic fields
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
        
        # Vapor is diagnosed: qᵛ = qᵗ - qᶜˡ - qʳ
        # where qᵗ = total moisture from Breeze's prognostic ρqᵗ
        qᵗ = total_specific_moisture(𝒰)
        qᵛ = max(zero(qᵗ), qᵗ - qᶜˡ - qʳ)
        
        # Update diagnostic fields
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        
        # Compute and cache microphysics rates (once per timestep)
        T = temperature(𝒰, constants)
        
        # Convert mass fractions to mixing ratios for Kessler formulas
        rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
        rᶜˡ = mass_fraction_to_mixing_ratio(qᶜˡ, qᵗ)
        rʳ = mass_fraction_to_mixing_ratio(qʳ, qᵗ)
        
        # Saturation: compute in mixing ratio space
        qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
        rᵛ⁺ = mass_fraction_to_mixing_ratio(qᵛ⁺, qᵗ)
        
        # Latent heat and heat capacity
        L = liquid_latent_heat(T, constants)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ)
        cₚ = mixture_heat_capacity(q, constants)
        
        # Compute all rates in mixing ratio space
        D = condensation_denominator(T, rᵛ⁺, L, cₚ)
        Cₖ_val = condensation_rate(rᵛ, rᵛ⁺, D)
        Eₖ_val = cloud_evaporation_rate(rᵛ, rᶜˡ, rᵛ⁺, D)
        Aₖ_val = autoconversion_rate(rᶜˡ, km)
        Kₖ_val = accretion_rate(rᶜˡ, rʳ, km)
        
        # Compute rain evaporation rate, coupled to cloud evaporation
        # Following Fortran DCMIP2016: rain evaporation is limited by the
        # remaining saturation deficit after cloud evaporation
        Eʳ_uncoupled = rain_evaporation_rate(ρ, rᵛ, rʳ, rᵛ⁺)
        
        # Net cloud condensate change: Cₖ - Eₖ (positive = condensation)
        # The saturation deficit "used up" by cloud evaporation is Eₖ
        # Remaining deficit available for rain evaporation: max(subsaturation - Eₖ/D, 0)
        # But simpler: limit rain evaporation to max(-Cₖ + Eₖ - rᶜˡ, 0) based on available deficit
        # Following Fortran: ern = min(ern, max(-prod - qc, 0), qr)
        # where prod = (qv - qvs)/D is net condensation (negative when subsaturated)
        # Here: prod ≈ Cₖ - Eₖ (net condensation rate)
        # Remaining deficit after cloud processes: max(-(Cₖ - Eₖ) - rᶜˡ, 0) doesn't quite work
        # Simpler interpretation: rain evaporation limited by remaining subsaturation after cloud evaporation
        # If cloud fully evaporates, remaining deficit = subsaturation - rᶜˡ
        remaining_deficit = max(zero(FT), (rᵛ⁺ - rᵛ) - Eₖ_val)
        Eʳ_val = min(Eʳ_uncoupled, remaining_deficit, rʳ)
        
        # Store rates for use in microphysical_tendency
        μ.Cₖ[i, j, k] = Cₖ_val
        μ.Eₖ[i, j, k] = Eₖ_val
        μ.Aₖ[i, j, k] = Aₖ_val
        μ.Kₖ[i, j, k] = Kₖ_val
        μ.Eʳ[i, j, k] = Eʳ_val
        
        # Compute terminal velocity at face k using surface density as reference
        # Following Fortran DCMIP2016: ρ₀ = ρ(1) (surface density from column)
        # Since we don't have direct access to dynamics here, use a fixed standard value
        # TODO: Pass reference_density field to update_microphysical_fields! for proper ρ₀
        ρ₀ = convert(FT, 1.225)  # Standard sea-level air density [kg/m³]
        wₜ = rain_terminal_velocity(ρ, rʳ, ρ₀)
        wʳ = -wₜ  # Negative = downward
        μ.wʳ[i, j, k] = wʳ
        
        # Compute surface precipitation rate at k=1 only (2D field)
        # precipitation_rate = -wʳ * qʳ [m/s] (positive = precipitation falling out)
        if k == 1
            μ.precipitation_rate[i, j, 1] = wₜ * qʳ
        end
    end
    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, ::KM, ρ, qᵗ, μ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
    end
    # Vapor is diagnosed from total moisture
    qᵛ = max(zero(qᵗ), qᵗ - qᶜˡ - qʳ)
    
    # Rain is counted as liquid in the liquid-ice potential temperature definition
    # Total liquid for θˡⁱ = cloud liquid + rain
    return MoistureMassFractions(qᵛ, qᶜˡ + qʳ)
end

# No saturation adjustment for explicit Kessler scheme
@inline maybe_adjust_thermodynamic_state(i, j, k, 𝒰, ::KM, ρ, μ, qᵗ, constants) = 𝒰

#####
##### Terminal velocity for rain sedimentation
#####

"""
$(TYPEDSIGNATURES)

Compute the terminal fall speed of rain droplets [m s⁻¹].

The terminal velocity is given by (following the DCMIP2016 Fortran Kessler reference):

```math
wₜ = 36.34 (0.001 ρ rʳ)^{0.1364} (ρ₀ / ρ)^{1/2}
```

where ρ is air density [kg m⁻³], rʳ is rain mixing ratio [kg kg⁻¹], and ρ₀ is reference 
surface density [kg m⁻³].

Note: The original formula gives velocity in cm s⁻¹ with coefficient 3634.
Here we use 36.34 m s⁻¹ for SI units.
"""
@inline function rain_terminal_velocity(ρ, rʳ, ρ₀)
    FT = typeof(ρ)
    # Match Fortran: r(k) = 0.001 * rho(k) is used inside (qr * r)^0.1364.
    ρrʳ = convert(FT, 0.001) * ρ * max(zero(FT), rʳ)
    
    # Avoid issues when there's no rain
    ρrʳ <= zero(FT) && return zero(FT)
    
    # Coefficient 36.34 m/s (converted from 3634 cm/s)
    # rhalf = sqrt(ρ₀/ρ) as in Fortran reference
    wₜ = convert(FT, 36.34) * ρrʳ^convert(FT, 0.1364) * sqrt(ρ₀ / ρ)
    
    return wₜ
end

"""
$(TYPEDSIGNATURES)

Compute the sedimentation flux for rain at level k.

Uses upstream differencing following the Fortran Kessler reference:
```math
\\text{sed}_k = \\frac{(ρ r^r w_t)_{k+1} - (ρ r^r w_t)_k}{ρ_k Δz_k}
```

At the top boundary (k = Nz), uses:
```math
\\text{sed}_{Nz} = -\\frac{r^r_{Nz} \\cdot w_{t,Nz}}{0.5 \\cdot Δz_{Nz}}
```

At the bottom boundary (k = 1), rain falling out is removed (precip).
"""
@inline function sedimentation_tendency(i, j, k, grid, ρᵣ, μ)
    FT = eltype(grid)
    Nz = size(grid, 3)
    
    # Get Δz at this level
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    
    @inbounds begin
        # Column densities (use reference-state profile to access k+1 in a local kernel)
        ρ_k = ρᵣ[i, j, k]
        ρ₀ = ρᵣ[i, j, 1]

        # Current level moisture: convert mass fractions -> mixing ratios (no q≈r shortcut)
        qʳ_k = μ.qʳ[i, j, k]
        qᵛ_k = μ.qᵛ[i, j, k]
        qᶜˡ_k = μ.qᶜˡ[i, j, k]
        qᵗ_k = min(qᵛ_k + qᶜˡ_k + qʳ_k, one(FT) - eps(one(FT)))
        rʳ_k = mass_fraction_to_mixing_ratio(qʳ_k, qᵗ_k)

        wₜ_k = rain_terminal_velocity(ρ_k, rʳ_k, ρ₀)
        
        if k == Nz
            # Top boundary: no flux from above, only outflow
            # sed = -qr * vt / (0.5 * Δz)  following Fortran
            Δz_half = Δz / 2
            sed = -rʳ_k * wₜ_k / Δz_half
        else
            # Interior: Fortran-style flux divergence normalized by local density (ρ_k)
            ρ_kp1 = ρᵣ[i, j, k+1]

            qʳ_kp1 = μ.qʳ[i, j, k+1]
            qᵛ_kp1 = μ.qᵛ[i, j, k+1]
            qᶜˡ_kp1 = μ.qᶜˡ[i, j, k+1]
            qᵗ_kp1 = min(qᵛ_kp1 + qᶜˡ_kp1 + qʳ_kp1, one(FT) - eps(one(FT)))
            rʳ_kp1 = mass_fraction_to_mixing_ratio(qʳ_kp1, qᵗ_kp1)

            wₜ_kp1 = rain_terminal_velocity(ρ_kp1, rʳ_kp1, ρ₀)

            F_kp1 = ρ_kp1 * rʳ_kp1 * wₜ_kp1
            F_k = ρ_k * rʳ_k * wₜ_k
            sed = (F_kp1 - F_k) / (ρ_k * Δz)
        end
        
        # At bottom (k=1), rain that would fall below is removed (precipitation)
        # This is handled by the flux divergence naturally - flux out at bottom
        # is not balanced by flux from below
    end
    
    return sed
end

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities for the Kessler scheme.

For rain (`ρqʳ`), returns the terminal velocity field `wʳ` so that Breeze's
advection machinery handles sedimentation. Cloud liquid has no sedimentation velocity.
"""
@inline function microphysical_velocities(::KM, μ, ::Val{:ρqʳ})
    wʳ = μ.wʳ
    return (; u = ZeroField(), v = ZeroField(), w = wʳ)
end
@inline microphysical_velocities(::KM, μ, ::Val{:ρqᶜˡ}) = nothing
@inline microphysical_velocities(::KM, μ, name) = nothing

#####
##### Source term calculations (in mixing ratio space)
#####

"""
$(TYPEDSIGNATURES)

Compute the denominator D for condensation/evaporation rate.

This follows Klemp & Wilhelmson (1978) eq. 3.10 and the DCMIP Kessler implementation.
The formula derives from the Tetens saturation vapor pressure approximation.

```math
D = 1 + \\frac{rᵛ⁺ \\cdot 4093 \\cdot L}{cₚ (T - 36)^2}
```

where T is temperature in **Kelvin**. The constant 36 K comes from the Tetens formula:
in Celsius, the denominator is (Tc + 237.3), and converting to Kelvin gives
(T - 273.15 + 237.3) = (T - 35.85) ≈ (T - 36).
"""
@inline function condensation_denominator(T, rᵛ⁺, L, cₚ)
    FT = typeof(T)
    return one(FT) + rᵛ⁺ * convert(FT, 4093) * L / (cₚ * (T - convert(FT, 36))^2)
end

"""
$(TYPEDSIGNATURES)

Compute condensation rate Cₖ [kg kg⁻¹ s⁻¹] in mixing ratio space.

If supersaturated (rᵛ > rᵛ⁺): Cₖ = (rᵛ - rᵛ⁺) / D
Otherwise: Cₖ = 0
"""
@inline function condensation_rate(rᵛ, rᵛ⁺, D)
    FT = typeof(rᵛ)
    return rᵛ > rᵛ⁺ ? (rᵛ - rᵛ⁺) / D : zero(FT)
end

"""
$(TYPEDSIGNATURES)

Compute cloud evaporation rate Eₖ [kg kg⁻¹ s⁻¹] in mixing ratio space.

If subsaturated (rᵛ < rᵛ⁺): Eₖ = min(rᶜˡ, (rᵛ⁺ - rᵛ) / D)
Otherwise: Eₖ = 0

The evaporation is limited by available cloud water.
"""
@inline function cloud_evaporation_rate(rᵛ, rᶜˡ, rᵛ⁺, D)
    FT = typeof(rᵛ)
    if rᵛ < rᵛ⁺
        return min(rᶜˡ, (rᵛ⁺ - rᵛ) / D)
    else
        return zero(FT)
    end
end

"""
$(TYPEDSIGNATURES)

Compute autoconversion rate Aₖ [kg kg⁻¹ s⁻¹] in mixing ratio space.

```math
Aₖ = \\max(0, k₁ (rᶜˡ - a))
```
"""
@inline function autoconversion_rate(rᶜˡ, km::KM)
    FT = typeof(rᶜˡ)
    k₁ = km.autoconversion_rate
    a = km.autoconversion_threshold
    return max(zero(FT), k₁ * (rᶜˡ - a))
end

"""
$(TYPEDSIGNATURES)

Compute accretion rate Kₖ [kg kg⁻¹ s⁻¹] in mixing ratio space.

```math
Kₖ = k₂ rᶜˡ rʳ^{0.875}
```
"""
@inline function accretion_rate(rᶜˡ, rʳ, km::KM)
    FT = typeof(rᶜˡ)
    k₂ = km.accretion_rate
    rʳ_safe = max(zero(FT), rʳ)
    return k₂ * rᶜˡ * rʳ_safe^convert(FT, 0.875)
end

"""
$(TYPEDSIGNATURES)

Compute rain evaporation rate Eʳ [kg kg⁻¹ s⁻¹] in mixing ratio space.

```math
Eʳ = \\frac{(1 - rᵛ/rᵛ⁺) C (ρ rʳ)^{0.525}}{ρ (5.4 \\times 10^5 + 2.55 \\times 10^6 / (ρ rᵛ⁺))}
```

where the ventilation factor is:
```math
C = 1.6 + 124.9 (ρ rʳ)^{0.2046}
```
"""
@inline function rain_evaporation_rate(ρ, rᵛ, rʳ, rᵛ⁺)
    FT = typeof(ρ)
    
    # No evaporation if saturated or supersaturated
    rᵛ >= rᵛ⁺ && return zero(FT)
    
    # No evaporation if no rain
    rʳ <= zero(FT) && return zero(FT)
    
    ρrʳ = ρ * rʳ
    ρrᵛ⁺ = ρ * rᵛ⁺
        
    # Ventilation factor
    C = convert(FT, 1.6) + convert(FT, 124.9) * ρrʳ^convert(FT, 0.2046)
    
    # Subsaturation factor
    subsaturation = one(FT) - rᵛ / rᵛ⁺
    
    # Denominator
    denom = convert(FT, 5.4e5) + convert(FT, 2.55e6) / ρrᵛ⁺
    
    # Rain evaporation rate
    Eʳ = subsaturation * C * ρrʳ^convert(FT, 0.525) / (ρ * denom)
    
    # Limit by available rain
    return min(Eʳ, rʳ)
end

#####
##### Microphysical tendencies
#####

"""
$(TYPEDSIGNATURES)

Compute the tendency for cloud liquid density (ρqᶜˡ).

The rates Cₖ, Eₖ, Aₖ, Kₖ are computed once per timestep in `update_microphysical_fields!`
and cached in the microphysical fields.

```math
\\frac{∂(ρqᶜˡ)}{∂t} = ρ \\cdot (1 - qᵗ) \\cdot (Cₖ - Eₖ - Aₖ - Kₖ)
```

where the rates Cₖ, Eₖ, Aₖ, Kₖ are in mixing ratio space.
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqᶜˡ}, ρᵣ, μ, 𝒰, constants)
    # Get thermodynamic quantities
    ρ = density(𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    
    # Get cached rates (computed in update_microphysical_fields!)
    @inbounds begin
        Cₖ = μ.Cₖ[i, j, k]
        Eₖ = μ.Eₖ[i, j, k]
        Aₖ = μ.Aₖ[i, j, k]
        Kₖ = μ.Kₖ[i, j, k]
    end
    
    # Tendency in mixing ratio space: drᶜˡ/dt = Cₖ - Eₖ - Aₖ - Kₖ
    drᶜˡdt = Cₖ - Eₖ - Aₖ - Kₖ
    
    # Convert to mass fraction tendency
    dqᶜˡdt = mixing_ratio_to_mass_fraction(drᶜˡdt, qᵗ)
    
    return ρ * dqᶜˡdt
end

"""
$(TYPEDSIGNATURES)

Compute the tendency for rain density (ρqʳ).

The rates Aₖ, Kₖ, Eʳ are computed once per timestep in `update_microphysical_fields!`
and cached in the microphysical fields.

**Sedimentation** is handled by Breeze's advection machinery via `microphysical_velocities`,
which adds the terminal velocity `wʳ` to the rain tracer advection.

```math
\\frac{∂(ρqʳ)}{∂t} = ρ \\cdot (1 - qᵗ) \\cdot (Aₖ + Kₖ - Eʳ)
```
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqʳ}, ρᵣ, μ, 𝒰, constants)
    # Get thermodynamic quantities
    ρ = density(𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    
    # Get cached rates (computed in update_microphysical_fields!)
    @inbounds begin
        Aₖ = μ.Aₖ[i, j, k]
        Kₖ = μ.Kₖ[i, j, k]
        Eʳ = μ.Eʳ[i, j, k]
    end
    
    # Tendency in mixing ratio space: drʳ/dt = Aₖ + Kₖ - Eʳ
    # Note: sedimentation is handled via microphysical_velocities, not here
    drʳdt = Aₖ + Kₖ - Eʳ
    
    # Convert to mass fraction tendency
    dqʳdt = mixing_ratio_to_mass_fraction(drʳdt, qᵗ)
    
    return ρ * dqʳdt
end

# Default: no tendency for other variables
# Phase changes (condensation/evaporation of cloud) conserve liquid-ice potential temperature by design.
# However, rain evaporation releases latent heat and cools the air, which requires an explicit θ tendency.
@inline microphysical_tendency(i, j, k, grid, ::KM, name, ρ, μ, 𝒰, constants) = zero(grid)

"""
$(TYPEDSIGNATURES)

Compute the tendency for potential temperature density (ρθˡⁱ) due to rain evaporation.

Rain evaporation cools the air by releasing latent heat:
```math
\\frac{∂(ρθ)}{∂t} = -ρ \\cdot \\frac{L}{cₚ Π} \\cdot Eʳ
```

where Eʳ is the rain evaporation rate (in mass fraction space), L is the latent heat,
cₚ is the mixture heat capacity, and Π is the Exner function.

Note: Condensation/evaporation of cloud liquid is already accounted for in the
liquid-ice potential temperature formulation. Only rain evaporation (which occurs
after rain has fallen from cloud) requires an explicit θ tendency.
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρθ}, ρ_local, μ, 𝒰, constants)
    # Get thermodynamic quantities
    ρ = density(𝒰, constants)
    T = temperature(𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    
    # Get moisture fractions for heat capacity calculation
    @inbounds qᵛ = μ.qᵛ[i, j, k]
    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]
    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ)
    
    # Get rain evaporation rate (in mixing ratio space)
    @inbounds Eʳ = μ.Eʳ[i, j, k]
    
    # Convert to mass fraction space
    dqʳdt_evap = mixing_ratio_to_mass_fraction(Eʳ, qᵗ)
    
    # Latent heat and heat capacity
    L = liquid_latent_heat(T, constants)
    cₚ = mixture_heat_capacity(q, constants)
    
    # Exner function for conversion to potential temperature
    Π = exner_function(𝒰, constants)
    
    # Rain evaporation cools the air:
    # dθ/dt = -L/(cₚ Π) * (dqʳ/dt from evaporation)
    # The negative sign: evaporation (Eʳ > 0 means rain is disappearing) cools air
    dθdt = -L / (cₚ * Π) * dqʳdt_evap
    
    return ρ * dθdt
end

#####
##### Precipitation rate diagnostics
#####

"""
$(TYPEDSIGNATURES)

Return the precipitation rate field for the Kessler scheme.

For `phase = :liquid`, returns the pre-computed `precipitation_rate` 2D field
from `model.microphysical_fields`, which represents the surface precipitation rate [m/s].

For `phase = :ice`, returns `nothing` (Kessler is a warm-rain scheme).
"""
precipitation_rate(model, ::KM, ::Val{:liquid}) = model.microphysical_fields.precipitation_rate
precipitation_rate(model, ::KM, ::Val{:ice}) = nothing

"""
$(TYPEDSIGNATURES)

Return the surface precipitation flux for the Kessler scheme.

The surface precipitation flux is `|wʳ| * ρqʳ` at k=1 (bottom face), representing
the rate at which rain mass leaves the domain through the bottom boundary.

Units: kg/m²/s (positive = downward, out of domain)
"""
function surface_precipitation_flux(model, ::KM)
    grid = model.grid
    μ = model.microphysical_fields
    ρᵣ = model.formulation.reference_state.density
    kernel = KesslerSurfacePrecipitationFluxKernel(μ.wʳ, μ.ρqʳ, ρᵣ)
    op = KernelFunctionOperation{Center, Center, ONothing}(kernel, grid)
    return Field(op)
end

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

struct KesslerSurfacePrecipitationFluxKernel{W, R, D}
    terminal_velocity :: W
    rain_density :: R
    reference_density :: D
end

Adapt.adapt_structure(to, k::KesslerSurfacePrecipitationFluxKernel) =
    KesslerSurfacePrecipitationFluxKernel(adapt(to, k.terminal_velocity),
                                           adapt(to, k.rain_density),
                                           adapt(to, k.reference_density))

@inline function (kernel::KesslerSurfacePrecipitationFluxKernel)(i, j, k_idx, grid)
    # Flux at bottom face (k=1), ignore k_idx since this is a 2D field
    # wʳ < 0 (downward), so -wʳ * ρqʳ > 0 represents flux out of domain
    @inbounds wʳ = kernel.terminal_velocity[i, j, 1]
    @inbounds ρqʳ = kernel.rain_density[i, j, 1]
    
    # Return positive flux for rain leaving domain (downward)
    return -wʳ * ρqʳ
end
