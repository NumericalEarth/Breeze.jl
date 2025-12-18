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

using Oceananigans: Oceananigans, CenterField
using Oceananigans.Operators: Δzᶜᶜᶜ
using DocStringExtensions: TYPEDSIGNATURES

import ..AtmosphereModels:
    prognostic_field_names,
    materialize_microphysical_fields,
    microphysical_velocities,
    compute_moisture_fractions,
    microphysical_tendency,
    update_microphysical_fields!

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

function materialize_microphysical_fields(::KM, grid, boundary_conditions)
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

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, Cₖ, Eₖ, Aₖ, Kₖ, Eʳ)
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
        Eʳ_val = rain_evaporation_rate(ρ, rᵛ, rʳ, rᵛ⁺)
        
        # Store rates for use in microphysical_tendency
        μ.Cₖ[i, j, k] = Cₖ_val
        μ.Eₖ[i, j, k] = Eₖ_val
        μ.Aₖ[i, j, k] = Aₖ_val
        μ.Kₖ[i, j, k] = Kₖ_val
        μ.Eʳ[i, j, k] = Eʳ_val
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
@inline maybe_adjust_thermodynamic_state(𝒰, ::KM, μ, qᵗ, constants) = 𝒰

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

Returns `nothing` for all fields - sedimentation is handled internally
via the sedimentation_tendency function in the rain tendency calculation.
"""
@inline microphysical_velocities(::KM, name::Val{:ρqʳ}) = nothing
@inline microphysical_velocities(::KM, ::Val{:ρqᶜˡ}) = nothing
@inline microphysical_velocities(::KM, name) = nothing

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
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqᶜˡ}, formulation, μ, 𝒰, constants)
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
and cached in the microphysical fields. Sedimentation is included using upstream differencing.

```math
\\frac{∂(ρqʳ)}{∂t} = ρ \\cdot (1 - qᵗ) \\cdot (Aₖ + Kₖ - Eʳ + S)
```

where S is the sedimentation term.
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqʳ}, formulation, μ, 𝒰, constants)
    # Get thermodynamic quantities
    ρ = density(𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    
    # Reference density profile for sedimentation (allows access to k+1 in local kernel)
    ρᵣ = formulation.reference_state.density
    
    # Get cached rates (computed in update_microphysical_fields!)
    @inbounds begin
        Aₖ = μ.Aₖ[i, j, k]
        Kₖ = μ.Kₖ[i, j, k]
        Eʳ = μ.Eʳ[i, j, k]
    end
    
    # Sedimentation term (in mixing ratio space)
    sed = sedimentation_tendency(i, j, k, grid, ρᵣ, μ)
    
    # Tendency in mixing ratio space: drʳ/dt = Aₖ + Kₖ - Eʳ + sed
    drʳdt = Aₖ + Kₖ - Eʳ + sed
    
    # Convert to mass fraction tendency
    dqʳdt = mixing_ratio_to_mass_fraction(drʳdt, qᵗ)
    
    return ρ * dqʳdt
end

# Default: no tendency for other variables
@inline microphysical_tendency(i, j, k, grid, ::KM, name, formulation, μ, 𝒰, constants) = zero(grid)

#####
##### Potential temperature tendency
#####

"""
$(TYPEDSIGNATURES)

Compute the tendency for liquid-ice potential temperature density (ρθˡⁱ).

In Breeze, the potential temperature is liquid-ice potential temperature (θˡⁱ), defined such that
temperature is computed as:

```math
T = Π θˡⁱ + (ℒˡ qˡ + ℒⁱ qⁱ) / cₚ
```

**Phase change processes** (condensation, evaporation) conserve θˡⁱ by design.

**Sedimentation** requires a θˡⁱ adjustment to maintain constant temperature when rain
enters or leaves a cell. When rain sediments, qˡ changes locally but T should not change
(no phase change during sedimentation). From the definition:

```math
\\frac{∂θˡⁱ}{∂t}\\bigg|_{sed} = -\\frac{ℒˡ}{cₚ Π} \\frac{∂qʳ}{∂t}\\bigg|_{sed}
```

This ensures:
- When rain enters a cell (∂qʳ/∂t > 0): θˡⁱ decreases to maintain T
- When rain leaves a cell (∂qʳ/∂t < 0): θˡⁱ increases to maintain T
- Rain falling out at the surface warms the air (removes "cold" liquid)
"""
@inline function microphysical_tendency(i, j, k, grid, ::KM, ::Val{:ρθ}, formulation, μ, 𝒰, constants)
    # Get thermodynamic quantities
    ρ = density(𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    T = temperature(𝒰, constants)
    
    # Reference density profile for sedimentation (allows access to k+1 in local kernel)
    ρᵣ = formulation.reference_state.density
    
    # Sedimentation tendency for rain (in mixing ratio space)
    sed = sedimentation_tendency(i, j, k, grid, ρᵣ, μ)
    
    # Convert to mass fraction tendency
    dqʳdt_sed = mixing_ratio_to_mass_fraction(sed, qᵗ)
    
    # Compute Exner function Π
    Π = exner_function(𝒰, constants)
    
    # Get latent heat and heat capacity
    q = 𝒰.moisture_mass_fractions
    ℒˡ = liquid_latent_heat(T, constants)
    cₚ = mixture_heat_capacity(q, constants)
    
    # θˡⁱ tendency from sedimentation: ∂θˡⁱ/∂t = -(ℒˡ / (cₚ Π)) * ∂qʳ/∂t|_sed
    dθdt_sed = -ℒˡ / (cₚ * Π) * dqʳdt_sed
    
    return ρ * dθdt_sed
end
