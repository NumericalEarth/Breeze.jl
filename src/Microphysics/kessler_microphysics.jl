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
using DocStringExtensions: TYPEDSIGNATURES

using ..Thermodynamics:
    MoistureMassFractions,
    PlanarLiquidSurface,
    saturation_specific_humidity,
    temperature,
    density,
    liquid_latent_heat,
    mixture_heat_capacity,
    total_specific_moisture

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
(ρᵣ[1,1,1]) rather than being stored as a parameter.
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

The terminal velocity is given by:

```math
wₜ = 36.34 (ρ rʳ)^{0.1346} (ρ / ρ₀)^{-1/2}
```

where ρ is air density, rʳ is rain mixing ratio, and ρ₀ is reference density
(obtained from Breeze's reference state at the surface, ρᵣ[1,1,1]).

Note: The original formula gives velocity in cm s⁻¹ with coefficient 3634.
Here we use 36.34 m s⁻¹ for SI units.
"""
@inline function rain_terminal_velocity(ρ, rʳ, ρ₀)
    FT = typeof(ρ)
    ρrʳ = ρ * max(zero(FT), rʳ)
    
    # Coefficient 36.34 m/s (converted from 3634 cm/s)
    wₜ = convert(FT, 36.34) * ρrʳ^convert(FT, 0.1346) * (ρ / ρ₀)^(-convert(FT, 0.5))
    
    return wₜ
end

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities for the Kessler scheme.

Currently returns `nothing` as sedimentation is not yet implemented via the velocity interface.
The terminal velocity formula is provided via `rain_terminal_velocity(ρ, rʳ, ρ₀)` for future
implementation, where ρ₀ should be obtained from the model's reference state density at
the surface (ρᵣ[1,1,1]).
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

```math
D = 1 + \\frac{rᵛ⁺ \\cdot 4093 \\cdot L}{cₚ (T - 36)^2}
```
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
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqᶜˡ}, μ, 𝒰, constants)
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

```math
\\frac{∂(ρqʳ)}{∂t} = ρ \\cdot (1 - qᵗ) \\cdot (Aₖ + Kₖ - Eʳ)
```

Note: Sedimentation is not yet implemented.
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqʳ}, μ, 𝒰, constants)
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
    drʳdt = Aₖ + Kₖ - Eʳ
    
    # Convert to mass fraction tendency
    dqʳdt = mixing_ratio_to_mass_fraction(drʳdt, qᵗ)
    
    return ρ * dqʳdt
end

# Default: no tendency for other variables
@inline microphysical_tendency(i, j, k, grid, ::KM, name, μ, 𝒰, constants) = zero(grid)

#####
##### Potential temperature tendency
#####

"""
$(TYPEDSIGNATURES)

Compute the tendency for liquid-ice potential temperature density (ρθ).

In Breeze, the potential temperature is liquid-ice potential temperature (θˡⁱ), defined such that
temperature is computed as:

```math
T = Π θˡⁱ + (ℒˡ qˡ + ℒⁱ qⁱ) / cₚ
```

where qˡ includes ALL liquid water (both cloud and rain). Since rain is counted as liquid,
all processes in the Kessler warm-rain scheme conserve θˡⁱ:

- **Condensation** (vapor → cloud liquid): θˡⁱ conserved
- **Cloud evaporation** (cloud liquid → vapor): θˡⁱ conserved
- **Autoconversion** (cloud → rain): θˡⁱ conserved (liquid → liquid)
- **Accretion** (cloud → rain): θˡⁱ conserved (liquid → liquid)
- **Rain evaporation** (rain → vapor): θˡⁱ conserved (liquid → vapor)

Therefore, the Kessler scheme has zero tendency for θˡⁱ.
"""
@inline microphysical_tendency(i, j, k, grid, ::KM, ::Val{:ρθ}, μ, 𝒰, constants) = zero(grid)
