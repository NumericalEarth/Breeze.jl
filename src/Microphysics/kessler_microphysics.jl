"""
Kessler warm-rain bulk microphysics scheme.

A "warm-rain" (Kessler-type) bulk microphysics scheme with water vapor, cloud liquid, and rain.
All hydrometeors are represented as mixing ratios (kg kg⁻¹).

Prognostic variables:
- qᵛ: water vapor mixing ratio
- qˡ: cloud liquid water mixing ratio
- qʳ: rain water mixing ratio

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
    mixture_heat_capacity

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
- `reference_density`: Reference density for terminal velocity calculation, ρ₀ [kg m⁻³]. Default: 1.0 kg m⁻³
"""
struct KesslerMicrophysics{FT}
    autoconversion_rate :: FT       # k₁ [s⁻¹]
    autoconversion_threshold :: FT  # a [kg kg⁻¹]
    accretion_rate :: FT            # k₂ [s⁻¹]
    reference_density :: FT         # ρ₀ [kg m⁻³]
end

Base.summary(::KesslerMicrophysics) = "KesslerMicrophysics"

function Base.show(io::IO, km::KesslerMicrophysics{FT}) where FT
    print(io, "KesslerMicrophysics{$FT}:\n",
              "├── autoconversion_rate: ", km.autoconversion_rate, " s⁻¹\n",
              "├── autoconversion_threshold: ", km.autoconversion_threshold, " kg kg⁻¹\n",
              "├── accretion_rate: ", km.accretion_rate, " s⁻¹\n",
              "└── reference_density: ", km.reference_density, " kg m⁻³")
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
- `reference_density`: Reference density ρ₀ [kg m⁻³]. Default: 1.0 kg m⁻³
"""
function KesslerMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                             autoconversion_rate = 0.001,
                             autoconversion_threshold = 0.001,
                             accretion_rate = 2.2,
                             reference_density = 1.0)

    return KesslerMicrophysics{FT}(convert(FT, autoconversion_rate),
                                   convert(FT, autoconversion_threshold),
                                   convert(FT, accretion_rate),
                                   convert(FT, reference_density))
end

const KM = KesslerMicrophysics

#####
##### Microphysics interface implementation
#####

prognostic_field_names(::KM) = (:ρqᵛ, :ρqˡ, :ρqʳ)

function materialize_microphysical_fields(::KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted mixing ratios)
    ρqᵛ = CenterField(grid; boundary_conditions=boundary_conditions.ρqᵛ)
    ρqˡ = CenterField(grid; boundary_conditions=boundary_conditions.ρqˡ)
    ρqʳ = CenterField(grid; boundary_conditions=boundary_conditions.ρqʳ)

    # Diagnostic fields (mixing ratios)
    qᵛ = CenterField(grid)
    qˡ = CenterField(grid)
    qʳ = CenterField(grid)

    return (; ρqᵛ, ρqˡ, ρqʳ, qᵛ, qˡ, qʳ)
end

@inline function update_microphysical_fields!(μ, ::KM, i, j, k, grid, ρ, 𝒰, constants)
    @inbounds begin
        # Update diagnostic mixing ratios from thermodynamic state
        μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
        μ.qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
        # Rain mixing ratio from prognostic field
        μ.qʳ[i, j, k] = μ.ρqʳ[i, j, k] / ρ
    end
    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, ::KM, ρ, qᵗ, μ)
    @inbounds begin
        qᵛ = μ.ρqᵛ[i, j, k] / ρ
        qˡ = μ.ρqˡ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
    end
    # Rain is counted as liquid in the liquid-ice potential temperature definition
    # Total liquid = cloud liquid + rain
    return MoistureMassFractions(qᵛ, qˡ + qʳ)
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
wₜ = 36.34 (ρ qʳ)^{0.1346} (ρ / ρ₀)^{-1/2}
```

where ρ is air density, qʳ is rain mixing ratio, and ρ₀ is reference density.

Note: The original formula gives velocity in cm s⁻¹ with coefficient 3634.
Here we use 36.34 m s⁻¹ for SI units.
"""
@inline function rain_terminal_velocity(ρ, qʳ, km::KM)
    FT = typeof(ρ)
    ρ₀ = km.reference_density
    ρqʳ = ρ * max(zero(FT), qʳ)
    
    # Coefficient 36.34 m/s (converted from 3634 cm/s)
    # wₜ = 36.34 * (ρqʳ)^0.1346 * (ρ/ρ₀)^(-0.5)
    wₜ = convert(FT, 36.34) * ρqʳ^convert(FT, 0.1346) * (ρ / ρ₀)^(-convert(FT, 0.5))
    
    return wₜ
end

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities for the Kessler scheme.

Currently returns `nothing` as sedimentation is not yet implemented via the velocity interface.
The terminal velocity formula is provided via `rain_terminal_velocity` for future implementation
or diagnostic purposes.
"""
@inline microphysical_velocities(::KM, name::Val{:ρqʳ}) = nothing
@inline microphysical_velocities(::KM, ::Val{:ρqᵛ}) = nothing
@inline microphysical_velocities(::KM, ::Val{:ρqˡ}) = nothing
@inline microphysical_velocities(::KM, name) = nothing

#####
##### Source term calculations
#####

"""
$(TYPEDSIGNATURES)

Compute the denominator D for condensation/evaporation rate.

```math
D = 1 + \\frac{qᵛ⁺ \\cdot 4093 \\cdot L}{cₚ (T - 36)^2}
```
"""
@inline function condensation_denominator(T, qᵛ⁺, L, cₚ)
    FT = typeof(T)
    return one(FT) + qᵛ⁺ * convert(FT, 4093) * L / (cₚ * (T - convert(FT, 36))^2)
end

"""
$(TYPEDSIGNATURES)

Compute condensation rate Cₖ [kg kg⁻¹ s⁻¹].

If supersaturated (qᵛ > qᵛ⁺): Cₖ = (qᵛ - qᵛ⁺) / D
Otherwise: Cₖ = 0
"""
@inline function condensation_rate(qᵛ, qᵛ⁺, D)
    FT = typeof(qᵛ)
    return qᵛ > qᵛ⁺ ? (qᵛ - qᵛ⁺) / D : zero(FT)
end

"""
$(TYPEDSIGNATURES)

Compute cloud evaporation rate Eₖ [kg kg⁻¹ s⁻¹].

If subsaturated (qᵛ < qᵛ⁺): Eₖ = min(qˡ, (qᵛ⁺ - qᵛ) / D)
Otherwise: Eₖ = 0

The evaporation is limited by available cloud water.
"""
@inline function cloud_evaporation_rate(qᵛ, qˡ, qᵛ⁺, D)
    FT = typeof(qᵛ)
    if qᵛ < qᵛ⁺
        # Limit evaporation by available cloud water
        return min(qˡ, (qᵛ⁺ - qᵛ) / D)
    else
        return zero(FT)
    end
end

"""
$(TYPEDSIGNATURES)

Compute autoconversion rate Aₖ [kg kg⁻¹ s⁻¹].

```math
Aₖ = \\max(0, k₁ (qˡ - a))
```

where k₁ is the autoconversion rate and a is the threshold.
"""
@inline function autoconversion_rate(qˡ, km::KM)
    FT = typeof(qˡ)
    k₁ = km.autoconversion_rate
    a = km.autoconversion_threshold
    return max(zero(FT), k₁ * (qˡ - a))
end

"""
$(TYPEDSIGNATURES)

Compute accretion rate Kₖ [kg kg⁻¹ s⁻¹].

```math
Kₖ = k₂ qˡ qʳ^{0.875}
```

where k₂ is the accretion rate constant.
"""
@inline function accretion_rate(qˡ, qʳ, km::KM)
    FT = typeof(qˡ)
    k₂ = km.accretion_rate
    qʳ_safe = max(zero(FT), qʳ)
    return k₂ * qˡ * qʳ_safe^convert(FT, 0.875)
end

"""
$(TYPEDSIGNATURES)

Compute rain evaporation rate Eʳ [kg kg⁻¹ s⁻¹].

```math
Eʳ = \\frac{(1 - qᵛ/qᵛ⁺) C (ρ qʳ)^{0.525}}{ρ (5.4 \\times 10^5 + 2.55 \\times 10^6 / (ρ qᵛ⁺))}
```

where the ventilation factor is:
```math
C = 1.6 + 124.9 (ρ qʳ)^{0.2046}
```

Note: Rain evaporation only occurs when subsaturated (qᵛ < qᵛ⁺).
"""
@inline function rain_evaporation_rate(ρ, qᵛ, qʳ, qᵛ⁺)
    FT = typeof(ρ)
    
    # No evaporation if saturated or supersaturated
    qᵛ >= qᵛ⁺ && return zero(FT)
    
    # No evaporation if no rain
    qʳ <= zero(FT) && return zero(FT)
    
    ρqʳ = ρ * qʳ
    ρqᵛ⁺ = ρ * qᵛ⁺
    
    # Ventilation factor
    C = convert(FT, 1.6) + convert(FT, 124.9) * ρqʳ^convert(FT, 0.2046)
    
    # Subsaturation factor
    subsaturation = one(FT) - qᵛ / qᵛ⁺
    
    # Denominator
    denom = convert(FT, 5.4e5) + convert(FT, 2.55e6) / ρqᵛ⁺
    
    # Rain evaporation rate (per unit mass)
    Eʳ = subsaturation * C * ρqʳ^convert(FT, 0.525) / (ρ * denom)
    
    # Limit by available rain
    return min(Eʳ, qʳ)
end

#####
##### Microphysical tendencies
#####

"""
$(TYPEDSIGNATURES)

Compute the tendency for water vapor density (ρqᵛ).

```math
\\frac{∂(ρqᵛ)}{∂t} = ρ(-Cₖ + Eₖ + Eʳ)
```
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqᵛ}, μ, 𝒰, constants)
    FT = eltype(grid)
    
    # Get thermodynamic quantities
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    
    # Get mixing ratios
    @inbounds begin
        qᵛ = μ.qᵛ[i, j, k]
        qˡ = μ.qˡ[i, j, k]
        qʳ = μ.qʳ[i, j, k]
    end
    
    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    
    # Latent heat and heat capacity
    L = liquid_latent_heat(T, constants)
    q = MoistureMassFractions(qᵛ, qˡ)
    cₚ = mixture_heat_capacity(q, constants)
    
    # Condensation/evaporation
    D = condensation_denominator(T, qᵛ⁺, L, cₚ)
    Cₖ = condensation_rate(qᵛ, qᵛ⁺, D)
    Eₖ = cloud_evaporation_rate(qᵛ, qˡ, qᵛ⁺, D)
    
    # Rain evaporation
    Eʳ = rain_evaporation_rate(ρ, qᵛ, qʳ, qᵛ⁺)
    
    # dqᵛ/dt = -Cₖ + Eₖ + Eʳ
    return ρ * (-Cₖ + Eₖ + Eʳ)
end

"""
$(TYPEDSIGNATURES)

Compute the tendency for cloud liquid density (ρqˡ).

```math
\\frac{∂(ρqˡ)}{∂t} = ρ(Cₖ - Eₖ - Aₖ - Kₖ)
```
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqˡ}, μ, 𝒰, constants)
    FT = eltype(grid)
    
    # Get thermodynamic quantities
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    
    # Get mixing ratios
    @inbounds begin
        qᵛ = μ.qᵛ[i, j, k]
        qˡ = μ.qˡ[i, j, k]
        qʳ = μ.qʳ[i, j, k]
    end
    
    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    
    # Latent heat and heat capacity
    L = liquid_latent_heat(T, constants)
    q = MoistureMassFractions(qᵛ, qˡ)
    cₚ = mixture_heat_capacity(q, constants)
    
    # Condensation/evaporation
    D = condensation_denominator(T, qᵛ⁺, L, cₚ)
    Cₖ = condensation_rate(qᵛ, qᵛ⁺, D)
    Eₖ = cloud_evaporation_rate(qᵛ, qˡ, qᵛ⁺, D)
    
    # Autoconversion and accretion
    Aₖ = autoconversion_rate(qˡ, km)
    Kₖ = accretion_rate(qˡ, qʳ, km)
    
    # dqˡ/dt = Cₖ - Eₖ - Aₖ - Kₖ
    return ρ * (Cₖ - Eₖ - Aₖ - Kₖ)
end

"""
$(TYPEDSIGNATURES)

Compute the tendency for rain density (ρqʳ).

Note: This tendency does NOT include sedimentation, which is handled separately
through `microphysical_velocities`.

```math
\\frac{∂(ρqʳ)}{∂t} = ρ(Aₖ + Kₖ - Eʳ)
```
"""
@inline function microphysical_tendency(i, j, k, grid, km::KM, ::Val{:ρqʳ}, μ, 𝒰, constants)
    FT = eltype(grid)
    
    # Get thermodynamic quantities
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    
    # Get mixing ratios
    @inbounds begin
        qᵛ = μ.qᵛ[i, j, k]
        qˡ = μ.qˡ[i, j, k]
        qʳ = μ.qʳ[i, j, k]
    end
    
    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    
    # Autoconversion and accretion
    Aₖ = autoconversion_rate(qˡ, km)
    Kₖ = accretion_rate(qˡ, qʳ, km)
    
    # Rain evaporation
    Eʳ = rain_evaporation_rate(ρ, qᵛ, qʳ, qᵛ⁺)
    
    # dqʳ/dt = Aₖ + Kₖ - Eʳ
    # Sedimentation is handled separately via microphysical_velocities
    return ρ * (Aₖ + Kₖ - Eʳ)
end

# Default: no tendency for other variables
@inline microphysical_tendency(i, j, k, grid, ::KM, name, μ, 𝒰, constants) = zero(grid)

#####
##### Potential temperature tendency from phase changes
#####

"""
$(TYPEDSIGNATURES)

Compute the tendency for liquid-ice potential temperature density (ρθ) due to microphysical processes.

In Breeze, the potential temperature is liquid-ice potential temperature (θₗᵢ), defined such that
temperature is computed as:

```math
T = Π θₗᵢ + (ℒˡ qˡ + ℒⁱ qⁱ) / cₚ
```

where qˡ includes ALL liquid water (both cloud and rain). Since rain is counted as liquid,
all processes in the Kessler warm-rain scheme conserve θₗᵢ:

- **Condensation** (vapor → cloud liquid): θₗᵢ conserved
- **Cloud evaporation** (cloud liquid → vapor): θₗᵢ conserved
- **Autoconversion** (cloud → rain): θₗᵢ conserved (liquid → liquid)
- **Accretion** (cloud → rain): θₗᵢ conserved (liquid → liquid)
- **Rain evaporation** (rain → vapor): θₗᵢ conserved (liquid → vapor)

Therefore, the Kessler scheme has zero tendency for θₗᵢ.
"""
@inline microphysical_tendency(i, j, k, grid, ::KM, ::Val{:ρθ}, μ, 𝒰, constants) = zero(grid)
