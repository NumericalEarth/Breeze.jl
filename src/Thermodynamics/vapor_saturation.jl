#####
##### Surface types for saturation vapor pressure calculations
#####

struct PlanarLiquidSurface end
struct PlanarIceSurface end

"""
$(TYPEDEF)

Return `PlanarMixedPhaseSurface` for computing the saturation vapor pressure over
a surface composed of a mixture of liquid and ice, with a given `liquid_fraction`.
"""
struct PlanarMixedPhaseSurface{FT}
    liquid_fraction :: FT
end

#####
##### Saturation specific humidity
#####

"""
$(TYPEDSIGNATURES)

Compute the saturation specific humidity for a gas at temperature `T`, total
density `ρ`, `constants`dynamics, and over `surface` via:

```math
qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) ,
```

where ``pᵛ⁺`` is the [`saturation_vapor_pressure`](@ref) over `surface`, ``ρ`` is total density,
and ``Rᵛ`` is the specific gas constant for water vapor.

# Examples

First we compute the saturation specific humidity over a liquid surface:

```jldoctest saturation
using Breeze
using Breeze.Thermodynamics: PlanarLiquidSurface, PlanarIceSurface, PlanarMixedPhaseSurface

constants = ThermodynamicConstants()
T = 288.0 # Room temperature (K)
p = 101325.0 # Mean sea-level pressure
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
q = zero(Breeze.Thermodynamics.MoistureMassFractions{Float64})
ρ = Breeze.Thermodynamics.density(T, p, q, constants)
qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

# output
0.010359995391195264
```

Note, this is slightly smaller than the saturation specific humidity over an ice surface:

```jldoctest saturation
julia> qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
0.011945100768555072
```

If a medium contains a mixture of 40% water and 60% ice that has (somehow) acquired
thermodynamic equilibrium, we can compute the saturation specific humidity
over the mixed phase surface,

```jldoctest saturation
mixed_surface = PlanarMixedPhaseSurface(0.4)
qᵛ⁺ᵐ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, mixed_surface)

# output
0.01128386068542303
```
"""
@inline function saturation_specific_humidity(T, ρ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵛ = vapor_gas_constant(constants)
    return pᵛ⁺ / (ρ * Rᵛ * T)
end

"""
$(TYPEDSIGNATURES)

Compute the supersaturation ``𝒮 = pᵛ/pᵛ⁺ - 1`` over a given `surface`.

- ``𝒮 < 0`` indicates subsaturation (evaporation conditions)
- ``𝒮 = 0`` indicates saturation (equilibrium)
- ``𝒮 > 0`` indicates supersaturation (condensation conditions)

# Arguments
- `T`: Temperature
- `ρ`: Total air density  
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `constants`: `ThermodynamicConstants`
- `surface`: Surface type (e.g., `PlanarLiquidSurface()`, `PlanarIceSurface()`)
"""
@inline function supersaturation(T, ρ, q::MoistureMassFractions, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    pᵛ = vapor_pressure(ρ, T, q.vapor, constants)
    return pᵛ / pᵛ⁺ - 1
end

#####
##### Phase equilibrium types
#####

abstract type AbstractPhaseEquilibrium end

"""
    WarmPhaseEquilibrium()

Represents a warm-phase equilibrium where only liquid water condensate is considered.
The equilibrated surface is always a planar liquid surface.
"""
struct WarmPhaseEquilibrium <: AbstractPhaseEquilibrium end

"""
$(TYPEDSIGNATURES)

Return the appropriate surface type for computing saturation vapor pressure
given the phase `equilibrium` model and temperature `T`.
"""
@inline equilibrated_surface(::WarmPhaseEquilibrium, T) = PlanarLiquidSurface()

"""
    MixedPhaseEquilibrium(; freezing_temperature=273.15, homogeneous_ice_nucleation_temperature=233.15)

Represents a mixed-phase equilibrium where both liquid and ice condensates are considered.
The liquid fraction varies linearly with temperature between the freezing temperature
and the homogeneous ice nucleation temperature.
"""
struct MixedPhaseEquilibrium{FT} <: AbstractPhaseEquilibrium
    freezing_temperature :: FT
    homogeneous_ice_nucleation_temperature :: FT
end

function MixedPhaseEquilibrium(FT::DataType = Oceananigans.defaults.FloatType;
                               freezing_temperature = 273.15,
                               homogeneous_ice_nucleation_temperature = 233.15)

    if freezing_temperature < homogeneous_ice_nucleation_temperature
        throw(ArgumentError("`freezing_temperature` must be greater than `homogeneous_ice_nucleation_temperature`"))
    end

    freezing_temperature = convert(FT, freezing_temperature)
    homogeneous_ice_nucleation_temperature = convert(FT, homogeneous_ice_nucleation_temperature)
    return MixedPhaseEquilibrium(freezing_temperature, homogeneous_ice_nucleation_temperature)
end

@inline function equilibrated_surface(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    T′ = clamp(T, Tʰ, Tᶠ)
    λ = (T′ - Tʰ) / (Tᶠ - Tʰ)
    return PlanarMixedPhaseSurface(λ)
end

#####
##### Saturation specific humidity with phase equilibrium
#####

@inline function saturation_specific_humidity(T, ρ, constants, equilibrium::AbstractPhaseEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    return saturation_specific_humidity(T, ρ, constants, surface)
end

"""
$(TYPEDSIGNATURES)

Compute the *equilibrium saturation specific humidity* ``qᵛ⁺`` for air at
temperature `T`, reference pressure `pᵣ`, and total specific moisture `qᵗ`,
over a given `surface`.

This function returns the correct saturation specific humidity in both saturated
and unsaturated conditions:

- In **unsaturated conditions** (``qᵗ < qᵛ⁺``), all moisture is vapor and the
  density is computed assuming ``qᵛ = qᵗ``.

- In **saturated conditions** (``qᵗ ≥ qᵛ⁺``), the vapor specific humidity equals
  the saturation value and the density is computed assuming ``qᵛ = qᵛ⁺``.

The saturated formula corresponds to equation (37) in [Pressel et al. (2015)](@cite Pressel2015).
"""
@inline function equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ϵᵈᵛ = Rᵈ / Rᵛ
    qᵛ⁺₁ = ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)

    # In unsaturated conditions, all moisture is vapor (qᵛ = qᵗ)
    # Compute density using mixture gas constant for this case
    Rᵐ = Rᵈ * (1 - qᵗ) + Rᵛ * qᵗ
    ρ = pᵣ / (Rᵐ * T)
    qᵛ⁺₀ = pᵛ⁺ / (ρ * Rᵛ * T)

    return ifelse(qᵗ >= qᵛ⁺₀, qᵛ⁺₁, qᵛ⁺₀)
end

"""
$(TYPEDSIGNATURES)

Compute the *saturation specific humidity* ``qᵛ⁺`` for use in saturation adjustment,
assuming **saturated conditions** where condensate is present.

This function always uses the saturated formula (equation 37 in paper by [Pressel et al. 2015](@cite Pressel2015)):

```math
qᵛ⁺ = ϵᵈᵛ (1 - qᵗ) \\frac{pᵛ⁺}{pᵣ - pᵛ⁺}
```

where ``ϵᵈᵛ = Rᵈ / Rᵛ ≈ 0.622``.

Unlike [`equilibrium_saturation_specific_humidity`](@ref), this function does not
check whether the air is actually saturated. It is intended for use within the
saturation adjustment iteration where we assume saturated conditions throughout.
"""
@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

"""
$(TYPEDSIGNATURES)

Compute the equilibrium saturation specific humidity using a phase `equilibrium`
model to determine the condensation surface based on temperature `T`.
"""
@inline function equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium::AbstractPhaseEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    return equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, surface)
end

"""
$(TYPEDSIGNATURES)

Compute the adjustment saturation specific humidity using a phase `equilibrium`
model to determine the condensation surface based on temperature `T`.
"""
@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium::AbstractPhaseEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    return adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, constants, surface)
end
