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

@inline specific_heat_difference(constants, ::PlanarLiquidSurface) = specific_heat_difference(constants, constants.liquid)
@inline specific_heat_difference(constants, ::PlanarIceSurface) = specific_heat_difference(constants, constants.ice)
@inline absolute_zero_latent_heat(constants, ::PlanarLiquidSurface) = absolute_zero_latent_heat(constants, constants.liquid)
@inline absolute_zero_latent_heat(constants, ::PlanarIceSurface) = absolute_zero_latent_heat(constants, constants.ice)

@inline function specific_heat_difference(constants, surf::PlanarMixedPhaseSurface)
    Δcˡ = specific_heat_difference(constants, constants.liquid)
    Δcⁱ = specific_heat_difference(constants, constants.ice)
    λ = surf.liquid_fraction
    return λ * Δcˡ + (1 - λ) * Δcⁱ
end

@inline function absolute_zero_latent_heat(constants, surf::PlanarMixedPhaseSurface)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    λ = surf.liquid_fraction
    return λ * ℒˡ₀ + (1 - λ) * ℒⁱ₀
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
    pᵛ = vapor_pressure(T, ρ, q.vapor, constants)
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
    equilibrated_surface(phase_equilibrium::AbstractPhaseEquilibrium, T)

Return the appropriate surface type for computing saturation vapor pressure
given the phase `equilibrium` model and temperature `T`.
"""
function equilibrated_surface end

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

Compute the *equilibrium saturation specific humidity* ``qᵛ⁺`` for air at temperature `T`, reference pressure `p`, 
and total specific moisture `qᵗ`, over a given `surface`. The function returns the correct saturation specific humidity 
in both saturated and unsaturated conditions:

- In **saturated conditions** (``qᵗ ≥ qᵛ⁺``), condensate is present and ``qᵛ = qᵛ⁺``. The dry-air mass fraction 
  is fixed by ``qᵗ`` (since ``qᵈ = 1 - qᵗ``), and the equation of state can be solved in closed form or ``qᵛ⁺``, 
  yielding equation (37) of [Pressel et al. (2015)](@cite Pressel2015):

  ```math
  qᵛ⁺ = \\frac{ϵᵈᵛ \\, (1 - qᵗ) \\, pᵛ⁺(T)}{p - pᵛ⁺(T)} ,
  ```

  where ``ϵᵈᵛ ≡ Rᵈ / Rᵛ ≈ 0.622``.

- In **unsaturated conditions** (``qᵗ < qᵛ⁺``), all moisture is vapor and ``qᵛ = qᵗ``. The density is then 
  ``ρ = p / (Rᵐ T)`` with mixture gas constant ``Rᵐ = (1 - qᵗ) Rᵈ + qᵗ Rᵛ``, and

  ```math
  qᵛ⁺ = \\frac{pᵛ⁺(T)}{ρ \\, Rᵛ \\, T} .
  ```

The function selects the branch by computing the unsaturated ``qᵛ⁺`` and comparing with `qᵗ`. See also
[`saturation_total_specific_moisture`](@ref Breeze.AtmosphereModels.Diagnostics.saturation_total_specific_moisture),
which is the special case ``qᵗ = qᵛ⁺``, and the [Atmosphere Thermodynamics](@ref Thermodynamics-section) section of
the documentation for a derivation.
"""
@inline function equilibrium_saturation_specific_humidity(T, p, qᵗ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ϵᵈᵛ = Rᵈ / Rᵛ
    qᵛ⁺₁ = ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (p - pᵛ⁺)

    # In unsaturated conditions, all moisture is vapor (qᵛ = qᵗ)
    # Compute density using mixture gas constant for this case
    Rᵐ = Rᵈ * (1 - qᵗ) + Rᵛ * qᵗ
    ρ = p / (Rᵐ * T)
    qᵛ⁺₀ = pᵛ⁺ / (ρ * Rᵛ * T)

    return ifelse(qᵗ ≥ qᵛ⁺₀, qᵛ⁺₁, qᵛ⁺₀)
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

#####
##### Dewpoint temperature
#####

"""
$(TYPEDSIGNATURES)

Compute the dewpoint temperature ``T⁺`` given the vapor pressure `pᵛ`,
actual temperature `T`, thermodynamic `constants`, and condensation `surface`.

The dewpoint temperature is defined as the temperature at which the saturation
vapor pressure equals the actual vapor pressure:

```math
pᵛ⁺(T⁺) = pᵛ
```

This implicit equation is solved using secant iteration, which works with any
saturation vapor pressure formulation.

If the air is saturated or supersaturated (``pᵛ ≥ pᵛ⁺(T)``), the dewpoint
equals the actual temperature and `T` is returned.

# Arguments
- `pᵛ`: Vapor pressure (Pa)
- `T`: Actual temperature (K), used as upper bound and first guess
- `constants`: `ThermodynamicConstants`
- `surface`: Surface type for saturation vapor pressure calculation

# Keyword arguments
- `tolerance`: Relative tolerance for convergence (default: 1e-4)
- `maxiter`: Maximum number of iterations (default: 10)
"""
@inline function dewpoint_temperature(pᵛ, T, constants, surface;
                                      tolerance = 1e-4,
                                      maxiter = 10)
    # First guess: current temperature
    T⁺₁ = T
    pᵛ⁺₁ = saturation_vapor_pressure(T⁺₁, constants, surface)
    r₁ = pᵛ⁺₁ - pᵛ

    # If saturated or supersaturated, dewpoint equals temperature
    r₁ ≤ 0 && return T

    # Second guess: lower temperature based on relative humidity
    ℋ = pᵛ / pᵛ⁺₁  # relative humidity
    T⁺₂ = T - (1 - ℋ) * 20  # heuristic initial step
    pᵛ⁺₂ = saturation_vapor_pressure(T⁺₂, constants, surface)
    r₂ = pᵛ⁺₂ - pᵛ

    # Secant iteration
    iter = 0
    while abs(r₂) > tolerance * pᵛ && iter < maxiter
        ΔTΔr = (T⁺₂ - T⁺₁) / (r₂ - r₁)
        r₁, T⁺₁ = r₂, T⁺₂
        T⁺₂ -= r₂ * ΔTΔr
        pᵛ⁺₂ = saturation_vapor_pressure(T⁺₂, constants, surface)
        r₂ = pᵛ⁺₂ - pᵛ
        iter += 1
    end

    return T⁺₂
end

"""
$(TYPEDSIGNATURES)

Compute the dewpoint temperature using a phase `equilibrium` model to determine
the condensation surface based on temperature `T`.
"""
@inline function dewpoint_temperature(pᵛ, T, constants, equilibrium::AbstractPhaseEquilibrium;
                                      tolerance = 1e-4,
                                      maxiter = 10)
    surface = equilibrated_surface(equilibrium, T)
    return dewpoint_temperature(pᵛ, T, constants, surface; tolerance, maxiter)
end
