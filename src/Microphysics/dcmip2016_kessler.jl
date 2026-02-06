using ..Thermodynamics:
    MoistureMassFractions,
    MoistureMixingRatio,
    PlanarLiquidSurface,
    mixture_gas_constant,
    mixture_heat_capacity,
    saturation_specific_humidity,
    temperature,
    total_mixing_ratio,
    total_specific_moisture,
    with_moisture

using ..AtmosphereModels:
    dynamics_density,
    dynamics_pressure,
    surface_pressure

using ..ParcelModels: ParcelModel

using Oceananigans: Oceananigans, CenterField, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: Center, znode
using Oceananigans.Utils: launch!

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

"""
    struct DCMIP2016KesslerMicrophysics{FT}

DCMIP2016 implementation of the Kessler (1969) warm-rain bulk microphysics scheme.
See the constructor [`DCMIP2016KesslerMicrophysics`](@ref) for full documentation.
"""
struct DCMIP2016KesslerMicrophysics{FT}
    # DCMIP2016 parameter (appears to be related to Tetens' saturation vapor pressure formula,
    # but cannot be reconciled with other parameters in a consistent application of that formula.)
    dcmip_temperature_scale :: FT

    # Rain terminal velocity (Klemp & Wilhelmson 1978)
    terminal_velocity_coefficient :: FT
    density_scale                 :: FT
    terminal_velocity_exponent    :: FT

    # Autoconversion
    autoconversion_rate      :: FT
    autoconversion_threshold :: FT

    # Accretion
    accretion_rate     :: FT
    accretion_exponent :: FT

    # Rain evaporation (Klemp & Wilhelmson 1978)
    evaporation_ventilation_coefficient_1 :: FT
    evaporation_ventilation_coefficient_2 :: FT
    evaporation_ventilation_exponent_1    :: FT
    evaporation_ventilation_exponent_2    :: FT
    diffusivity_coefficient               :: FT
    thermal_conductivity_coefficient      :: FT

    # Numerical
    substep_cfl :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a DCMIP2016 implementation of the Kessler (1969) warm-rain bulk microphysics scheme.

This implementation follows the DCMIP2016 test case specification, which is based on
Klemp and Wilhelmson (1978).

# Positional Arguments
- `FT`: Floating-point type for all parameters (default: `Oceananigans.defaults.FloatType`).

# References
- Zarzycki, C. M., et al. (2019). DCMIP2016: the splitting supercell test case. Geoscientific Model Development, 12, 879–892.
- Kessler, E. (1969). On the Distribution and Continuity of Water Substance in Atmospheric Circulations.
  Meteorological Monographs, 10(32).
- Klemp, J. B., & Wilhelmson, R. B. (1978). The Simulation of Three-Dimensional Convective Storm Dynamics.
  Journal of the Atmospheric Sciences, 35(6), 1070-1096.
- DCMIP2016 Fortran implementation (`kessler.f90` in [DOI: 10.5281/zenodo.1298671](https://doi.org/10.5281/zenodo.1298671))

# Moisture Categories
This scheme represents moisture in three categories:
- Water vapor mixing ratio (`rᵛ`)
- Cloud water mixing ratio (`rᶜˡ`)
- Rain water mixing ratio (`rʳ`)

Breeze tracks moisture using mass fractions (`q`), whereas the Kessler scheme uses mixing ratios (`r`).
Conversions between these representations are performed internally. In Breeze, water vapor is not a prognostic variable;
instead, it is diagnosed from the total specific moisture `qᵗ` and the liquid condensates.

# Physical Processes
1. **Autoconversion**: Cloud water converts to rain water when the cloud water mixing ratio exceeds a threshold.
2. **Accretion**: Rain water collects cloud water as it falls.
3. **Saturation Adjustment**: Water vapor condenses to cloud water or cloud water evaporates to maintain saturation.
4. **Rain Evaporation**: Rain water evaporates into subsaturated air.
5. **Rain Sedimentation**: Rain water falls gravitationally.

# Implementation Details
- The microphysics update is applied via a GPU-compatible kernel launched from `microphysics_model_update!`.
- Rain sedimentation uses subcycling to satisfy CFL constraints, following the Fortran implementation.
- All microphysical updates are applied directly to the state variables in the kernel.

# Keyword Arguments

## Saturation (Tetens/Clausius-Clapeyron formula)
- `dcmip_temperature_scale` (`T_DCMIP2016`): A parameter of uncertain provenance that appears in the DCMIP2016 implementation
                            of the Kessler scheme (line 105 of `kessler.f90` in [DOI: 10.5281/zenodo.1298671](https://doi.org/10.5281/zenodo.1298671))

The "saturation adjustment coefficient" `f₅` is then computed as

```math
f₅ = a × T_DCMIP2016 × ℒˡᵣ / cᵖᵈ
```

where `a` is the liquid_coefficient for Tetens' saturation vapor pressure formula,
`ℒˡᵣ` is the latent heat of vaporization of liquid water, and `cᵖᵈ` is the heat capacity of dry air.

## Rain Terminal Velocity (Klemp & Wilhelmson 1978, eq. 2.15)
Terminal velocity: `𝕎ʳ = a𝕎 × (ρ × rʳ × Cᵨ)^β𝕎 × √(ρ₀/ρ)`
- `terminal_velocity_coefficient` (`a𝕎`): Terminal velocity coefficient in m/s (default: 36.34)
- `density_scale` (`Cᵨ`): Density scale factor for unit conversion (default: 0.001)
- `terminal_velocity_exponent` (`β𝕎`): Terminal velocity exponent (default: 0.1364)
- `ρ`: Density
- `ρ₀`: Reference density at z=0

## Autoconversion
- `autoconversion_rate` (`k₁`): Autoconversion rate coefficient in s⁻¹ (default: 0.001)
- `autoconversion_threshold` (`rᶜˡ★`): Critical cloud water mixing ratio threshold in kg/kg (default: 0.001)

## Accretion
- `accretion_rate` (`k₂`): Accretion rate coefficient in s⁻¹ (default: 2.2)
- `accretion_exponent` (`βᵃᶜᶜ`): Accretion exponent for rain mixing ratio (default: 0.875)

## Rain Evaporation (Klemp & Wilhelmson 1978, eq. 2.14)
Ventilation: `(Cᵉᵛ₁ + Cᵉᵛ₂ × (ρ rʳ)^βᵉᵛ₁) × (ρ rʳ)^βᵉᵛ₂`
- `evaporation_ventilation_coefficient_1` (`Cᵉᵛ₁`): Evaporation ventilation coefficient 1 (default: 1.6)
- `evaporation_ventilation_coefficient_2` (`Cᵉᵛ₂`): Evaporation ventilation coefficient 2 (default: 124.9)
- `evaporation_ventilation_exponent_1` (`βᵉᵛ₁`): Evaporation ventilation exponent 1 (default: 0.2046)
- `evaporation_ventilation_exponent_2` (`βᵉᵛ₂`): Evaporation ventilation exponent 2 (default: 0.525)
- `diffusivity_coefficient` (`Cᵈⁱᶠᶠ`): Diffusivity-related denominator coefficient (default: 2.55e8)
- `thermal_conductivity_coefficient` (`Cᵗʰᵉʳᵐ`): Thermal conductivity-related denominator coefficient (default: 5.4e5)

## Numerical
- `substep_cfl`: CFL safety factor for sedimentation subcycling (default: 0.8)
"""
function DCMIP2016KesslerMicrophysics(FT = Oceananigans.defaults.FloatType;
                                      dcmip_temperature_scale               = 237.3,
                                      terminal_velocity_coefficient         = 36.34,
                                      density_scale                         = 0.001,
                                      terminal_velocity_exponent            = 0.1364,
                                      autoconversion_rate                   = 0.001,
                                      autoconversion_threshold              = 0.001,
                                      accretion_rate                        = 2.2,
                                      accretion_exponent                    = 0.875,
                                      evaporation_ventilation_coefficient_1 = 1.6,
                                      evaporation_ventilation_coefficient_2 = 124.9,
                                      evaporation_ventilation_exponent_1    = 0.2046,
                                      evaporation_ventilation_exponent_2    = 0.525,
                                      diffusivity_coefficient               = 2.55e8,
                                      thermal_conductivity_coefficient      = 5.4e5,
                                      substep_cfl                           = 0.8)

    return DCMIP2016KesslerMicrophysics{FT}(convert(FT, dcmip_temperature_scale),
                                            convert(FT, terminal_velocity_coefficient),
                                            convert(FT, density_scale),
                                            convert(FT, terminal_velocity_exponent),
                                            convert(FT, autoconversion_rate),
                                            convert(FT, autoconversion_threshold),
                                            convert(FT, accretion_rate),
                                            convert(FT, accretion_exponent),
                                            convert(FT, evaporation_ventilation_coefficient_1),
                                            convert(FT, evaporation_ventilation_coefficient_2),
                                            convert(FT, evaporation_ventilation_exponent_1),
                                            convert(FT, evaporation_ventilation_exponent_2),
                                            convert(FT, diffusivity_coefficient),
                                            convert(FT, thermal_conductivity_coefficient),
                                            convert(FT, substep_cfl))
end

const DCMIP2016KM = DCMIP2016KesslerMicrophysics

"""
$(TYPEDSIGNATURES)

Return the names of prognostic microphysical fields for the Kessler scheme.

# Fields
- `:ρqᶜˡ`: Density-weighted cloud liquid mass fraction (\$kg/m^3\$).
- `:ρqʳ`: Density-weighted rain mass fraction (\$kg/m^3\$).
"""
AtmosphereModels.prognostic_field_names(::DCMIP2016KM) = (:ρqᶜˡ, :ρqʳ)

# Gridless microphysical state: convert density-weighted prognostics to specific quantities.
# The grid-indexed version is a generic wrapper that extracts μ from fields and calls this.
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, μ, 𝒰)
    qᶜˡ = μ.ρqᶜˡ / ρ
    qʳ = μ.ρqʳ / ρ
    return AtmosphereModels.WarmRainState(qᶜˡ, qʳ)
end

# Disambiguation for μ::Nothing (no prognostics yet)
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, ::Nothing, 𝒰)
    return AtmosphereModels.NothingMicrophysicalState(typeof(ρ))
end

# Disambiguation for empty NamedTuple
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, ::NamedTuple{(), Tuple{}}, 𝒰)
    return AtmosphereModels.NothingMicrophysicalState(typeof(ρ))
end

"""
$(TYPEDSIGNATURES)

Create and return the microphysical fields for the Kessler scheme.

# Prognostic Fields (Density-Weighted)
- `ρqᶜˡ`: Density-weighted cloud liquid mass fraction.
- `ρqʳ`: Density-weighted rain mass fraction.

# Diagnostic Fields (Mass Fractions)
- `qᵛ`: Water vapor mass fraction, diagnosed as \$q^v = q^t - q^{cl} - q^r\$.
- `qᶜˡ`: Cloud liquid mass fraction (\$kg/kg\$).
- `qʳ`: Rain mass fraction (\$kg/kg\$).
- `precipitation_rate`: Surface precipitation rate (\$m/s\$), defined as \$q^r \times v^t_{rain}\$ to match one-moment microphysics.
- `𝕎ʳ`: Rain terminal velocity (\$m/s\$).
"""
function AtmosphereModels.materialize_microphysical_fields(::DCMIP2016KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted)
    ρqᶜˡ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᶜˡ)
    ρqʳ  = CenterField(grid, boundary_conditions=boundary_conditions.ρqʳ)

    # Diagnostic fields (mass fractions)
    qᵛ  = CenterField(grid)
    qᶜˡ = CenterField(grid)
    qʳ  = CenterField(grid)

    # Precipitation and velocity diagnostics
    precipitation_rate = Field{Center, Center, Nothing}(grid)
    𝕎ʳ = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, precipitation_rate, 𝕎ʳ)
end

#####
##### Interface functions for AtmosphereModel integration
#####

# Note: grid_moisture_fractions uses the generic implementation.
# microphysical_state is called with 𝒰 = nothing, which works because
# DCMIP2016Kessler's microphysical_state doesn't use 𝒰.

"""
$(TYPEDSIGNATURES)

Return the thermodynamic state without adjustment.

The Kessler scheme performs its own saturation adjustment internally via the kernel.
"""
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::DCMIP2016KM, qᵗ, constants) = 𝒰

"""
$(TYPEDSIGNATURES)

Return `nothing`.

Rain sedimentation is handled internally by the kernel rather than through the advection interface.
"""
@inline AtmosphereModels.microphysical_velocities(::DCMIP2016KM, μ, name) = nothing

"""
$(TYPEDSIGNATURES)

Return zero tendency.

All microphysical source/sink terms are applied directly to the prognostic fields via the
`microphysics_model_update!` kernel, bypassing the standard tendency interface.
"""
@inline AtmosphereModels.microphysical_tendency(::DCMIP2016KM, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

#####
##### Precipitation rate and surface flux diagnostics
#####

"""
$(TYPEDSIGNATURES)

Return the liquid precipitation rate field for the DCMIP2016 Kessler microphysics scheme.

The precipitation rate is computed internally by the Kessler kernel and stored in
`μ.precipitation_rate`. It is defined as \$q^r \times v^t_{rain}\$ (rain mass fraction
times terminal velocity), matching the one-moment microphysics definition. Units are m/s.

This implements the Breeze `precipitation_rate(model, phase)` interface, allowing
the DCMIP2016 Kessler scheme to integrate with Breeze's standard diagnostics.
"""
AtmosphereModels.precipitation_rate(model, ::DCMIP2016KM, ::Val{:liquid}) = model.microphysical_fields.precipitation_rate

# Ice precipitation is not supported for this warm-phase Kessler scheme
AtmosphereModels.precipitation_rate(model, ::DCMIP2016KM, ::Val{:ice}) = nothing

"""
$(TYPEDSIGNATURES)

Return the surface precipitation flux field for the DCMIP2016 Kessler microphysics scheme.

The surface precipitation flux is \$\rho q^r v^t_{rain}\$ at the surface, matching the
one-moment microphysics definition. Units are kg/m²/s.

This implements the Breeze `surface_precipitation_flux(model)` interface.
"""
function AtmosphereModels.surface_precipitation_flux(model, ::DCMIP2016KM)
    grid = model.grid
    μ = model.microphysical_fields
    ρ = model.dynamics.reference_state.density
    # precipitation_rate = qʳ × vᵗ (m/s)
    # surface_precipitation_flux = ρ × qʳ × vᵗ = ρ × precipitation_rate (kg/m²/s)
    kernel = DCMIP2016KesslerSurfaceFluxKernel(μ.precipitation_rate, ρ)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid)
    return Field(op)
end

struct DCMIP2016KesslerSurfaceFluxKernel{P, R}
    precipitation_rate :: P
    reference_density :: R
end

Adapt.adapt_structure(to, k::DCMIP2016KesslerSurfaceFluxKernel) =
    DCMIP2016KesslerSurfaceFluxKernel(adapt(to, k.precipitation_rate),
                                      adapt(to, k.reference_density))

@inline function (kernel::DCMIP2016KesslerSurfaceFluxKernel)(i, j, k_idx, grid)
    # precipitation_rate = qʳ × vᵗ at surface
    # surface_precipitation_flux = ρ × precipitation_rate
    @inbounds P = kernel.precipitation_rate[i, j]
    @inbounds ρ = kernel.reference_density[i, j, 1]
    return ρ * P
end

"""
$(TYPEDSIGNATURES)

Compute rain terminal velocity (m/s) following Klemp and Wilhelmson (1978) eq. 2.15.

The terminal velocity is computed as:
```math
𝕎ʳ = a^𝕎 × (ρ × rʳ × Cᵨ)^{β^𝕎} × \\sqrt{ρ₀/ρ}
```

where `a^𝕎` is `terminal_velocity_coefficient`, `Cᵨ` is `density_scale`,
and `β^𝕎` is `terminal_velocity_exponent`.
"""
@inline function kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
    a𝕎 = microphysics.terminal_velocity_coefficient
    Cᵨ = microphysics.density_scale
    β𝕎 = microphysics.terminal_velocity_exponent
    return a𝕎 * (rʳ * Cᵨ * ρ)^β𝕎 * sqrt(ρ₁ / ρ)
end

"""
    cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)

Compute cloud-to-rain production rate from autoconversion and accretion (Klemp & Wilhelmson 1978, eq. 2.13).

This implements the combined effect of:
- **Autoconversion**: Cloud water spontaneously converting to rain when `rᶜˡ > rᶜˡ★`
- **Accretion**: Rain collecting cloud water as it falls

The formula uses an implicit time integration for numerical stability.
"""
@inline function cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)
    k₁   = microphysics.autoconversion_rate
    rᶜˡ★ = microphysics.autoconversion_threshold
    k₂   = microphysics.accretion_rate
    βᵃᶜᶜ = microphysics.accretion_exponent

    Aʳ = max(0, k₁ * (rᶜˡ - rᶜˡ★))    # Autoconversion rate
    denom = 1 + Δt * k₂ * rʳ^βᵃᶜᶜ       # Implicit accretion factor
    Δrᴾ = rᶜˡ - (rᶜˡ - Δt * Aʳ) / denom
    return Δrᴾ
end

#####
##### Main update function - launches GPU kernel
#####

"""
$(TYPEDSIGNATURES)

Apply the Kessler microphysics to the model.

This function launches a kernel that processes each column independently, with rain sedimentation subcycling.

The kernel handles conversion between mass fractions and mixing ratios
internally for efficiency. Water vapor is diagnosed from \$q^v = q^t - q^{cl} - q^r\$.
"""
function AtmosphereModels.microphysics_model_update!(microphysics::DCMIP2016KM, model)
    grid = model.grid
    arch = architecture(grid)
    Nz = grid.Nz
    Δt = model.clock.last_Δt

    # Skip microphysics update if timestep is zero, infinite, or invalid
    # (e.g., during model construction before any time step has been taken)
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    # Density and pressure fields (compatible with both Anelastic and Compressible dynamics)
    ρ = dynamics_density(model.dynamics)
    p = dynamics_pressure(model.dynamics)

    # Surface pressure for Exner function
    p₀ = surface_pressure(model.dynamics)

    # Thermodynamic constants for liquid-ice potential temperature conversion
    constants = model.thermodynamic_constants

    # Thermodynamic fields (liquid-ice potential temperature, NOT regular potential temperature)
    θˡⁱ  = model.formulation.potential_temperature
    ρθˡⁱ = model.formulation.potential_temperature_density

    # Total moisture density (prognostic variable of AtmosphereModel)
    ρqᵗ = model.moisture_density

    # Microphysical fields
    μ = model.microphysical_fields

    launch!(arch, grid, :xy, _microphysical_update!,
            microphysics, grid, Nz, Δt, ρ, p, p₀, constants, θˡⁱ, ρθˡⁱ, ρqᵗ, μ)

    return nothing
end

function saturation_adjustment_coefficient(T_DCMIP2016, constants)
    a = constants.saturation_vapor_pressure.liquid_coefficient
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    return a * T_DCMIP2016 * ℒˡᵣ / cᵖᵈ
end

#####
##### Shared core Kessler microphysics
#####
# These @inline functions encapsulate the core Kessler physics shared between
# the Eulerian grid kernel and the Lagrangian parcel model.

"""
$(TYPEDSIGNATURES)

Apply one Kessler microphysics step: autoconversion, accretion, saturation adjustment,
rain evaporation, and condensation.

`Δr𝕎` is the sedimentation flux divergence (zero for parcel models).

Returns `(rᵛ, rᶜˡ, rʳ, Δrˡ)`.
"""
@inline function step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, Δr𝕎, T, ρ, p, Δt,
                                           microphysics, constants, f₅, δT, FT)
    surface = PlanarLiquidSurface()
    Cᵨ     = microphysics.density_scale
    Cᵉᵛ₁   = microphysics.evaporation_ventilation_coefficient_1
    Cᵉᵛ₂   = microphysics.evaporation_ventilation_coefficient_2
    βᵉᵛ₁   = microphysics.evaporation_ventilation_exponent_1
    βᵉᵛ₂   = microphysics.evaporation_ventilation_exponent_2
    Cᵈⁱᶠᶠ  = microphysics.diffusivity_coefficient
    Cᵗʰᵉʳᵐ = microphysics.thermal_conductivity_coefficient

    # Autoconversion + Accretion: cloud → rain (KW eq. 2.13)
    Δrᴾ = cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)
    rᶜˡ = max(0, rᶜˡ - Δrᴾ)
    rʳ = max(0, rʳ + Δrᴾ + Δr𝕎)

    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, surface)
    rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

    # Saturation adjustment
    Δrˢᵃᵗ = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (T - δT)^2)

    # Rain evaporation (KW eq. 2.14)
    ρᵏ = Cᵨ * ρ
    ρrʳ = ρᵏ * rʳ
    Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂
    Dᵗʰ = Cᵈⁱᶠᶠ / (p * rᵛ⁺) + Cᵗʰᵉʳᵐ
    Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)
    Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (ρᵏ * rᵛ⁺ + FT(1e-20))
    Δrᴱmax = max(0, -Δrˢᵃᵗ - rᶜˡ)
    Δrᴱ = min(min(Δt * Ėʳ, Δrᴱmax), rʳ)

    # Condensation (limited by available cloud water)
    Δrᶜ = max(Δrˢᵃᵗ, -rᶜˡ)
    rᵛ = max(0, rᵛ - Δrᶜ + Δrᴱ)
    rᶜˡ = rᶜˡ + Δrᶜ
    rʳ = rʳ - Δrᴱ

    Δrˡ = Δrᶜ - Δrᴱ

    return rᵛ, rᶜˡ, rʳ, Δrˡ
end

"""
$(TYPEDSIGNATURES)

Convert from mixing ratios back to mass fractions.

Returns `(qᵛ, qᶜˡ, qʳ, qᵗ)`.
"""
@inline function mixing_ratios_to_mass_fractions(rᵛ, rᶜˡ, rʳ)
    rˡ = rᶜˡ + rʳ
    r = MoistureMixingRatio(rᵛ, rˡ)
    q = MoistureMassFractions(r)
    qᵛ = q.vapor
    qᵗ = total_specific_moisture(q)
    rᵗ = total_mixing_ratio(r)
    qᶜˡ = rᶜˡ / (1 + rᵗ)
    qʳ  = rʳ / (1 + rᵗ)

    return qᵛ, qᶜˡ, qʳ, qᵗ
end

"""
$(TYPEDSIGNATURES)

Convert from mass fractions to mixing ratios.

Returns `(rᵛ, rᶜˡ, rʳ)` mixing ratios for use in Kessler physics.
"""
@inline function mass_fractions_to_mixing_ratios(qᵗ, ρqᶜˡ, ρqʳ, ρ)
    qᶜˡ = max(0, ρqᶜˡ / ρ)
    qʳ  = max(0, ρqʳ / ρ)
    qˡ_sum = qᶜˡ + qʳ
    qᵗ = max(qᵗ, qˡ_sum)
    qᵛ = qᵗ - qˡ_sum

    q = MoistureMassFractions(qᵛ, qˡ_sum)
    r = MoistureMixingRatio(q)
    rᵛ = r.vapor
    rᵗ = total_mixing_ratio(r)
    rᶜˡ = qᶜˡ * (1 + rᵗ)
    rʳ  = qʳ * (1 + rᵗ)

    return rᵛ, rᶜˡ, rʳ
end

#####
##### GPU kernel for Kessler microphysics
#####

# Algorithm overview:
# 1. Convert mass fractions → mixing ratios; compute terminal velocities and CFL timestep
# 2. Subcycle: sedimentation, autoconversion, accretion, saturation adjustment, evaporation
# 3. Convert mixing ratios → mass fractions; update prognostic fields
#
# Note: Breeze uses liquid-ice potential temperature (θˡⁱ), related to T by:
#   T = Π θˡⁱ + ℒˡᵣ qˡ / cᵖᵐ

@kernel function _microphysical_update!(microphysics, grid, Nz, Δt,
                                        density, pressure, p₀, constants,
                                        θˡⁱ, ρθˡⁱ, ρqᵗ, μ)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    precipitation_rate_field = μ.precipitation_rate

    # Thermodynamic constants
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    # Saturation adjustment coefficient: f₅ = a × T_DCMIP2016 × ℒˡᵣ / cᵖᵈ
    T_DCMIP2016 = microphysics.dcmip_temperature_scale
    f₅ = saturation_adjustment_coefficient(T_DCMIP2016, constants)

    # Temperature offset for saturation adjustment (from TetensFormula)
    δT = constants.saturation_vapor_pressure.liquid_temperature_offset

    # Microphysics parameters
    cfl = microphysics.substep_cfl
    Cᵨ  = microphysics.density_scale

    # Reference density at surface for terminal velocity (KW eq. 2.15)
    @inbounds ρ₁ = density[i, j, 1]

    #####
    ##### PHASE 1: Convert mass fraction → mixing ratio
    #####

    max_Δt = Δt
    zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())

    for k = 1:(Nz-1)
        @inbounds begin
            ρ = density[i, j, k]
            qᵗ = ρqᵗ[i, j, k] / ρ
            rᵛ, rᶜˡ, rʳ = mass_fractions_to_mixing_ratios(qᵗ, μ.ρqᶜˡ[i, j, k], μ.ρqʳ[i, j, k], ρ)

            𝕎ʳᵏ = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
            μ.𝕎ʳ[i, j, k] = 𝕎ʳᵏ

            # Store mixing ratios in diagnostic fields during physics
            μ.qᵛ[i, j, k]  = rᵛ
            μ.qᶜˡ[i, j, k] = rᶜˡ
            μ.qʳ[i, j, k]  = rʳ

            # CFL check for sedimentation
            zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Center())
            Δz = zᵏ⁺¹ - zᵏ
            max_Δt = min(max_Δt, cfl * Δz / 𝕎ʳᵏ)
            zᵏ = zᵏ⁺¹
        end
    end

    # k = Nz: no CFL update needed
    @inbounds begin
        ρ = density[i, j, Nz]
        qᵗ = ρqᵗ[i, j, Nz] / ρ
        rᵛ, rᶜˡ, rʳ = mass_fractions_to_mixing_ratios(qᵗ, μ.ρqᶜˡ[i, j, Nz], μ.ρqʳ[i, j, Nz], ρ)

        μ.𝕎ʳ[i, j, Nz] = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
        μ.qᵛ[i, j, Nz]  = rᵛ
        μ.qᶜˡ[i, j, Nz] = rᶜˡ
        μ.qʳ[i, j, Nz]  = rʳ
    end

    # Subcycling for CFL constraint on rain sedimentation
    Ns = max(1, ceil(Int, Δt / max_Δt))
    inv_Ns = inv(FT(Ns))
    Δtₛ = Δt * inv_Ns
    # Pˢᵘʳᶠ: accumulated surface precipitation rate (qʳ × 𝕎ʳ) over subcycles
    Pˢᵘʳᶠ = zero(FT)

    #####
    ##### PHASE 2: Subcycle microphysics (in mixing ratio space)
    #####

    for m = 1:Ns

        # Accumulate surface precipitation (qʳ × vᵗ)
        @inbounds begin
            rᵛ₁ = μ.qᵛ[i, j, 1]
            rᶜˡ₁ = μ.qᶜˡ[i, j, 1]
            rʳ₁ = μ.qʳ[i, j, 1]
            rᵗ₁ = rᵛ₁ + rᶜˡ₁ + rʳ₁
            qʳ₁ = rʳ₁ / (1 + rᵗ₁)
            Pˢᵘʳᶠ += qʳ₁ * μ.𝕎ʳ[i, j, 1]
        end

        zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())

        for k = 1:(Nz-1)
            @inbounds begin
                ρ = density[i, j, k]
                p = pressure[i, j, k]
                θˡⁱᵏ = θˡⁱ[i, j, k]
                rᵛ = μ.qᵛ[i, j, k]
                rᶜˡ = μ.qᶜˡ[i, j, k]
                rʳ = μ.qʳ[i, j, k]

                # Compute temperature from θˡⁱ
                rˡ = rᶜˡ + rʳ
                r = MoistureMixingRatio(rᵛ, rˡ)
                cᵖᵐ = mixture_heat_capacity(r, constants)
                Rᵐ  = mixture_gas_constant(r, constants)
                q = MoistureMassFractions(r)
                qˡ = q.liquid
                Π = (p / p₀)^(Rᵐ / cᵖᵐ)
                Tᵏ = Π * θˡⁱᵏ + ℒˡᵣ * qˡ / cᵖᵐ

                # Rain sedimentation flux (upstream differencing)
                ρᵏ = Cᵨ * ρ
                𝕎ʳᵏ = μ.𝕎ʳ[i, j, k]
                zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Center())
                Δz = zᵏ⁺¹ - zᵏ
                ρᵏ⁺¹ = Cᵨ * density[i, j, k+1]
                rʳᵏ⁺¹ = μ.qʳ[i, j, k+1]
                𝕎ʳᵏ⁺¹ = μ.𝕎ʳ[i, j, k+1]
                Δr𝕎 = Δtₛ * (ρᵏ⁺¹ * rʳᵏ⁺¹ * 𝕎ʳᵏ⁺¹ - ρᵏ * rʳ * 𝕎ʳᵏ) / (ρᵏ * Δz)
                zᵏ = zᵏ⁺¹

                # Core microphysics step
                rᵛ, rᶜˡ, rʳ, Δrˡ = step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, Δr𝕎, Tᵏ, ρ, p, Δtₛ,
                                                             microphysics, constants, f₅, δT, FT)

                μ.qᵛ[i, j, k]  = rᵛ
                μ.qᶜˡ[i, j, k] = rᶜˡ
                μ.qʳ[i, j, k]  = rʳ

                # Update θˡⁱ from latent heating
                ΔT_phase = ℒˡᵣ / cᵖᵈ * Δrˡ
                T = Tᵏ + ΔT_phase

                rˡ = rᶜˡ + rʳ
                r = MoistureMixingRatio(rᵛ, rˡ)
                cᵖᵐ = mixture_heat_capacity(r, constants)
                Rᵐ  = mixture_gas_constant(r, constants)
                q = MoistureMassFractions(r)
                qˡ = q.liquid
                Π = (p / p₀)^(Rᵐ / cᵖᵐ)
                θˡⁱ_new = (T - ℒˡᵣ * qˡ / cᵖᵐ) / Π

                θˡⁱ[i, j, k]  = θˡⁱ_new
                ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
            end
        end

        # k = Nz: top boundary, rain falls out
        @inbounds begin
            k = Nz
            ρ = density[i, j, k]
            p = pressure[i, j, k]
            θˡⁱᵏ = θˡⁱ[i, j, k]
            rᵛ = μ.qᵛ[i, j, k]
            rᶜˡ = μ.qᶜˡ[i, j, k]
            rʳ = μ.qʳ[i, j, k]

            # Compute temperature from θˡⁱ
            rˡ = rᶜˡ + rʳ
            r = MoistureMixingRatio(rᵛ, rˡ)
            cᵖᵐ = mixture_heat_capacity(r, constants)
            Rᵐ  = mixture_gas_constant(r, constants)
            q = MoistureMassFractions(r)
            qˡ = q.liquid
            Π = (p / p₀)^(Rᵐ / cᵖᵐ)
            Tᵏ = Π * θˡⁱᵏ + ℒˡᵣ * qˡ / cᵖᵐ

            # Rain sedimentation flux at top boundary
            𝕎ʳᵏ = μ.𝕎ʳ[i, j, k]
            zᵏ = znode(i, j, k, grid, Center(), Center(), Center())
            zᵏ⁻¹ = znode(i, j, k-1, grid, Center(), Center(), Center())
            Δz_half = (zᵏ - zᵏ⁻¹) / 2
            Δr𝕎 = -Δtₛ * rʳ * 𝕎ʳᵏ / Δz_half

            # Core microphysics step (shared with ParcelModel)
            rᵛ, rᶜˡ, rʳ, Δrˡ = step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, Δr𝕎, Tᵏ, ρ, p, Δtₛ,
                                                         microphysics, constants, f₅, δT, FT)

            μ.qᵛ[i, j, k]  = rᵛ
            μ.qᶜˡ[i, j, k] = rᶜˡ
            μ.qʳ[i, j, k]  = rʳ

            # Update θˡⁱ from latent heating
            ΔT_phase = ℒˡᵣ / cᵖᵈ * Δrˡ
            T = Tᵏ + ΔT_phase

            rˡ = rᶜˡ + rʳ
            r = MoistureMixingRatio(rᵛ, rˡ)
            cᵖᵐ = mixture_heat_capacity(r, constants)
            Rᵐ  = mixture_gas_constant(r, constants)
            q = MoistureMassFractions(r)
            qˡ = q.liquid
            Π = (p / p₀)^(Rᵐ / cᵖᵐ)
            θˡⁱ_new = (T - ℒˡᵣ * qˡ / cᵖᵐ) / Π

            θˡⁱ[i, j, k]  = θˡⁱ_new
            ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
        end

        # Update terminal velocities for next subcycle
        if m < Ns
            for k = 1:Nz
                @inbounds begin
                    ρ = density[i, j, k]
                    rʳ = μ.qʳ[i, j, k]
                    μ.𝕎ʳ[i, j, k] = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
                end
            end
        end
    end

    @inbounds precipitation_rate_field[i, j, 1] = Pˢᵘʳᶠ * inv_Ns

    #####
    ##### PHASE 3: Convert mixing ratio → mass fraction
    #####

    for k = 1:Nz
        @inbounds begin
            ρ = density[i, j, k]
            rᵛ = μ.qᵛ[i, j, k]
            rᶜˡ = μ.qᶜˡ[i, j, k]
            rʳ = μ.qʳ[i, j, k]

            qᵛ, qᶜˡ, qʳ, qᵗ = mixing_ratios_to_mass_fractions(rᵛ, rᶜˡ, rʳ)

            ρqᵗ[i, j, k]    = ρ * qᵗ
            μ.ρqᶜˡ[i, j, k] = ρ * qᶜˡ
            μ.ρqʳ[i, j, k]  = ρ * qʳ
            μ.qᵛ[i, j, k]   = qᵛ
            μ.qᶜˡ[i, j, k]  = qᶜˡ
            μ.qʳ[i, j, k]   = qʳ
        end
    end
end

#####
##### update_microphysical_auxiliaries! for DCMIP2016 Kessler
#####
#
# DCMIP2016 has specific auxiliary fields (no qˡ total liquid field).
# Rain sedimentation is handled by the internal kernel, not microphysical_velocities.

@inline function AtmosphereModels.update_microphysical_auxiliaries!(μ, i, j, k, grid, ::DCMIP2016KM, ℳ::AtmosphereModels.WarmRainState, ρ, 𝒰, constants)
    # State fields
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qʳ[i, j, k] = ℳ.qʳ

    # Vapor from thermodynamic state
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Note: DCMIP2016 does NOT have a qˡ (total liquid) field
    # Rain sedimentation is handled internally, not via microphysical_velocities

    return nothing
end

#####
##### Parcel model implementation
#####
# For parcel models, apply Kessler microphysics to the parcel's scalar state
# using the same shared core functions as the Eulerian kernel.
# Rain sedimentation is not applicable to a Lagrangian parcel (rain falls with the parcel).

"""
$(TYPEDSIGNATURES)

Apply DCMIP2016 Kessler microphysics to a parcel model.

For a Lagrangian parcel, the microphysics processes are:
1. **Autoconversion**: Cloud water → rain when cloud exceeds threshold
2. **Accretion**: Rain + cloud → rain (collection)
3. **Saturation adjustment**: Vapor ↔ cloud to maintain equilibrium
4. **Rain evaporation**: Rain → vapor in subsaturated air

Note: Rain sedimentation is not applicable to a Lagrangian parcel since
the parcel is a closed system (rain does not fall out of the parcel).
"""
function AtmosphereModels.microphysics_model_update!(microphysics::DCMIP2016KM, model::ParcelModel)
    Δt = model.clock.last_Δt

    # Skip microphysics update if timestep is zero, infinite, or invalid
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    state = model.dynamics.state
    constants = model.thermodynamic_constants

    # Extract parcel state
    ρ = state.ρ
    𝒰 = state.𝒰
    μ = state.μ

    # Get pressure at parcel height (interpolate from environmental profile)
    p_parcel = interpolate(state.z, model.dynamics.pressure)

    # Convert mass fractions → mixing ratios (shared helper)
    rᵛ, rᶜˡ, rʳ = mass_fractions_to_mixing_ratios(state.qᵗ, μ.ρqᶜˡ, μ.ρqʳ, ρ)

    # Temperature from thermodynamic state
    T = temperature(𝒰, constants)

    # Saturation adjustment parameters
    f₅ = saturation_adjustment_coefficient(microphysics.dcmip_temperature_scale, constants)
    δT = constants.saturation_vapor_pressure.liquid_temperature_offset
    FT = typeof(ρ)

    # Core microphysics step (no sedimentation for parcel: Δr𝕎 = 0)
    rᵛ, rᶜˡ, rʳ, _ = step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, zero(FT), T, ρ, p_parcel, Δt,
                                               microphysics, constants, f₅, δT, FT)

    # Convert mixing ratios → mass fractions (shared helper)
    _, qᶜˡ, qʳ, qᵗ = mixing_ratios_to_mass_fractions(rᵛ, rᶜˡ, rʳ)

    # Update parcel state
    state.μ = (; ρqᶜˡ = ρ * qᶜˡ, ρqʳ = ρ * qʳ)
    state.qᵗ = qᵗ
    state.ρqᵗ = ρ * qᵗ

    # Update thermodynamic state with new moisture fractions.
    # Parcel models conserve specific static energy; latent heating is implicit.
    rˡ = rᶜˡ + rʳ
    r = MoistureMixingRatio(rᵛ, rˡ)
    q = MoistureMassFractions(r)
    state.𝒰 = with_moisture(𝒰, q)

    # Keep static energy consistent with the thermodynamic state.
    state.ℰ = state.𝒰.static_energy
    state.ρℰ = ρ * state.ℰ

    return nothing
end
