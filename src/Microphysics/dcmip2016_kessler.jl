using ..Thermodynamics:
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    MoistureMixingRatio,
    PlanarLiquidSurface,
    StaticEnergyState,
    mixture_gas_constant,
    mixture_heat_capacity,
    saturation_specific_humidity,
    temperature,
    TetensFormulaThermodynamicConstants,
    total_mixing_ratio,
    total_specific_moisture,
    vapor_gas_constant,
    with_moisture,
    with_temperature

using ..AtmosphereModels:
    dynamics_density,
    dynamics_pressure,
    kernel_time_step,
    standard_pressure,
    thermodynamic_density_name,
    total_density

using ..ParcelModels: ParcelModel

using Oceananigans: Oceananigans, CenterField, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: Center
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

"""
    struct DCMIP2016KesslerMicrophysics{FT}

DCMIP2016 implementation of the [Kessler (1969)](@cite Kessler1969) warm-rain bulk microphysics scheme.
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

Construct a DCMIP2016 implementation of the [Kessler (1969)](@cite Kessler1969) warm-rain bulk microphysics scheme.

This implementation follows the DCMIP2016 test case specification, which is based on
[Klemp and Wilhelmson (1978)](@cite Klemp1978).

# Positional Arguments
- `FT`: Floating-point type for all parameters (default: `Oceananigans.defaults.FloatType`).

# References
- Zarzycki, C. M., et al. (2019). DCMIP2016: the splitting supercell test case. Geoscientific Model Development, 12, 879–892.
- Kessler, E. (1969). On the Distribution and Continuity of Water Substance in Atmospheric Circulations.
  Meteorological Monographs, 10(32).
- Klemp, J. B., & Wilhelmson, R. B. (1978). The simulation of three-dimensional convective storm dynamics.
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
- The microphysics update is applied via a GPU-compatible kernel launched from `microphysics_model_update!`,
  once per time step after the dynamics (operator splitting). All microphysical updates are applied
  directly to the prognostic fields in the kernel.
- **Density basis.** The prognostic water fields are partial densities, `ρqˣ = ρˣ`. The Kessler
  processes act on dry-air mixing ratios `rˣ = ρˣ / ρᵈ`, where the dry-air density is the prognostic
  `ρᵈ` on the compressible core and `ρᵈ = ρᵣ - ρᵗ` on the anelastic core (the reference density `ρᵣ`
  is the total density there; see [`dry_air_density`](@ref Breeze.Microphysics.dry_air_density)).
  Phase changes and cloud-to-rain conversion conserve `rᵗ` and hence `ρᵗ`.
- **Rain sedimentation** is an upwind, flux-form update of the rain partial density on the
  finite-volume cells: the flux `ρqʳ 𝕎ʳ` through each face is the same on both sides of the face
  and is divided by the thickness of the receiving cell, `Δzᶜᶜᶜ`, so the column rain budget closes
  to the surface flux on any vertical grid (uniform, stretched, or piecewise), including the top
  cell, which only loses rain through its bottom face. Vapor and cloud partial densities are not
  touched by sedimentation. The update is subcycled to satisfy `substep_cfl` on every cell's
  thickness (the DCMIP2016 Fortran uses the distance between levels and a half cell at the top).
- **Surface precipitation.** `surface_precipitation_flux` is the substep-mean bottom-face flux
  `(ρqʳ 𝕎ʳ)₁`, exactly the water removed from the column; `precipitation_rate` is that flux per
  unit of the final surface density.
- **Thermodynamics** (`LiquidIcePotentialTemperatureFormulation` only). Temperature is recovered
  from the prognostic `θˡⁱ` with Breeze's own `θˡⁱ ↔ T` relation. Sedimentation happens at fixed
  temperature, so `θˡⁱ` absorbs the change of liquid loading (falling rain carries water, not heat).
  Phase changes (saturation adjustment and rain evaporation) conserve `θˡⁱ`, as
  [`SaturationAdjustment`](@ref) does: the saturation adjustment is the DCMIP2016 single Newton step,
  linearized with the temperature response `∂T/∂rˡ` of the invariant
  ([`phase_change_temperature_slope`](@ref Breeze.Microphysics.phase_change_temperature_slope)), and
  the invariant is what is written back, so the post-step temperature follows from `θˡⁱ` and the new
  partition. This differs from the DCMIP2016 Fortran, which increments `T` by `ℒˡᵣ Δrˡ / cᵖᵈ`; that
  increment is not consistent with `θˡⁱ` (moist heat capacity, composition dependence of the Exner
  function) and acts as a spurious net `θˡⁱ` source. Conserving `θˡⁱ` is consistency with the
  prognostic invariant of the formulation; it is not exact conservation of the physical mixture
  enthalpy, which the formulation's constant-latent-heat `θˡⁱ` does not represent either.

# Keyword Arguments

## Saturation (Tetens/Clausius-Clapeyron formula)
- `dcmip_temperature_scale` (`T_DCMIP2016`): A parameter of uncertain provenance that appears in the DCMIP2016 implementation
                            of the Kessler scheme (line 105 of `kessler.f90` in [DOI: 10.5281/zenodo.1298671](https://doi.org/10.5281/zenodo.1298671))

The "saturation adjustment coefficient" `f₅` is then computed as

```math
f₅ = a T_DCMIP2016 ℒˡᵣ / cᵖᵈ
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

# The DCMIP2016 Kessler saturation adjustment reads the liquid coefficient and temperature
# offset of the Tetens formula (see `saturation_adjustment_coefficient` and the kernel), so it
# requires thermodynamic constants built with a `TetensFormula` saturation vapor pressure. With
# other formulations (e.g. the default `ClausiusClapeyron`) those fields are absent and the
# scheme would fail inside the kernel — an opaque `getproperty` error on the CPU and a GPU
# compilation failure. Validate at model construction to give a clear, early error instead.
function AtmosphereModels.validate_microphysics(::DCMIP2016KM, thermodynamic_constants)
    if !(thermodynamic_constants isa TetensFormulaThermodynamicConstants)
        svp = thermodynamic_constants.saturation_vapor_pressure
        throw(ArgumentError(string(
            "DCMIP2016KesslerMicrophysics requires `thermodynamic_constants` with a `TetensFormula` ",
            "saturation vapor pressure formulation, but got `", summary(svp), "`. ",
            "Construct the model with, e.g., ",
            "`thermodynamic_constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))`.")))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the names of prognostic microphysical fields for the Kessler scheme.

# Fields
- `:ρqᶜˡ`: Density-weighted cloud liquid mass fraction (kg/m³).
- `:ρqʳ`: Density-weighted rain mass fraction (kg/m³).
"""
AtmosphereModels.prognostic_field_names(::DCMIP2016KM) = (:ρqᶜˡ, :ρqʳ)

# Gridless microphysical state: convert density-weighted prognostics to specific quantities.
# The grid-indexed version is a generic wrapper that extracts μ from fields and calls this.
# The velocities argument is required for interface compatibility but not used by the Kessler schemes.
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    qʳ = μ.ρqʳ / ρ
    return AtmosphereModels.WarmRainState(qᶜˡ, qʳ)
end

# Disambiguation for μ::Nothing (no prognostics yet)
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, ::Nothing, 𝒰, velocities)
    return AtmosphereModels.NothingMicrophysicalState(typeof(ρ))
end

# Disambiguation for empty NamedTuple
@inline function AtmosphereModels.microphysical_state(::DCMIP2016KM, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities)
    return AtmosphereModels.NothingMicrophysicalState(typeof(ρ))
end

"""
$(TYPEDSIGNATURES)

Create and return the microphysical fields for the Kessler scheme.

# Prognostic Fields (Density-Weighted)
- `ρqᶜˡ`: Density-weighted cloud liquid mass fraction.
- `ρqʳ`: Density-weighted rain mass fraction.

# Diagnostic Fields (Mass Fractions)
- `qᵛ`: Water vapor mass fraction, diagnosed as ``q^v = q^t - q^{cl} - q^r``.
- `qᶜˡ`: Cloud liquid mass fraction (kg/kg).
- `qʳ`: Rain mass fraction (kg/kg).
- `precipitation_rate`: Surface precipitation rate (m/s), obtained by normalizing the
  substep-mean rain mass flux by the final surface air density.
- `𝕎ʳ`: Rain terminal velocity (m/s).
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
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::DCMIP2016KM, qᵛ, constants) = 𝒰

AtmosphereModels.moisture_prognostic_name(::DCMIP2016KM) = :ρqᵛ

# DCMIP2016 Kessler stores vapor as prognostic; subtract all condensate from total.
@inline function AtmosphereModels.specific_prognostic_moisture_from_total(::DCMIP2016KM, qᵗ, ℳ::AtmosphereModels.WarmRainState)
    return max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ)
end
AtmosphereModels.liquid_mass_fraction(::DCMIP2016KM, model) = model.microphysical_fields.qᶜˡ + model.microphysical_fields.qʳ

# Grid model: prognostic stores true vapor; construct fractions directly from fields.
@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::DCMIP2016KM, ρ, qᵛ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    qʳ = @inbounds μ.qʳ[i, j, k]
    qˡ = qᶜˡ + qʳ
    return MoistureMassFractions(qᵛ, qˡ)
end
AtmosphereModels.ice_mass_fraction(::DCMIP2016KM, model) = nothing

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
`μ.precipitation_rate`. The kernel time-averages the rain mass flux across sedimentation
substeps and normalizes it by the final surface air density. For a fixed-density column this
reduces to ``q^r v^t_{rain}``, matching the one-moment microphysics definition. Units are m/s.

This implements the Breeze `precipitation_rate(model, phase)` interface, allowing
the DCMIP2016 Kessler scheme to integrate with Breeze's standard diagnostics.
"""
AtmosphereModels.precipitation_rate(model, ::DCMIP2016KM, ::Val{:liquid}) = model.microphysical_fields.precipitation_rate

# Ice precipitation is not supported for this warm-phase Kessler scheme
AtmosphereModels.precipitation_rate(model, ::DCMIP2016KM, ::Val{:ice}) = nothing

"""
$(TYPEDSIGNATURES)

Return the surface precipitation flux field for the DCMIP2016 Kessler microphysics scheme.

The surface precipitation flux is the substep-mean ``ρ^r v^t_{rain}`` at the surface.
The stored precipitation rate is normalized so multiplying it by the final total density
recovers this mass flux exactly. Units are kg/m²/s.

This implements the Breeze `surface_precipitation_flux(model)` interface.
"""
function AtmosphereModels.surface_precipitation_flux(model, ::DCMIP2016KM)
    grid = model.grid
    μ = model.microphysical_fields
    ρ = total_density(model.dynamics)
    # surface_precipitation_flux = ρ × precipitation_rate (kg/m²/s)
    kernel = DCMIP2016KesslerSurfaceFluxKernel(μ.precipitation_rate, ρ)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid)
    return Field(op)
end

struct DCMIP2016KesslerSurfaceFluxKernel{P, R}
    precipitation_rate :: P
    density :: R
end

Adapt.adapt_structure(to, k::DCMIP2016KesslerSurfaceFluxKernel) =
    DCMIP2016KesslerSurfaceFluxKernel(adapt(to, k.precipitation_rate),
                                      adapt(to, k.density))

@inline function (kernel::DCMIP2016KesslerSurfaceFluxKernel)(i, j, k_idx, grid)
    # surface_precipitation_flux = ρ × precipitation_rate
    @inbounds P = kernel.precipitation_rate[i, j]
    @inbounds ρ = kernel.density[i, j, 1]
    return ρ * P
end

"""
$(TYPEDSIGNATURES)

Compute rain terminal velocity (m/s) following Klemp and Wilhelmson (1978) eq. 2.15.

The terminal velocity is computed as:
```math
𝕎ʳ = a^𝕎 (ρ rʳ Cᵨ)^{β^𝕎} \\sqrt{ρ₀/ρ}
```

where ``a^𝕎`` is the `terminal_velocity_coefficient`, ``Cᵨ`` is the `density_scale`,
and ``β^𝕎`` is the `terminal_velocity_exponent`.
"""
@inline function kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
    a𝕎 = microphysics.terminal_velocity_coefficient
    Cᵨ = microphysics.density_scale
    β𝕎 = microphysics.terminal_velocity_exponent
    return a𝕎 * (rʳ * Cᵨ * ρ)^β𝕎 * sqrt(ρ₁ / ρ)
end

"""
    cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)

Compute cloud-to-rain production rate from autoconversion and accretion
([Klemp and Wilhelmson 1978](@cite Klemp1978), eq. 2.13).

This implements the combined effect of:
- **Autoconversion**: Cloud water spontaneously converting to rain when `rᶜˡ > rᶜˡ★`
- **Accretion**: Rain collecting cloud water as it falls

The formula uses an implicit time integration for numerical stability.

# References
- Klemp, J. B., & Wilhelmson, R. B. (1978). The simulation of three-dimensional convective storm dynamics.
  Journal of the Atmospheric Sciences, 35(6), 1070-1096.
"""
@inline function cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)
    k₁   = microphysics.autoconversion_rate
    rᶜˡ★ = microphysics.autoconversion_threshold
    k₂   = microphysics.accretion_rate
    βᵃᶜᶜ = microphysics.accretion_exponent

    Aʳ = max(0, k₁ * (rᶜˡ - rᶜˡ★))      # Autoconversion rate
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
internally for efficiency. Water vapor is diagnosed from ``q^v = q^t - q^{cl} - q^r``.

The kernel writes prognostic fields in the interior only, so `update_state!` is called
afterwards to restore a consistent model state (halos, diagnostics, and tendencies).
"""
function AtmosphereModels.microphysics_model_update!(microphysics::DCMIP2016KM, model)
    grid = model.grid
    arch = architecture(grid)
    Nz = grid.Nz
    Δt = model.clock.last_Δt

    # Skip microphysics update if timestep is zero, infinite, or invalid
    # (e.g., during model construction before any time step has been taken)
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    # The kernel carries θˡⁱ through sedimentation and phase change (see the constructor
    # docstring); it has no static-energy counterpart yet.
    thermodynamic_density_name(model.formulation) === :ρθ ||
        throw(ArgumentError(string("DCMIP2016KesslerMicrophysics requires the ",
                                   "LiquidIcePotentialTemperatureFormulation (prognostic ρθ), but the model's ",
                                   "formulation evolves ", thermodynamic_density_name(model.formulation), ".")))

    # Total density weights water mass fractions and enters the Kessler air-density corrections.
    # The coupling density weights the thermodynamic prognostic ρθˡⁱ and, for compressible
    # dynamics, is the dry-air carrier of Kessler mixing ratios. These fields alias for anelastic
    # dynamics and are distinct (total and dry density) for compressible dynamics.
    ρ = total_density(model.dynamics)
    ρᵈ = dynamics_density(model.dynamics)
    dry_air_coupled = ρ !== ρᵈ
    p = dynamics_pressure(model.dynamics)

    # Standard pressure (reference pressure for the Exner function, pˢᵗ)
    pˢᵗ = standard_pressure(model.dynamics)

    # Thermodynamic constants for liquid-ice potential temperature conversion
    constants = model.thermodynamic_constants

    # Thermodynamic fields (liquid-ice potential temperature, NOT regular potential temperature)
    θˡⁱ  = model.formulation.potential_temperature
    ρθˡⁱ = model.formulation.potential_temperature_density

    # Vapor density (prognostic variable of AtmosphereModel for DCMIP2016KM)
    ρqᵛ = model.moisture_density

    # Microphysical fields
    μ = model.microphysical_fields

    kernel_Δt = kernel_time_step(arch, grid, Δt)

    launch!(arch, grid, :xy, _microphysical_update!,
            microphysics, grid, Nz, kernel_Δt, ρ, ρᵈ, dry_air_coupled,
            p, pˢᵗ, constants, θˡⁱ, ρθˡⁱ, ρqᵛ, μ)

    # The kernel mutated prognostic fields in the interior; refill halos and recompute
    # diagnostics and tendencies so the post-update state is consistent for the next step.
    update_state!(model)

    return nothing
end

# Constrained to `TetensFormulaThermodynamicConstants`: the coefficient `a` is the liquid
# coefficient of the Tetens formula, which only exists for those constants. The annotation
# makes the dependence explicit to the compiler (and to static analysis like JETLS), so an
# incompatible formulation fails as a clear `MethodError` rather than a dynamic `getproperty`
# inside the GPU kernel. `validate_microphysics` catches the mismatch earlier still.
"""
$(TYPEDSIGNATURES)

Return the DCMIP2016 saturation adjustment coefficient ``f₅ = a T_DCMIP2016 ∂T/∂rˡ``, where `a` is
the liquid coefficient of the Tetens formula and `∂T∂rˡ` is the temperature change per unit of
vapor mixing ratio converted to liquid. `a T_DCMIP2016 / (T - δT)²` approximates
``∂ \\ln pᵛ⁺ / ∂T`` of the Tetens formula, so `f₅ rᵛ⁺ / (T - δT)²` is the change of the saturation
mixing ratio produced by the latent heating of a unit condensation — the denominator of the
linearized (single Newton step) saturation adjustment of [Klemp and Wilhelmson (1978)](@cite Klemp1978).

The two-argument form uses the DCMIP2016 Fortran value ``∂T/∂rˡ = ℒˡᵣ / cᵖᵈ``. The kernel uses
[`phase_change_temperature_slope`](@ref) instead, which is the derivative of the temperature
recovered from the model's own prognostic invariant.
"""
@inline function saturation_adjustment_coefficient(T_DCMIP2016, ∂T∂rˡ, constants::TetensFormulaThermodynamicConstants)
    a = constants.saturation_vapor_pressure.liquid_coefficient
    return a * T_DCMIP2016 * ∂T∂rˡ
end

@inline function saturation_adjustment_coefficient(T_DCMIP2016, constants::TetensFormulaThermodynamicConstants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    return saturation_adjustment_coefficient(T_DCMIP2016, ℒˡᵣ / cᵖᵈ, constants)
end

#####
##### Thermodynamic coupling: density basis and the prognostic invariant
#####

"""
$(TYPEDSIGNATURES)

Return the dry-air density that carries the Kessler mixing ratios, `rˣ = ρˣ / ρᵈ`.

`ϱ` is the coupling density of the dynamics and `ρᵗ` the total water partial density. On the
compressible core (`dry_air_coupled = true`) the coupling density is the prognostic dry-air
density itself. On the anelastic core (`dry_air_coupled = false`) the coupling density is the
fixed reference density `ρᵣ`, which is the *total* density of the anelastic state, so the dry air
is what remains once the water is removed, `ρᵈ = ρᵣ - ρᵗ`. Sedimentation changes `ρᵗ`, and hence
`ρᵈ` on the anelastic core: water that falls into a cell displaces dry air there.
"""
@inline dry_air_density(ϱ, ρᵗ, dry_air_coupled) = ifelse(dry_air_coupled, ϱ, ϱ - ρᵗ)

# The pressure-based liquid-ice potential temperature state of a Kessler cell, from the
# prognostic θˡⁱ and the dry-air mixing ratios of vapor and (cloud plus rain) liquid. This is
# the state Breeze diagnoses for the anelastic θˡⁱ formulation, so `temperature` and
# `with_temperature` on it are Breeze's own θˡⁱ ↔ T relations.
@inline function kessler_thermodynamic_state(θˡⁱ, rᵛ, rˡ, p, pˢᵗ)
    r = MoistureMixingRatio(rᵛ, rˡ)
    q = MoistureMassFractions(r)
    return LiquidIcePotentialTemperatureState(θˡⁱ, q, pˢᵗ, p)
end

"""
$(TYPEDSIGNATURES)

Return ``∂T/∂rˡ``, the temperature change per unit of vapor mixing ratio converted to liquid at
fixed total moisture, holding the prognostic invariant of the thermodynamic state `𝒰` fixed.

For a [`LiquidIcePotentialTemperatureState`](@ref Breeze.Thermodynamics.LiquidIcePotentialTemperatureState),
``T = Π θˡⁱ + ℒˡᵣ qˡ / cᵖᵐ`` with ``Π = (p / pˢᵗ)^{Rᵐ / cᵖᵐ}``, so at fixed ``θˡⁱ``, ``p`` and ``qᵗ``
(``δqᵛ = -δqˡ``),

```math
\\frac{∂T}{∂qˡ} = θˡⁱ Π \\ln\\frac{p}{pˢᵗ} \\frac{∂κ}{∂qˡ} + \\frac{ℒˡᵣ}{cᵖᵐ} - \\frac{ℒˡᵣ qˡ (cˡ - cᵖᵛ)}{(cᵖᵐ)^2} ,
\\qquad
\\frac{∂κ}{∂qˡ} = -\\frac{Rᵛ cᵖᵐ + Rᵐ (cˡ - cᵖᵛ)}{(cᵖᵐ)^2} ,
```

and ``∂T/∂rˡ = (1 - qᵗ) ∂T/∂qˡ``. The leading term is ``ℒˡᵣ / cᵖᵐ``; the DCMIP2016 Fortran uses
``ℒˡᵣ / cᵖᵈ``, which differs by the moist heat capacity and neglects the composition dependence of
the Exner function. For a [`StaticEnergyState`](@ref Breeze.Thermodynamics.StaticEnergyState),
``T = (s - g z + ℒˡᵣ qˡ) / cᵖᵐ`` gives ``∂T/∂qˡ = (ℒˡᵣ - T (cˡ - cᵖᵛ)) / cᵖᵐ``.
"""
@inline function phase_change_temperature_slope(𝒰::LiquidIcePotentialTemperatureState, constants)
    q = 𝒰.moisture_mass_fractions
    θ = 𝒰.potential_temperature
    p = 𝒰.reference_pressure
    pˢᵗ = 𝒰.standard_pressure
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡᵣ = constants.liquid.reference_latent_heat
    qˡ = q.liquid
    qᵗ = total_specific_moisture(q)
    Δc = cˡ - cᵖᵛ
    Π = (p / pˢᵗ)^(Rᵐ / cᵖᵐ)
    ∂κ∂qˡ = -(Rᵛ * cᵖᵐ + Rᵐ * Δc) / cᵖᵐ^2
    ∂T∂qˡ = θ * Π * log(p / pˢᵗ) * ∂κ∂qˡ + ℒˡᵣ / cᵖᵐ - ℒˡᵣ * qˡ * Δc / cᵖᵐ^2
    return ∂T∂qˡ * (1 - qᵗ)
end

@inline function phase_change_temperature_slope(𝒰::StaticEnergyState, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡᵣ = constants.liquid.reference_latent_heat
    T = temperature(𝒰, constants)
    qᵗ = total_specific_moisture(q)
    ∂T∂qˡ = (ℒˡᵣ - T * (cˡ - cᵖᵛ)) / cᵖᵐ
    return ∂T∂qˡ * (1 - qᵗ)
end

#####
##### Shared core Kessler microphysics
#####
# These @inline functions encapsulate the core Kessler physics shared between
# the Eulerian grid kernel and the Lagrangian parcel model.

"""
$(TYPEDSIGNATURES)

Apply the local (cell-internal) Kessler processes to dry-air mixing ratios: autoconversion,
accretion, saturation adjustment, and rain evaporation.

`T` is the temperature of the incoming state and `∂T∂rˡ` the temperature response to
condensation at fixed prognostic invariant (see [`phase_change_temperature_slope`](@ref)), which
linearizes the saturation adjustment about that invariant. The function returns the new
partition only; the caller keeps the invariant fixed, so the temperature after the step is the
one the invariant implies for the new partition. Sedimentation is applied by the caller,
before this step, in partial-density space.

Returns `(rᵛ, rᶜˡ, rʳ, Δrˡ)` where `Δrˡ` is the net vapor → liquid conversion.
"""
@inline function step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, T, ∂T∂rˡ, ρ, p, Δt,
                                           microphysics, constants, δT, FT)
    surface = PlanarLiquidSurface()
    Cᵨ     = microphysics.density_scale
    Cᵉᵛ₁   = microphysics.evaporation_ventilation_coefficient_1
    Cᵉᵛ₂   = microphysics.evaporation_ventilation_coefficient_2
    βᵉᵛ₁   = microphysics.evaporation_ventilation_exponent_1
    βᵉᵛ₂   = microphysics.evaporation_ventilation_exponent_2
    Cᵈⁱᶠᶠ  = microphysics.diffusivity_coefficient
    Cᵗʰᵉʳᵐ = microphysics.thermal_conductivity_coefficient
    f₅ = saturation_adjustment_coefficient(microphysics.dcmip_temperature_scale, ∂T∂rˡ, constants)

    # Autoconversion + Accretion: cloud → rain (KW eq. 2.13)
    Δrᴾ = cloud_to_rain_production(rᶜˡ, rʳ, Δt, microphysics)
    rᶜˡ = max(0, rᶜˡ - Δrᴾ)
    rʳ = max(0, rʳ + Δrᴾ)

    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, surface)
    rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

    # Saturation adjustment: one Newton step of rᵛ - Δr = rᵛ⁺(T + ∂T∂rˡ Δr)
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
@inline function mass_fractions_to_mixing_ratios(qᵛ, ρqᶜˡ, ρqʳ, ρ)
    qᶜˡ = max(0, ρqᶜˡ / ρ)
    qʳ  = max(0, ρqʳ / ρ)
    qˡ_sum = qᶜˡ + qʳ
    qᵛ = max(0, qᵛ)

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

# Algorithm overview (one column per work item):
# 1. Clip negative inputs, compute rain terminal velocities, and pick the number of
#    sedimentation substeps from the CFL condition on every cell's thickness.
# 2. Subcycle, bottom to top within each substep. Per cell:
#    a. recover T from the prognostic θˡⁱ and the incoming partition (Breeze's θˡⁱ ↔ T relation);
#    b. sedimentation: upwind flux-form update of the rain partial density ρqʳ on the
#       finite-volume cell (fluxes ρqʳ 𝕎ʳ through the faces, divided by the cell thickness
#       Δzᶜᶜᶜ), at fixed temperature, so θˡⁱ absorbs the change of liquid loading;
#    c. local Kessler physics on the dry-air mixing ratios of the post-sedimentation state at
#       fixed θˡⁱ: the partition is linearized about the invariant and the invariant is what is
#       written back, so phase change is exactly θˡⁱ-conserving, like `SaturationAdjustment`.
# 3. The surface precipitation is the substep-mean bottom-face flux (ρqʳ 𝕎ʳ)₁ — exactly the
#    mass removed from the column — normalized by the final surface density.
#
# The prognostic fields ρqᵛ, ρqᶜˡ, ρqʳ hold the water partial densities throughout; the
# mixing ratios are formed per cell from the dry-air density (see `dry_air_density`).

@inline function kessler_column_cell!(i, j, k, grid, Fᵗᵒᵖ, Δt, microphysics,
                                      density, coupling_density, dry_air_coupled,
                                      pressure, pˢᵗ, constants, δT,
                                      θˡⁱ, ρθˡⁱ, ρqᵛ, μ)
    FT = eltype(grid)

    @inbounds begin
        ρ   = density[i, j, k]
        ϱ   = coupling_density[i, j, k]
        p   = pressure[i, j, k]
        θ₀  = θˡⁱ[i, j, k]
        ρᵛ  = ρqᵛ[i, j, k]
        ρᶜˡ = μ.ρqᶜˡ[i, j, k]
        ρʳ  = μ.ρqʳ[i, j, k]
        𝕎ʳ  = μ.𝕎ʳ[i, j, k]
    end

    # Temperature of the incoming state from the prognostic invariant
    ρᵈ₀ = dry_air_density(ϱ, ρᵛ + ρᶜˡ + ρʳ, dry_air_coupled)
    𝒰₀ = kessler_thermodynamic_state(θ₀, ρᵛ / ρᵈ₀, (ρᶜˡ + ρʳ) / ρᵈ₀, p, pˢᵗ)
    T = temperature(𝒰₀, constants)

    # Rain sedimentation: upwind fluxes ρqʳ 𝕎ʳ through the top (from the cell above) and
    # bottom faces, divided by the thickness of this cell. The same products are used on
    # both sides of every face, so the column budget telescopes to the surface flux.
    Fᵇᵒᵗ = ρʳ * 𝕎ʳ
    ρʳ = max(0, ρʳ + Δt * (Fᵗᵒᵖ - Fᵇᵒᵗ) / Δzᶜᶜᶜ(i, j, k, grid))

    # Sedimentation at fixed temperature: the invariant absorbs the change of liquid loading
    ρᵈ = dry_air_density(ϱ, ρᵛ + ρᶜˡ + ρʳ, dry_air_coupled)
    rᵛ  = ρᵛ / ρᵈ
    rᶜˡ = ρᶜˡ / ρᵈ
    rʳ  = ρʳ / ρᵈ
    𝒰₁ = with_temperature(kessler_thermodynamic_state(θ₀, rᵛ, rᶜˡ + rʳ, p, pˢᵗ), T, constants)
    θ₁ = 𝒰₁.potential_temperature

    # Local Kessler physics at fixed θ₁ (the temperature after the step is implied by θ₁ and
    # the new partition; `update_state!` diagnoses it)
    ∂T∂rˡ = phase_change_temperature_slope(𝒰₁, constants)
    rᵛ, rᶜˡ, rʳ, _ = step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, T, ∂T∂rˡ, ρ, p, Δt,
                                               microphysics, constants, δT, FT)

    @inbounds begin
        ρqᵛ[i, j, k]    = ρᵈ * rᵛ
        μ.ρqᶜˡ[i, j, k] = ρᵈ * rᶜˡ
        μ.ρqʳ[i, j, k]  = ρᵈ * rʳ
        θˡⁱ[i, j, k]    = θ₁
        ρθˡⁱ[i, j, k]   = ϱ * θ₁
    end

    return nothing
end

# Rain terminal velocity of cell k from the current partial densities
@inline function kessler_rain_terminal_velocity(i, j, k, microphysics, density, coupling_density,
                                                dry_air_coupled, ρ₁, ρqᵛ, μ)
    @inbounds begin
        ρ   = density[i, j, k]
        ϱ   = coupling_density[i, j, k]
        ρᵗ  = ρqᵛ[i, j, k] + μ.ρqᶜˡ[i, j, k] + μ.ρqʳ[i, j, k]
        rʳ  = μ.ρqʳ[i, j, k] / dry_air_density(ϱ, ρᵗ, dry_air_coupled)
    end
    return kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
end

@kernel function _microphysical_update!(microphysics, grid, Nz, Δt,
                                        density, coupling_density, dry_air_coupled,
                                        pressure, pˢᵗ, constants,
                                        θˡⁱ, ρθˡⁱ, ρqᵛ, μ)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    # Temperature offset for saturation adjustment (from TetensFormula)
    δT = constants.saturation_vapor_pressure.liquid_temperature_offset

    cfl = microphysics.substep_cfl

    # Reference density at surface for terminal velocity (KW eq. 2.15)
    @inbounds ρ₁ = density[i, j, 1]

    #####
    ##### PHASE 1: clip inputs, terminal velocities, and the sedimentation CFL limit
    #####

    max_Δt = Δt

    for k = 1:Nz
        @inbounds begin
            ρqᵛ[i, j, k]    = max(0, ρqᵛ[i, j, k])
            μ.ρqᶜˡ[i, j, k] = max(0, μ.ρqᶜˡ[i, j, k])
            μ.ρqʳ[i, j, k]  = max(0, μ.ρqʳ[i, j, k])

            𝕎ʳᵏ = kessler_rain_terminal_velocity(i, j, k, microphysics, density, coupling_density,
                                                 dry_air_coupled, ρ₁, ρqᵛ, μ)
            μ.𝕎ʳ[i, j, k] = 𝕎ʳᵏ

            # Rain leaves cell k through its bottom face at 𝕎ʳᵏ: the substep may not empty
            # more than a `cfl` fraction of the cell's thickness (every cell, including the top)
            max_Δt = min(max_Δt, cfl * Δzᶜᶜᶜ(i, j, k, grid) / 𝕎ʳᵏ)
        end
    end

    # Subcycling for CFL constraint on rain sedimentation
    Ns = max(1, ceil(Int, Δt / max_Δt))
    inv_Ns = inv(FT(Ns))
    Δtₛ = Δt * inv_Ns
    # Fˢᵘʳᶠ: accumulated surface rain mass flux over subcycles.
    Fˢᵘʳᶠ = zero(FT)

    #####
    ##### PHASE 2: Subcycle sedimentation and microphysics (in partial-density space)
    #####

    for m = 1:Ns
        # The bottom-face flux of cell 1, with the same product cell 1 uses for its outflow
        @inbounds Fˢᵘʳᶠ += μ.ρqʳ[i, j, 1] * μ.𝕎ʳ[i, j, 1]

        for k = 1:(Nz-1)
            # Inflow through the top face: the cell above has not been updated yet this substep
            @inbounds Fᵗᵒᵖ = μ.ρqʳ[i, j, k+1] * μ.𝕎ʳ[i, j, k+1]
            kessler_column_cell!(i, j, k, grid, Fᵗᵒᵖ, Δtₛ, microphysics,
                                 density, coupling_density, dry_air_coupled,
                                 pressure, pˢᵗ, constants, δT, θˡⁱ, ρθˡⁱ, ρqᵛ, μ)
        end

        # k = Nz: no rain enters through the model top
        kessler_column_cell!(i, j, Nz, grid, zero(FT), Δtₛ, microphysics,
                             density, coupling_density, dry_air_coupled,
                             pressure, pˢᵗ, constants, δT, θˡⁱ, ρθˡⁱ, ρqᵛ, μ)

        # Update terminal velocities for next subcycle
        if m < Ns
            for k = 1:Nz
                @inbounds μ.𝕎ʳ[i, j, k] = kessler_rain_terminal_velocity(i, j, k, microphysics, density,
                                                                         coupling_density, dry_air_coupled,
                                                                         ρ₁, ρqᵛ, μ)
            end
        end
    end

    # Surface precipitation rate: the substep-mean surface mass flux per unit of the final
    # surface density, so that `surface_precipitation_flux` (ρ × rate) recovers the flux exactly.
    @inbounds begin
        ρᵗ₁ = ρqᵛ[i, j, 1] + μ.ρqᶜˡ[i, j, 1] + μ.ρqʳ[i, j, 1]
        final_surface_density = ifelse(dry_air_coupled, coupling_density[i, j, 1] + ρᵗ₁, density[i, j, 1])
        μ.precipitation_rate[i, j, 1] = Fˢᵘʳᶠ * inv_Ns / final_surface_density
    end

    #####
    ##### PHASE 3: diagnostic mass fractions on the final total density
    #####
    # `grid_moisture_fractions` reads these when `update_state!` diagnoses the thermodynamic
    # state, before `update_microphysical_fields!` refreshes them, so they must be current here.

    for k = 1:Nz
        @inbounds begin
            ρᵗ = ρqᵛ[i, j, k] + μ.ρqᶜˡ[i, j, k] + μ.ρqʳ[i, j, k]
            ρ = ifelse(dry_air_coupled, coupling_density[i, j, k] + ρᵗ, density[i, j, k])
            μ.qᵛ[i, j, k]  = ρqᵛ[i, j, k] / ρ
            μ.qᶜˡ[i, j, k] = μ.ρqᶜˡ[i, j, k] / ρ
            μ.qʳ[i, j, k]  = μ.ρqʳ[i, j, k] / ρ
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

    # Convert mass fractions → mixing ratios (shared helper).
    # Parcel model stores total moisture in qᵗ; compute vapor by subtracting condensate.
    qᶜˡ_s = max(0, μ.ρqᶜˡ / ρ)
    qʳ_s = max(0, μ.ρqʳ / ρ)
    qᵛ_s = max(0, state.qᵗ - qᶜˡ_s - qʳ_s)
    rᵛ, rᶜˡ, rʳ = mass_fractions_to_mixing_ratios(qᵛ_s, μ.ρqᶜˡ, μ.ρqʳ, ρ)

    # Temperature from thermodynamic state, and its response to condensation at fixed
    # invariant (static energy or θˡⁱ, whichever the parcel carries)
    T = temperature(𝒰, constants)
    ∂T∂rˡ = phase_change_temperature_slope(𝒰, constants)

    # Saturation adjustment parameters
    δT = constants.saturation_vapor_pressure.liquid_temperature_offset
    FT = typeof(ρ)

    # Core microphysics step (no sedimentation for a parcel)
    rᵛ, rᶜˡ, rʳ, _ = step_kessler_microphysics(rᵛ, rᶜˡ, rʳ, T, ∂T∂rˡ, ρ, p_parcel, Δt,
                                               microphysics, constants, δT, FT)

    # Convert mixing ratios → mass fractions (shared helper)
    _, qᶜˡ, qʳ, qᵗ = mixing_ratios_to_mass_fractions(rᵛ, rᶜˡ, rʳ)

    # Update parcel state (parcel model stores total moisture in qᵗ)
    state.μ = (; ρqᶜˡ = ρ * qᶜˡ, ρqʳ = ρ * qʳ)
    state.qᵗ = qᵗ
    state.ρqᵗ = ρ * qᵗ

    # Update thermodynamic state with new moisture fractions. The parcel's invariant (static
    # energy or θˡⁱ) is conserved by the phase change; latent heating is implicit in it.
    rˡ = rᶜˡ + rʳ
    r = MoistureMixingRatio(rᵛ, rˡ)
    q = MoistureMassFractions(r)
    state.𝒰 = with_moisture(𝒰, q)

    # Keep static energy consistent with the thermodynamic state.
    state.ℰ = state.𝒰.static_energy
    state.ρℰ = ρ * state.ℰ

    return nothing
end
