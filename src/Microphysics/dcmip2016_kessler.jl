using ..Thermodynamics:
    MoistureMassFractions,
    MoistureMixingRatio,
    mixture_heat_capacity,
    mixture_gas_constant,
    total_mixing_ratio,
    total_specific_moisture,
    saturation_specific_humidity,
    PlanarLiquidSurface

using ..AtmosphereModels:
    dynamics_density,
    dynamics_pressure,
    surface_pressure

using Oceananigans: CenterField, Field
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, Center
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

using Oceananigans.AbstractOperations: KernelFunctionOperation

using Adapt: Adapt, adapt

using DocStringExtensions: TYPEDSIGNATURES

"""
    struct DCMIP2016KesslerMicrophysics

DCMIP2016 implementation of the Kessler (1969) warm-rain bulk microphysics scheme.

This implementation follows the DCMIP2016 test case specification, which is based on
Klemp and Wilhelmson (1978).

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

# Parameters

## Saturation (Tetens/Clausius-Clapeyron formula)
- `dcmip_temperature_scale`: A parameter of uncertain provenance that appears in the DCMIP2016 implementation
                             of the Kessler scheme (line 105 of https://gitlab.in2p3.fr/ipsl/projets/dynamico/dynamico/-/blob/master/src/dcmip2016_kessler_physic.f90)

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
- `ρ₀`: Density at z=0

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
Base.@kwdef struct DCMIP2016KesslerMicrophysics{FT}
    # DCMIP2016 parameter (appears to be related to Tetens' saturation vapor pressure formula,
    # but cannot be reconciled with other parameters in a consistent application of that formula.)
    dcmip_temperature_scale :: FT = 237.3

    # Rain terminal velocity (Klemp & Wilhelmson 1978)
    terminal_velocity_coefficient :: FT = 36.34
    density_scale                 :: FT = 0.001
    terminal_velocity_exponent    :: FT = 0.1364

    # Autoconversion
    autoconversion_rate      :: FT = 0.001
    autoconversion_threshold :: FT = 0.001

    # Accretion
    accretion_rate     :: FT = 2.2
    accretion_exponent :: FT = 0.875

    # Rain evaporation (Klemp & Wilhelmson 1978)
    evaporation_ventilation_coefficient_1 :: FT = 1.6
    evaporation_ventilation_coefficient_2 :: FT = 124.9
    evaporation_ventilation_exponent_1    :: FT = 0.2046
    evaporation_ventilation_exponent_2    :: FT = 0.525
    diffusivity_coefficient               :: FT = 2.55e8
    thermal_conductivity_coefficient      :: FT = 5.4e5

    # Numerical
    substep_cfl :: FT = 0.8
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

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions at grid point `(i, j, k)` for the thermodynamic state.

Water vapor is diagnosed as \$q^v = q^t - q^{cl} - q^r\$.
Returns `MoistureMassFractions(qᵛ, qˡ)` where \$q^l = q^{cl} + q^r\$ is the total liquid mass fraction.
"""
@inline function AtmosphereModels.compute_moisture_fractions(i, j, k, grid, ::DCMIP2016KM, ρ, qᵗ, μ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ  = μ.ρqʳ[i, j, k] / ρ
    end
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

"""
$(TYPEDSIGNATURES)

Return the thermodynamic state without adjustment.

The Kessler scheme performs its own saturation adjustment internally via the kernel.
"""
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(i, j, k, 𝒰, ::DCMIP2016KM, ρᵣ, μ, qᵗ, constants) = 𝒰

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
@inline AtmosphereModels.microphysical_tendency(i, j, k, grid, ::DCMIP2016KM, name, ρ, μ, 𝒰, constants) = zero(grid)

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
    surface = PlanarLiquidSurface()

    # Extract microphysical fields from μ
    precipitation_rate_field = μ.precipitation_rate

    # Latent heat of vaporization for θˡⁱ ↔ T conversion
    ℒˡᵣ = constants.liquid.reference_latent_heat

    # Dry air heat capacity for latent heating calculation
    cᵖᵈ = constants.dry_air.heat_capacity

    # Saturation adjustment coefficient: f₅ = a × T_DCMIP2016 × ℒˡᵣ / cᵖᵈ
    T_DCMIP2016 = microphysics.dcmip_temperature_scale
    f₅ = saturation_adjustment_coefficient(T_DCMIP2016, constants)

    # Temperature offset for saturation adjustment (from TetensFormula)
    δT = constants.saturation_vapor_pressure.liquid_temperature_offset

    # CFL safety factor for sedimentation
    cfl = microphysics.substep_cfl

    # Parameters from microphysics struct (hoisted out of the inner vertical loops)
    Cᵨ     = microphysics.density_scale
    Cᵉᵛ₁   = microphysics.evaporation_ventilation_coefficient_1
    Cᵉᵛ₂   = microphysics.evaporation_ventilation_coefficient_2
    βᵉᵛ₁   = microphysics.evaporation_ventilation_exponent_1
    βᵉᵛ₂   = microphysics.evaporation_ventilation_exponent_2
    Cᵈⁱᶠᶠ  = microphysics.diffusivity_coefficient
    Cᵗʰᵉʳᵐ = microphysics.thermal_conductivity_coefficient

    # Reference density at surface for terminal velocity (KW eq. 2.15)
    @inbounds ρ₁ = density[i, j, 1]

    #####
    ##### PHASE 1: Convert mass fraction → mixing ratio
    #####

    max_Δt = Δt

    # Avoid a branch in the vertical loop and cut down `znode` calls:
    # we only need `Δz` for k = 1:Nz-1.
    zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())
    for k = 1:(Nz-1)
        @inbounds begin
            ρ = density[i, j, k]

            qᵗ = ρqᵗ[i, j, k] / ρ
            qᶜˡ = max(0, μ.ρqᶜˡ[i, j, k] / ρ)
            qʳ  = max(0, μ.ρqʳ[i, j, k] / ρ)
            qˡ_sum = qᶜˡ + qʳ
            qᵗ = max(qᵗ, qˡ_sum)  # Prevent negative vapor
            qᵛ = qᵗ - qˡ_sum       # Diagnose vapor

            # Convert to mixing ratios for Kessler physics
            q = MoistureMassFractions(qᵛ, qˡ_sum)
            r = MoistureMixingRatio(q)
            rᵛ = r.vapor
            rᵗ = total_mixing_ratio(r)
            rᶜˡ = qᶜˡ * (1 + rᵗ)
            rʳ  = qʳ * (1 + rᵗ)

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

    # k = Nz (no `Δz` / CFL update needed)
    @inbounds begin
        ρ = density[i, j, Nz]

        qᵗ = ρqᵗ[i, j, Nz] / ρ
        qᶜˡ = max(0, μ.ρqᶜˡ[i, j, Nz] / ρ)
        qʳ  = max(0, μ.ρqʳ[i, j, Nz] / ρ)
        qˡ_sum = qᶜˡ + qʳ
        qᵗ = max(qᵗ, qˡ_sum)
        qᵛ = qᵗ - qˡ_sum

        q = MoistureMassFractions(qᵛ, qˡ_sum)
        r = MoistureMixingRatio(q)
        rᵛ = r.vapor
        rᵗ = total_mixing_ratio(r)
        rᶜˡ = qᶜˡ * (1 + rᵗ)
        rʳ  = qʳ * (1 + rᵗ)

        μ.𝕎ʳ[i, j, Nz] = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)

        μ.qᵛ[i, j, Nz]  = rᵛ
        μ.qᶜˡ[i, j, Nz] = rᶜˡ
        μ.qʳ[i, j, Nz]  = rʳ
    end

    # Subcycling for CFL constraint on rain sedimentation
    Ns = max(1, ceil(Int, Δt / max_Δt))
    inv_Ns = inv(FT(Ns))  # Precompute for final averaging
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

        # Rolling z-coordinate to reduce `znode` calls (and avoid a branch in the loop body)
        zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())
        for k = 1:(Nz-1)
            @inbounds begin
                ρ = density[i, j, k]
                p = pressure[i, j, k]
                θˡⁱᵏ = θˡⁱ[i, j, k]

                rᵛ = μ.qᵛ[i, j, k]
                rᶜˡ = μ.qᶜˡ[i, j, k]
                rʳ = μ.qʳ[i, j, k]

                # Moist thermodynamics using mixing ratio abstraction
                rˡ = rᶜˡ + rʳ
                r = MoistureMixingRatio(rᵛ, rˡ)
                cᵖᵐ = mixture_heat_capacity(r, constants)
                Rᵐ  = mixture_gas_constant(r, constants)
                q = MoistureMassFractions(r)
                qˡ_current = q.liquid
                Π = (p / p₀)^(Rᵐ / cᵖᵐ)
                Tᵏ = Π * θˡⁱᵏ + ℒˡᵣ * qˡ_current / cᵖᵐ

                # Rain sedimentation (upstream differencing)
                ρᵏ = Cᵨ * ρ
                𝕎ʳᵏ = μ.𝕎ʳ[i, j, k]

                zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Center())
                Δz = zᵏ⁺¹ - zᵏ

                ρᵏ⁺¹ = density[i, j, k+1]
                ρᵏ⁺¹ = Cᵨ * ρᵏ⁺¹
                rʳᵏ⁺¹ = μ.qʳ[i, j, k+1]  # Mixing ratio
                𝕎ʳᵏ⁺¹ = μ.𝕎ʳ[i, j, k+1]

                # Δr𝕎: change in rain mixing ratio due to sedimentation (upstream differencing)
                Δr𝕎 = Δtₛ * (ρᵏ⁺¹ * rʳᵏ⁺¹ * 𝕎ʳᵏ⁺¹ - ρᵏ * rʳ * 𝕎ʳᵏ) / (ρᵏ * Δz)
                zᵏ = zᵏ⁺¹

                # Δrᴾ: cloud-to-rain production from autoconversion + accretion (KW eq. 2.13)
                Δrᴾ = cloud_to_rain_production(rᶜˡ, rʳ, Δtₛ, microphysics)
                rᶜˡ_new = max(0, rᶜˡ - Δrᴾ)
                rʳ_new = max(0, rʳ + Δrᴾ + Δr𝕎)

                # Saturation specific humidity using Breeze thermodynamics
                # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) is the saturation mass fraction
                qᵛ⁺ = saturation_specific_humidity(Tᵏ, ρ, constants, surface)
                # Convert to saturation mixing ratio: rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
                rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

                # Δrˢᵃᵗ: mixing ratio adjustment to restore saturation equilibrium
                δT = constants.saturation_vapor_pressure.liquid_temperature_offset
                Δrˢᵃᵗ = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (Tᵏ - δT)^2)

                # Δrᴱ: rain evaporation into subsaturated air (KW eq. 2.14)
                ρrʳ = ρᵏ * rʳ_new                                  # Scaled rain water content
                Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂          # Ventilation factor
                Dᵗʰ = Cᵈⁱᶠᶠ / (p * rᵛ⁺) + Cᵗʰᵉʳᵐ                   # Diffusion-thermal term
                Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)                            # Subsaturation
                Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (ρᵏ * rᵛ⁺ + FT(1e-20))     # Rain evaporation rate
                Δrᴱmax = max(0, -Δrˢᵃᵗ - rᶜˡ_new)                   # Maximum evaporation
                Δrᴱ = min(min(Δtₛ * Ėʳ, Δrᴱmax), rʳ_new)            # Limited evaporation

                # Δrᶜ: condensation of vapor to cloud liquid (limited by available cloud water)
                Δrᶜ = max(Δrˢᵃᵗ, -rᶜˡ_new)
                rᵛ_new = max(0, rᵛ - Δrᶜ + Δrᴱ)
                rᶜˡ_final = rᶜˡ_new + Δrᶜ
                rʳ_final = rʳ_new - Δrᴱ

                μ.qᵛ[i, j, k]  = rᵛ_new
                μ.qᶜˡ[i, j, k] = rᶜˡ_final
                μ.qʳ[i, j, k]  = rʳ_final

                # Update θˡⁱ from latent heating
                # Uses Breeze's thermodynamic constants for consistency
                net_phase_change = Δrᶜ - Δrᴱ
                ΔT_phase = ℒˡᵣ / cᵖᵈ * net_phase_change
                T_new = Tᵏ + ΔT_phase

                # Convert back to θˡⁱ with updated moisture
                rˡ_new = rᶜˡ_final + rʳ_final
                r_new = MoistureMixingRatio(rᵛ_new, rˡ_new)
                cᵖᵐ_new = mixture_heat_capacity(r_new, constants)
                Rᵐ_new  = mixture_gas_constant(r_new, constants)
                q_new = MoistureMassFractions(r_new)
                qˡ_new = q_new.liquid
                Π_new = (p / p₀)^(Rᵐ_new / cᵖᵐ_new)

                # θˡⁱ = (T - ℒˡᵣ qˡ / cᵖᵐ) / Π
                θˡⁱ_new = (T_new - ℒˡᵣ * qˡ_new / cᵖᵐ_new) / Π_new

                θˡⁱ[i, j, k]  = θˡⁱ_new
                ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
            end
        end

        # k = Nz (top boundary: rain falls out)
        @inbounds begin
            k = Nz
            ρ = density[i, j, k]
            p = pressure[i, j, k]
            θˡⁱᵏ = θˡⁱ[i, j, k]

            rᵛ = μ.qᵛ[i, j, k]
            rᶜˡ = μ.qᶜˡ[i, j, k]
            rʳ = μ.qʳ[i, j, k]

            # Moist thermodynamics using mixing ratio abstraction
            rˡ = rᶜˡ + rʳ
            r = MoistureMixingRatio(rᵛ, rˡ)
            cᵖᵐ = mixture_heat_capacity(r, constants)
            Rᵐ  = mixture_gas_constant(r, constants)
            q = MoistureMassFractions(r)
            qˡ_current = q.liquid
            Π = (p / p₀)^(Rᵐ / cᵖᵐ)
            Tᵏ = Π * θˡⁱᵏ + ℒˡᵣ * qˡ_current / cᵖᵐ

            # Δr𝕎: sedimentation at top boundary (rain falls out of domain)
            ρᵏ = Cᵨ * ρ
            𝕎ʳᵏ = μ.𝕎ʳ[i, j, k]
            zᵏ = znode(i, j, k, grid, Center(), Center(), Center())
            zᵏ⁻¹ = znode(i, j, k-1, grid, Center(), Center(), Center())
            Δz_half = (zᵏ - zᵏ⁻¹) / 2
            Δr𝕎 = -Δtₛ * rʳ * 𝕎ʳᵏ / Δz_half

            # Δrᴾ: cloud-to-rain production (KW eq. 2.13)
            Δrᴾ = cloud_to_rain_production(rᶜˡ, rʳ, Δtₛ, microphysics)
            rᶜˡ_new = max(0, rᶜˡ - Δrᴾ)
            rʳ_new = max(0, rʳ + Δrᴾ + Δr𝕎)

            # Δrˢᵃᵗ: saturation adjustment
            qᵛ⁺ = saturation_specific_humidity(Tᵏ, ρ, constants, surface)
            rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
            Δrˢᵃᵗ = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (Tᵏ - δT)^2)

            # Δrᴱ: rain evaporation (KW eq. 2.14)
            ρrʳ = ρᵏ * rʳ_new                                          # Scaled rain water content
            Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂                 # Ventilation factor
            Dᵗʰ = Cᵈⁱᶠᶠ / (p * rᵛ⁺) + Cᵗʰᵉʳᵐ                          # Diffusion-thermal term
            Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)                                    # Subsaturation
            Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (ρᵏ * rᵛ⁺ + FT(1e-20))            # Rain evaporation rate
            Δrᴱmax = max(0, -Δrˢᵃᵗ - rᶜˡ_new)                          # Maximum evaporation
            Δrᴱ = min(min(Δtₛ * Ėʳ, Δrᴱmax), rʳ_new)                   # Limited evaporation

            # Δrᶜ: condensation
            Δrᶜ = max(Δrˢᵃᵗ, -rᶜˡ_new)
            rᵛ_new = max(0, rᵛ - Δrᶜ + Δrᴱ)
            rᶜˡ_final = rᶜˡ_new + Δrᶜ
            rʳ_final = rʳ_new - Δrᴱ

            μ.qᵛ[i, j, k]  = rᵛ_new
            μ.qᶜˡ[i, j, k] = rᶜˡ_final
            μ.qʳ[i, j, k]  = rʳ_final

            net_phase_change = Δrᶜ - Δrᴱ
            ΔT_phase = ℒˡᵣ / cᵖᵈ * net_phase_change
            T_new = Tᵏ + ΔT_phase

            rˡ_new = rᶜˡ_final + rʳ_final
            r_new = MoistureMixingRatio(rᵛ_new, rˡ_new)
            cᵖᵐ_new = mixture_heat_capacity(r_new, constants)
            Rᵐ_new  = mixture_gas_constant(r_new, constants)
            q_new = MoistureMassFractions(r_new)
            qˡ_new = q_new.liquid
            Π_new = (p / p₀)^(Rᵐ_new / cᵖᵐ_new)

            θˡⁱ_new = (T_new - ℒˡᵣ * qˡ_new / cᵖᵐ_new) / Π_new

            θˡⁱ[i, j, k]  = θˡⁱ_new
            ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
        end

        # Recalculate terminal velocities for next subcycle
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

            # Convert mixing ratios to mass fractions
            rˡ = rᶜˡ + rʳ
            r = MoistureMixingRatio(rᵛ, rˡ)
            q = MoistureMassFractions(r)
            qᵛ = q.vapor
            qˡ = q.liquid
            qᵗ = total_specific_moisture(q)

            # Compute cloud and rain mass fractions using the same conversion factor
            rᵗ = total_mixing_ratio(r)
            qᶜˡ = rᶜˡ / (1 + rᵗ)
            qʳ  = rʳ / (1 + rᵗ)

            # Update prognostic fields (density-weighted)
            ρqᵗ[i, j, k]  = ρ * qᵗ
            μ.ρqᶜˡ[i, j, k] = ρ * qᶜˡ
            μ.ρqʳ[i, j, k]  = ρ * qʳ

            # Update diagnostic fields (mass fractions)
            μ.qᵛ[i, j, k]  = qᵛ
            μ.qᶜˡ[i, j, k] = qᶜˡ
            μ.qʳ[i, j, k]  = qʳ
        end
    end
end

#####
##### Diagnostic field update
#####

# Update diagnostic mass fraction fields from prognostic density-weighted fields
@inline function AtmosphereModels.update_microphysical_fields!(μ, ::DCMIP2016KM, i, j, k, grid, ρ, 𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    @inbounds begin
        μ.qᶜˡ[i, j, k] = μ.ρqᶜˡ[i, j, k] / ρ
        μ.qʳ[i, j, k]  = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k]  = qᵗ - μ.qᶜˡ[i, j, k] - μ.qʳ[i, j, k]
    end
    return nothing
end
