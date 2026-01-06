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

using Oceananigans: CenterField, Field, interior
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
- `f₂ₓ`: Clausius-Clapeyron exponent coefficient (default: 17.27)
- `T_f`: Clausius-Clapeyron denominator coefficient in K (default: 237.3)
- `T_offset`: Temperature offset in saturation adjustment in K (default: 36.0)

The saturation adjustment coefficient is computed as `f₅ = T_f × f₂ₓ × ℒˡᵣ / cᵖᵈ`.

## Rain Terminal Velocity (Klemp & Wilhelmson 1978, eq. 2.15)
Terminal velocity: `vᵗ = a_vᵗ × (ρ × rʳ × ρ_scale)^β_vᵗ × √(ρ₀/ρ)`
- `a_vᵗ`: Terminal velocity coefficient in m/s (default: 36.34)
- `ρ_scale`: Density scale factor for unit conversion (default: 0.001)
- `β_vᵗ`: Terminal velocity exponent (default: 0.1364)
- `ρ`: Density
- `ρ₀`: Density at z=0

## Autoconversion
- `k₁`: Autoconversion rate coefficient in s⁻¹ (default: 0.001)
- `rᶜˡ★`: Critical cloud water mixing ratio threshold in kg/kg (default: 0.001)

## Accretion
- `k₂`: Accretion rate coefficient in s⁻¹ (default: 2.2)
- `β_acc`: Accretion exponent for rain mixing ratio (default: 0.875)

## Rain Evaporation (Klemp & Wilhelmson 1978, eq. 2.14)
Ventilation: `(Cᵉᵛ₁ + Cᵉᵛ₂ × (ρ rʳ)^βᵉᵛ₁) × (ρ rʳ)^βᵉᵛ₂`
- `Cᵉᵛ₁`: Evaporation ventilation coefficient 1 (default: 1.6)
- `Cᵉᵛ₂`: Evaporation ventilation coefficient 2 (default: 124.9)
- `βᵉᵛ₁`: Evaporation ventilation exponent 1 (default: 0.2046)
- `βᵉᵛ₂`: Evaporation ventilation exponent 2 (default: 0.525)
- `Cᵈⁱᶠᶠ`: Diffusivity-related denominator coefficient (default: 2.55e8)
- `Cᵗʰᵉʳᵐ`: Thermal conductivity-related denominator coefficient (default: 5.4e5)

## Numerical
- `substep_cfl`: CFL safety factor for sedimentation subcycling (default: 0.8)
"""
Base.@kwdef struct DCMIP2016KesslerMicrophysics{FT}
    # Saturation (Tetens/Clausius-Clapeyron)
    f₂ₓ      :: FT = 17.27
    T_f      :: FT = 237.3
    T_offset :: FT = 36.0

    # Rain terminal velocity (Klemp & Wilhelmson 1978)
    a_vᵗ    :: FT = 36.34
    ρ_scale :: FT = 0.001
    β_vᵗ    :: FT = 0.1364

    # Autoconversion
    k₁      :: FT = 0.001
    rᶜˡ★ :: FT = 0.001

    # Accretion
    k₂    :: FT = 2.2
    β_acc :: FT = 0.875

    # Rain evaporation (Klemp & Wilhelmson 1978)
    Cᵉᵛ₁   :: FT = 1.6
    Cᵉᵛ₂   :: FT = 124.9
    βᵉᵛ₁   :: FT = 0.2046
    βᵉᵛ₂   :: FT = 0.525
    Cᵈⁱᶠᶠ  :: FT = 2.55e8
    Cᵗʰᵉʳᵐ :: FT = 5.4e5

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
- `𝕍ʳ`: Rain terminal velocity (\$m/s\$).
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
    𝕍ʳ = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, precipitation_rate, 𝕍ʳ)
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

#####
##### Mass fraction ↔ mixing ratio conversion
#####

"""
    mass_fraction_to_mixing_ratio(q, qᵗ)

Convert mass fraction `q` to mixing ratio: `r = q / (1 - qᵗ)`.
"""
@inline mass_fraction_to_mixing_ratio(q, qᵗ) = q / (1 - qᵗ)

"""
    mixing_ratio_to_mass_fraction(r, rᵗ)

Convert mixing ratio `r` to mass fraction: `q = r / (1 + rᵗ)`.
"""
@inline mixing_ratio_to_mass_fraction(r, rᵗ) = r / (1 + rᵗ)


"""
$(TYPEDSIGNATURES)

Compute rain terminal velocity (m/s) following Klemp and Wilhelmson (1978) eq. 2.15.

The terminal velocity is computed as:
```math
vᵗ = a_{vᵗ} × (ρ × rʳ × ρ_{scale})^{β_{vᵗ}} × \\sqrt{ρ₀/ρ}
```

where the parameters `a_vᵗ`, `ρ_scale`, and `β_vᵗ` are taken from the `microphysics` struct.
"""
@inline function kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
    a_vᵗ    = microphysics.a_vᵗ
    ρ_scale = microphysics.ρ_scale
    β_vᵗ    = microphysics.β_vᵗ
    rhalf = sqrt(ρ₁ / ρ)
    return a_vᵗ * (rʳ * ρ_scale * ρ)^β_vᵗ * rhalf
end

"""
    cloud_to_rain_production(rᶜˡ, rʳ, Δt, k₁, k₂, rᶜˡ★, β_acc, FT)

Compute cloud-to-rain production rate from autoconversion and accretion (Klemp & Wilhelmson 1978, eq. 2.13).

This implements the combined effect of:
- **Autoconversion**: Cloud water spontaneously converting to rain when `rᶜˡ > rᶜˡ★`
- **Accretion**: Rain collecting cloud water as it falls

The formula uses an implicit time integration for numerical stability.
"""
@inline function cloud_to_rain_production(rᶜˡ, rʳ, Δt, k₁, k₂, rᶜˡ★, β_acc, FT)
    Aʳ = max(0, k₁ * (rᶜˡ - rᶜˡ★))  # Autoconversion rate
    denom = 1 + Δt * k₂ * rʳ^β_acc             # Implicit accretion factor
    Pʳ = rᶜˡ - (rᶜˡ - Δt * Aʳ) / denom
    return Pʳ
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

@kernel function _microphysical_update!(microphysics, grid, Nz, Δt, ρ_field, p_field, p₀, constants, θˡⁱ, ρθˡⁱ, ρqᵗ, μ)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    surface = PlanarLiquidSurface()

    # Extract microphysical fields from μ
    ρqᶜˡ = μ.ρqᶜˡ
    ρqʳ = μ.ρqʳ
    qᵛ_field = μ.qᵛ
    qᶜˡ_field = μ.qᶜˡ
    qʳ_field = μ.qʳ
    precipitation_rate = μ.precipitation_rate
    𝕍ʳ = μ.𝕍ʳ

    # Latent heat of vaporization for θˡⁱ ↔ T conversion
    ℒˡᵣ = constants.liquid.reference_latent_heat

    # Dry air heat capacity for latent heating calculation
    cᵖᵈ = constants.dry_air.heat_capacity
    inv_cᵖᵈ = inv(cᵖᵈ)  # Precompute inverse for efficiency

    # Get scheme-specific parameters from microphysics struct
    f₂ₓ      = microphysics.f₂ₓ
    T_f      = microphysics.T_f
    T_offset = microphysics.T_offset

    # Compute f₅ = T_f × f₂ₓ × ℒˡᵣ / cᵖᵈ (saturation adjustment coefficient)
    f₅ = T_f * f₂ₓ * ℒˡᵣ * inv_cᵖᵈ

    # CFL safety factor for sedimentation
    substep_cfl = microphysics.substep_cfl

    # Precompute latent heating factor
    ℒˡᵣ_over_cᵖᵈ = ℒˡᵣ * inv_cᵖᵈ

    # Parameters from microphysics struct (hoisted out of the inner vertical loops)
    ρ_scale = microphysics.ρ_scale

    k₁      = microphysics.k₁
    rᶜˡ★    = microphysics.rᶜˡ★
    k₂      = microphysics.k₂
    β_acc   = microphysics.β_acc

    Cᵉᵛ₁   = microphysics.Cᵉᵛ₁
    Cᵉᵛ₂   = microphysics.Cᵉᵛ₂
    βᵉᵛ₁   = microphysics.βᵉᵛ₁
    βᵉᵛ₂   = microphysics.βᵉᵛ₂
    Cᵈⁱᶠᶠ  = microphysics.Cᵈⁱᶠᶠ
    Cᵗʰᵉʳᵐ = microphysics.Cᵗʰᵉʳᵐ

    # Reference density at surface for terminal velocity (KW eq. 2.15)
    @inbounds ρ₁ = ρ_field[i, j, 1]

    #####
    ##### PHASE 1: Convert mass fraction → mixing ratio
    #####

    max_Δt = Δt

    # Avoid a branch in the vertical loop and cut down `znode` calls:
    # we only need `Δz` for k = 1:Nz-1.
    zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())
    for k = 1:(Nz-1)
        @inbounds begin
            ρ = ρ_field[i, j, k]
            inv_ρ = inv(ρ)  # Precompute inverse density

            qᵗ = ρqᵗ[i, j, k] * inv_ρ
            qᶜˡ = max(0, ρqᶜˡ[i, j, k] * inv_ρ)
            qʳ  = max(0, ρqʳ[i, j, k] * inv_ρ)
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

            𝕍ʳᵏ = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
            𝕍ʳ[i, j, k] = 𝕍ʳᵏ

            # Store mixing ratios in diagnostic fields during physics
            qᵛ_field[i, j, k]  = rᵛ
            qᶜˡ_field[i, j, k] = rᶜˡ
            qʳ_field[i, j, k]  = rʳ

            # CFL check for sedimentation
            zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Center())
            Δz = zᵏ⁺¹ - zᵏ
            max_Δt = min(max_Δt, substep_cfl * Δz / 𝕍ʳᵏ)
            zᵏ = zᵏ⁺¹
        end
    end

    # k = Nz (no `Δz` / CFL update needed)
    @inbounds begin
        ρ = ρ_field[i, j, Nz]
        inv_ρ = inv(ρ)

        qᵗ = ρqᵗ[i, j, Nz] * inv_ρ
        qᶜˡ = max(0, ρqᶜˡ[i, j, Nz] * inv_ρ)
        qʳ  = max(0, ρqʳ[i, j, Nz] * inv_ρ)
        qˡ_sum = qᶜˡ + qʳ
        qᵗ = max(qᵗ, qˡ_sum)
        qᵛ = qᵗ - qˡ_sum

        q = MoistureMassFractions(qᵛ, qˡ_sum)
        r = MoistureMixingRatio(q)
        rᵛ = r.vapor
        rᵗ = total_mixing_ratio(r)
        rᶜˡ = qᶜˡ * (1 + rᵗ)
        rʳ  = qʳ * (1 + rᵗ)

        velqr = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
        𝕍ʳ[i, j, Nz] = velqr

        qᵛ_field[i, j, Nz]  = rᵛ
        qᶜˡ_field[i, j, Nz] = rᶜˡ
        qʳ_field[i, j, Nz]  = rʳ
    end

    # Subcycling for CFL constraint on rain sedimentation
    Ns = max(1, ceil(Int, Δt / max_Δt))
    inv_Ns = inv(FT(Ns))  # Precompute for final averaging
    Δtₛ = Δt * inv_Ns
    precip_accum = zero(FT)  # Local accumulator to reduce global memory writes

    #####
    ##### PHASE 2: Subcycle microphysics (in mixing ratio space)
    #####

    for m = 1:Ns

        # Accumulate surface precipitation (qʳ × vᵗ)
        @inbounds begin
            rᵛ₁ = qᵛ_field[i, j, 1]
            rᶜˡ₁ = qᶜˡ_field[i, j, 1]
            rʳ₁ = qʳ_field[i, j, 1]
            rᵗ₁ = rᵛ₁ + rᶜˡ₁ + rʳ₁
            # qʳ = rʳ / (1 + rᵗ)
            qʳ₁ = rʳ₁ / (1 + rᵗ₁)
            precip_accum += qʳ₁ * 𝕍ʳ[i, j, 1]
        end

        # Rolling z-coordinate to reduce `znode` calls (and avoid a branch in the loop body)
        zᵏ = znode(i, j, 1, grid, Center(), Center(), Center())
        for k = 1:(Nz-1)
            @inbounds begin
                ρ = ρ_field[i, j, k]
                p = p_field[i, j, k]
                θˡⁱᵏ = θˡⁱ[i, j, k]

                rᵛ = qᵛ_field[i, j, k]
                rᶜˡ = qᶜˡ_field[i, j, k]
                rʳ = qʳ_field[i, j, k]

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
                rᵏ = ρ_scale * ρ
                𝕍ʳᵏ = 𝕍ʳ[i, j, k]

                zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Center())
                Δz = zᵏ⁺¹ - zᵏ

                ρᵏ⁺¹ = ρ_field[i, j, k+1]
                rᵏ⁺¹ = ρ_scale * ρᵏ⁺¹
                rʳᵏ⁺¹ = qʳ_field[i, j, k+1]  # Mixing ratio
                𝕍ʳᵏ⁺¹ = 𝕍ʳ[i, j, k+1]

                sed = Δtₛ * (rᵏ⁺¹ * rʳᵏ⁺¹ * 𝕍ʳᵏ⁺¹ - rᵏ * rʳ * 𝕍ʳᵏ) / (rᵏ * Δz)
                zᵏ = zᵏ⁺¹

                # Autoconversion + accretion (KW eq. 2.13)
                # Pʳ is the cloud-to-rain production from autoconversion and accretion
                Pʳ = cloud_to_rain_production(rᶜˡ, rʳ, Δtₛ, k₁, k₂, rᶜˡ★, β_acc, FT)
                rᶜˡ_new = max(0, rᶜˡ - Pʳ)
                rʳ_new = max(0, rʳ + Pʳ + sed)

                # Saturation specific humidity using Breeze thermodynamics
                # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) is the saturation mass fraction
                qᵛ⁺ = saturation_specific_humidity(Tᵏ, ρ, constants, surface)
                # Convert to saturation mixing ratio: rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
                rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

                # Saturation adjustment
                prod = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (Tᵏ - T_offset)^2)

                # Rain evaporation (KW eq. 2.14)
                ρrʳ = rᵏ * rʳ_new                                        # Scaled rain water content
                Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂               # Ventilation factor
                Dᵗʰ = Cᵈⁱᶠᶠ / (p * rᵛ⁺) + Cᵗʰᵉʳᵐ                        # Diffusion-thermal term
                Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)                                  # Subsaturation
                Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (rᵏ * rᵛ⁺ + FT(1e-20))          # Rain evaporation rate
                Eʳₘₐₓ = max(0, -prod - rᶜˡ_new)                          # Maximum evaporation
                Eʳ = min(min(Δtₛ * Ėʳ, Eʳₘₐₓ), rʳ_new)                   # Limited evaporation

                # Apply adjustments
                condensation = max(prod, -rᶜˡ_new)
                rᵛ_new = max(0, rᵛ - condensation + Eʳ)
                rᶜˡ_final = rᶜˡ_new + condensation
                rʳ_final = rʳ_new - Eʳ

                qᵛ_field[i, j, k]  = rᵛ_new
                qᶜˡ_field[i, j, k] = rᶜˡ_final
                qʳ_field[i, j, k]  = rʳ_final

                # Update θˡⁱ from latent heating
                # Uses Breeze's thermodynamic constants for consistency
                net_phase_change = condensation - Eʳ
                ΔT_phase = ℒˡᵣ_over_cᵖᵈ * net_phase_change
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
            ρ = ρ_field[i, j, k]
            p = p_field[i, j, k]
            θˡⁱᵏ = θˡⁱ[i, j, k]

            rᵛ = qᵛ_field[i, j, k]
            rᶜˡ = qᶜˡ_field[i, j, k]
            rʳ = qʳ_field[i, j, k]

            # Moist thermodynamics using mixing ratio abstraction
            rˡ = rᶜˡ + rʳ
            r = MoistureMixingRatio(rᵛ, rˡ)
            cᵖᵐ = mixture_heat_capacity(r, constants)
            Rᵐ  = mixture_gas_constant(r, constants)
            q = MoistureMassFractions(r)
            qˡ_current = q.liquid
            Π = (p / p₀)^(Rᵐ / cᵖᵐ)
            Tᵏ = Π * θˡⁱᵏ + ℒˡᵣ * qˡ_current / cᵖᵐ

            # Top boundary: rain falls out
            rᵏ = ρ_scale * ρ
            𝕍ʳᵏ = 𝕍ʳ[i, j, k]
            zᵏ = znode(i, j, k, grid, Center(), Center(), Center())
            zᵏ⁻¹ = znode(i, j, k-1, grid, Center(), Center(), Center())
            Δz_half = 0.5 * (zᵏ - zᵏ⁻¹)
            sed = -Δtₛ * rʳ * 𝕍ʳᵏ / Δz_half

            # Autoconversion + accretion (KW eq. 2.13)
            # Pʳ is the cloud-to-rain production from autoconversion and accretion
            Pʳ = cloud_to_rain_production(rᶜˡ, rʳ, Δtₛ, k₁, k₂, rᶜˡ★, β_acc, FT)
            rᶜˡ_new = max(0, rᶜˡ - Pʳ)
            rʳ_new = max(0, rʳ + Pʳ + sed)

            qᵛ⁺ = saturation_specific_humidity(Tᵏ, ρ, constants, surface)
            rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

            prod = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (Tᵏ - T_offset)^2)

            # Rain evaporation (KW eq. 2.14)
            ρrʳ = rᵏ * rʳ_new                                        # Scaled rain water content
            Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂               # Ventilation factor
            Dᵗʰ = Cᵈⁱᶠᶠ / (p * rᵛ⁺) + Cᵗʰᵉʳᵐ                        # Diffusion-thermal term
            Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)                                  # Subsaturation
            Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (rᵏ * rᵛ⁺ + FT(1e-20))          # Rain evaporation rate
            Eʳₘₐₓ = max(0, -prod - rᶜˡ_new)                          # Maximum evaporation
            Eʳ = min(min(Δtₛ * Ėʳ, Eʳₘₐₓ), rʳ_new)                   # Limited evaporation

            condensation = max(prod, -rᶜˡ_new)
            rᵛ_new = max(0, rᵛ - condensation + Eʳ)
            rᶜˡ_final = rᶜˡ_new + condensation
            rʳ_final = rʳ_new - Eʳ

            qᵛ_field[i, j, k]  = rᵛ_new
            qᶜˡ_field[i, j, k] = rᶜˡ_final
            qʳ_field[i, j, k]  = rʳ_final

            net_phase_change = condensation - Eʳ
            ΔT_phase = ℒˡᵣ_over_cᵖᵈ * net_phase_change
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
                    ρ = ρ_field[i, j, k]
                    rʳ = qʳ_field[i, j, k]
                    𝕍ʳ[i, j, k] = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
                end
            end
        end
    end

    @inbounds precipitation_rate[i, j, 1] = precip_accum * inv_Ns

    #####
    ##### PHASE 3: Convert mixing ratio → mass fraction
    #####

    for k = 1:Nz
        @inbounds begin
            ρ = ρ_field[i, j, k]
            rᵛ = qᵛ_field[i, j, k]
            rᶜˡ = qᶜˡ_field[i, j, k]
            rʳ = qʳ_field[i, j, k]

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
            ρqᶜˡ[i, j, k] = ρ * qᶜˡ
            ρqʳ[i, j, k]  = ρ * qʳ

            # Update diagnostic fields (mass fractions)
            qᵛ_field[i, j, k]  = qᵛ
            qᶜˡ_field[i, j, k] = qᶜˡ
            qʳ_field[i, j, k]  = qʳ
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
