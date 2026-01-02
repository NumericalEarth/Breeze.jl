using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant,
    dry_air_gas_constant,
    total_specific_moisture,
    saturation_specific_humidity,
    PlanarLiquidSurface

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
- `f2x :: FT = 17.27`: Clausius-Clapeyron exponent coefficient (empirical constant for saturation)
- `p₀ :: FT = 100000.0`: Reference sea level pressure in Pascals

The saturation adjustment coefficient `f5 = 237.3 × f2x × ℒˡᵣ / cᵖᵈ` is computed dynamically
from `f2x` and the thermodynamic constants.
"""
Base.@kwdef struct DCMIP2016KesslerMicrophysics{FT}
    f2x :: FT = 17.27
    p₀  :: FT = 100000.0
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
- `vᵗ_rain`: Rain terminal velocity (\$m/s\$).
"""
function materialize_microphysical_fields(::DCMIP2016KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted)
    ρqᶜˡ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᶜˡ)
    ρqʳ  = CenterField(grid, boundary_conditions=boundary_conditions.ρqʳ)

    # Diagnostic fields (mass fractions)
    qᵛ  = CenterField(grid)
    qᶜˡ = CenterField(grid)
    qʳ  = CenterField(grid)

    # Precipitation and velocity diagnostics
    precipitation_rate = Field{Center, Center, Nothing}(grid)
    vᵗ_rain = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, precipitation_rate, vᵗ_rain)
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
    kessler_terminal_velocity(rʳ, ρ, ρ_bottom)

Compute rain terminal velocity (m/s) following Klemp and Wilhelmson (1978) eq. 2.15.
"""
@inline function kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
    rhalf = sqrt(ρ_bottom / ρ)
    return 36.34 * (rʳ * 0.001 * ρ)^0.1364 * rhalf
end

#####
##### Main update function - launches GPU kernel
#####

"""
$(TYPEDSIGNATURES)

Apply the Kessler microphysics to the model.

This function launches a kernel that processes each column independently, with rain sedimentation subcycling.

The kernel handles conversion between mass fractions (Breeze) and mixing ratios (Kessler)
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

    # Reference state - use interior() for reduced fields to get GPU-compatible arrays
    ρᵣ = interior(model.dynamics.reference_state.density, 1, 1, :)
    pᵣ = interior(model.dynamics.reference_state.pressure, 1, 1, :)

    # Surface pressure for Exner function
    p₀ = model.dynamics.reference_state.surface_pressure

    # Thermodynamic constants for liquid-ice potential temperature conversion
    constants = model.thermodynamic_constants

    # Thermodynamic fields (liquid-ice potential temperature, NOT regular potential temperature)
    θˡⁱ  = model.formulation.potential_temperature
    ρθˡⁱ = model.formulation.potential_temperature_density

    # Total moisture density (prognostic variable of AtmosphereModel)
    ρqᵗ = model.moisture_density

    # Microphysical fields
    μ = model.microphysical_fields

    # Use interior() for 2D field to avoid GPU indexing issues
    precipitation_rate_data = interior(μ.precipitation_rate, :, :, 1)

    launch!(arch, grid, :xy, _microphysical_update!,
            microphysics, grid, Nz, Δt, ρᵣ, pᵣ, p₀, constants, θˡⁱ, ρθˡⁱ,
            ρqᵗ, μ.ρqᶜˡ, μ.ρqʳ,
            μ.qᵛ, μ.qᶜˡ, μ.qʳ,
            precipitation_rate_data, μ.vᵗ_rain)

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

@kernel function _microphysical_update!(microphysics, grid, Nz, Δt, ρᵣ, pᵣ, p₀, constants, θˡⁱ, ρθˡⁱ,
                                                 ρqᵗ, ρqᶜˡ, ρqʳ,
                                                 qᵛ_field, qᶜˡ_field, qʳ_field,
                                                 precipitation_rate, vᵗ_rain)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    # Latent heat of vaporization for θˡⁱ ↔ T conversion
    ℒˡᵣ = constants.liquid.reference_latent_heat

    # Dry air heat capacity for latent heating calculation
    cᵖᵈ = constants.dry_air.heat_capacity
    inv_cᵖᵈ = inv(cᵖᵈ)  # Precompute inverse for efficiency

    # Compute Kessler scheme constants from Breeze thermodynamic constants
    # κ = Rᵈ / cᵖᵈ (Poisson constant for dry air)
    Rᵈ = dry_air_gas_constant(constants)
    κ = Rᵈ * inv_cᵖᵈ

    # Get scheme-specific parameters from microphysics struct
    f2x = microphysics.f2x
    p₀_kessler = microphysics.p₀

    # Compute f5 = 237.3 × f2x × ℒˡᵣ / cᵖᵈ (saturation adjustment coefficient)
    f5 = 237.3 * f2x * ℒˡᵣ * inv_cᵖᵈ

    # Precompute latent heating factor
    ℒˡᵣ_over_cᵖᵈ = ℒˡᵣ * inv_cᵖᵈ

    # Reference density at surface for terminal velocity (KW eq. 2.15)
    @inbounds ρ_bottom = ρᵣ[1]

    #####
    ##### PHASE 1: Convert mass fraction → mixing ratio
    #####

    dt_max = Δt
    for k = 1:Nz
        @inbounds begin
            ρ = ρᵣ[k]
            inv_ρ = inv(ρ)  # Precompute inverse density

            qᵗ = ρqᵗ[i, j, k] * inv_ρ
            qᶜˡ = max(ρqᶜˡ[i, j, k] * inv_ρ, zero(FT))
            qʳ  = max(ρqʳ[i, j, k] * inv_ρ, zero(FT))
            qˡ_sum = qᶜˡ + qʳ
            qᵗ = max(qᵗ, qˡ_sum)  # Prevent negative vapor
            qᵛ = qᵗ - qˡ_sum       # Diagnose vapor

            # Convert to mixing ratios for Kessler physics
            # Precompute common denominator: 1/(1 - qᵗ)
            inv_one_minus_qᵗ = inv(1 - qᵗ)
            rᵛ = qᵛ * inv_one_minus_qᵗ
            rᶜ = qᶜˡ * inv_one_minus_qᵗ
            rʳ = qʳ * inv_one_minus_qᵗ

            velqr = kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
            vᵗ_rain[i, j, k] = velqr

            # Store mixing ratios in diagnostic fields during physics
            qᵛ_field[i, j, k]  = rᵛ
            qᶜˡ_field[i, j, k] = rᶜ
            qʳ_field[i, j, k]  = rʳ

            # CFL check for sedimentation (fused into main loop to avoid extra znode call)
            if k < Nz
                z_k   = znode(i, j, k, grid, Center(), Center(), Center())
                z_kp1 = znode(i, j, k+1, grid, Center(), Center(), Center())
                dz = z_kp1 - z_k
                if velqr > 0
                    dt_max = min(dt_max, 0.8 * dz / velqr)
                end
            end
        end
    end

    # Subcycling for CFL constraint on rain sedimentation
    rainsplit = max(1, ceil(Int, Δt / dt_max))
    inv_rainsplit = inv(FT(rainsplit))  # Precompute for final averaging
    dt0 = Δt * inv_rainsplit
    precip_accum = zero(FT)  # Local accumulator to reduce global memory writes

    #####
    ##### PHASE 2: Subcycle microphysics (in mixing ratio space)
    #####

    for nt = 1:rainsplit

        # Accumulate surface precipitation (qʳ × vᵗ)
        @inbounds begin
            rᵛ_1 = qᵛ_field[i, j, 1]
            rᶜ_1 = qᶜˡ_field[i, j, 1]
            rʳ_1 = qʳ_field[i, j, 1]
            rᵗ_1 = rᵛ_1 + rᶜ_1 + rʳ_1
            # qʳ = rʳ / (1 + rᵗ)
            qʳ_1 = rʳ_1 / (1 + rᵗ_1)
            precip_accum += qʳ_1 * vᵗ_rain[i, j, 1]
        end
        for k = 1:Nz
            @inbounds begin
                ρ = ρᵣ[k]
                p = pᵣ[k]
                θˡⁱ_k = θˡⁱ[i, j, k]

                rᵛ = qᵛ_field[i, j, k]
                rᶜ = qᶜˡ_field[i, j, k]
                rʳ = qʳ_field[i, j, k]

                # Convert to mass fractions for thermodynamic calculation
                rᵗ = rᵛ + rᶜ + rʳ
                rˡ = rᶜ + rʳ
                inv_one_plus_rᵗ = inv(1 + rᵗ)
                qᵛ_current = rᵛ * inv_one_plus_rᵗ
                qˡ_current = rˡ * inv_one_plus_rᵗ

                # Moist thermodynamics: T = Π θˡⁱ + ℒˡᵣ qˡ / cᵖᵐ
                q = MoistureMassFractions(qᵛ_current, qˡ_current)
                cᵖᵐ = mixture_heat_capacity(q, constants)
                Rᵐ  = mixture_gas_constant(q, constants)
                Π = (p / p₀)^(Rᵐ / cᵖᵐ)
                T_k = Π * θˡⁱ_k + ℒˡᵣ * qˡ_current / cᵖᵐ

                # Exner-like pressure ratio for Kessler scheme
                pk = (p / p₀_kessler)^κ

                # Rain sedimentation (upstream differencing)
                r_k = 0.001 * ρ
                velqr_k = vᵗ_rain[i, j, k]

                if k < Nz
                    z_k   = znode(i, j, k, grid, Center(), Center(), Center())
                    z_kp1 = znode(i, j, k+1, grid, Center(), Center(), Center())
                    dz = z_kp1 - z_k

                    ρ_kp1 = ρᵣ[k+1]
                    r_kp1 = 0.001 * ρ_kp1
                    rʳ_kp1 = qʳ_field[i, j, k+1]  # Mixing ratio
                    velqr_kp1 = vᵗ_rain[i, j, k+1]

                    sed = dt0 * (r_kp1 * rʳ_kp1 * velqr_kp1 - r_k * rʳ * velqr_k) / (r_k * dz)
                else
                    # Top boundary: rain falls out
                    z_k   = znode(i, j, k, grid, Center(), Center(), Center())
                    z_km1 = znode(i, j, k-1, grid, Center(), Center(), Center())
                    dz_half = 0.5 * (z_k - z_km1)
                    sed = -dt0 * rʳ * velqr_k / dz_half
                end

                # Autoconversion + accretion (KW eq. 2.13)
                rrprod = rᶜ - (rᶜ - dt0 * max(0.001 * (rᶜ - 0.001), 0)) /
                         (1 + dt0 * 2.2 * rʳ^0.875)
                rᶜ_new = max(rᶜ - rrprod, 0)
                rʳ_new = max(rʳ + rrprod + sed, 0)

                # Saturation specific humidity using Breeze thermodynamics
                # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) is the saturation mass fraction
                qᵛ⁺ = saturation_specific_humidity(T_k, ρ, constants, PlanarLiquidSurface())
                # Convert to saturation mixing ratio: rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
                rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)

                # Saturation adjustment
                prod = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f5 / (T_k - 36)^2)

                # Rain evaporation (KW eq. 2.14)
                # Note: The evaporation formula uses empirical constants from DCMIP2016
                # We compute pc for the evaporation denominator (scheme-specific factor)
                pc = 380 / (pk^(1 / κ) * p₀_kessler)
                rrr = r_k * rʳ_new
                ern_num = (1.6 + 124.9 * rrr^0.2046) * rrr^0.525
                ern_den = 2550000 * pc / (3.8 * rᵛ⁺) + 540000
                subsaturation = max(rᵛ⁺ - rᵛ, 0)
                ern_rate = ern_num / ern_den * subsaturation / (r_k * rᵛ⁺ + FT(1e-20))
                ern = min(dt0 * ern_rate, max(-prod - rᶜ_new, 0), rʳ_new)

                # Apply adjustments
                condensation = max(prod, -rᶜ_new)
                rᵛ_new = max(rᵛ - condensation + ern, 0)
                rᶜ_final = rᶜ_new + condensation
                rʳ_final = rʳ_new - ern

                qᵛ_field[i, j, k]  = rᵛ_new
                qᶜˡ_field[i, j, k] = rᶜ_final
                qʳ_field[i, j, k]  = rʳ_final

                # Update θˡⁱ from latent heating
                # Uses Breeze's thermodynamic constants for consistency
                net_phase_change = condensation - ern
                ΔT_phase = ℒˡᵣ_over_cᵖᵈ * net_phase_change
                T_new = T_k + ΔT_phase

                # Convert back to θˡⁱ with updated moisture
                rᵗ_new = rᵛ_new + rᶜ_final + rʳ_final
                rˡ_new = rᶜ_final + rʳ_final
                inv_one_plus_rᵗ_new = inv(1 + rᵗ_new)
                qᵛ_new_mf = rᵛ_new * inv_one_plus_rᵗ_new
                qˡ_new = rˡ_new * inv_one_plus_rᵗ_new

                q_new = MoistureMassFractions(qᵛ_new_mf, qˡ_new)
                cᵖᵐ_new = mixture_heat_capacity(q_new, constants)
                Rᵐ_new  = mixture_gas_constant(q_new, constants)
                Π_new = (p / p₀)^(Rᵐ_new / cᵖᵐ_new)

                # θˡⁱ = (T - ℒˡᵣ qˡ / cᵖᵐ) / Π
                θˡⁱ_new = (T_new - ℒˡᵣ * qˡ_new / cᵖᵐ_new) / Π_new

                θˡⁱ[i, j, k]  = θˡⁱ_new
                ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
            end
        end

        # Recalculate terminal velocities for next subcycle
        if nt < rainsplit
            for k = 1:Nz
                @inbounds begin
                    ρ = ρᵣ[k]
                    rʳ = qʳ_field[i, j, k]
                    vᵗ_rain[i, j, k] = kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
                end
            end
        end
    end

    @inbounds precipitation_rate[i, j] = precip_accum * inv_rainsplit

    #####
    ##### PHASE 3: Convert mixing ratio → mass fraction
    #####

    for k = 1:Nz
        @inbounds begin
            ρ = ρᵣ[k]
            rᵛ = qᵛ_field[i, j, k]
            rᶜ = qᶜˡ_field[i, j, k]
            rʳ = qʳ_field[i, j, k]

            rᵗ = rᵛ + rᶜ + rʳ
            # Precompute common factor for all conversions
            inv_one_plus_rᵗ = inv(1 + rᵗ)
            qᵛ  = rᵛ * inv_one_plus_rᵗ
            qᶜˡ = rᶜ * inv_one_plus_rᵗ
            qʳ  = rʳ * inv_one_plus_rᵗ
            # qᵗ = (rᵛ + rᶜ + rʳ) / (1 + rᵗ) = rᵗ / (1 + rᵗ)
            qᵗ  = rᵗ * inv_one_plus_rᵗ

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
@inline function update_microphysical_fields!(μ, ::DCMIP2016KM, i, j, k, grid, ρ, 𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    @inbounds begin
        μ.qᶜˡ[i, j, k] = μ.ρqᶜˡ[i, j, k] / ρ
        μ.qʳ[i, j, k]  = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k]  = qᵗ - μ.qᶜˡ[i, j, k] - μ.qʳ[i, j, k]
    end
    return nothing
end
