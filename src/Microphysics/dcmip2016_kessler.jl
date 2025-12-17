using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant,
    dry_air_gas_constant,
    vapor_gas_constant,
    PlanarLiquidSurface,
    saturation_vapor_pressure,
    temperature,
    density,
    is_absolute_zero,
    with_moisture,
    total_specific_moisture,
    AbstractThermodynamicState

using Oceananigans: Oceananigans, CenterField, Field, interior
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, Center
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

using DocStringExtensions: TYPEDSIGNATURES

"""
    struct DCMIP2016KesslerMicrophysics <: AbstractMicrophysics

DCMIP2016 implementation of the Kessler (1969) warm-rain bulk microphysics scheme.

This implementation follows the DCMIP2016 test case specification, which is based on
Klemp and Wilhelmson (1978).

# References
- Zarzycki, C. M., et al. (2019). DCMIP2016: the splitting supercell test case. Geoscientific Model Development, 12, 879–892.
- Kessler, E. (1969). On the Distribution and Continuity of Water Substance in Atmospheric Circulations.
  Meteorological Monographs, 10(32).
- Klemp, J. B., & Wilhelmson, R. B. (1978). The Simulation of Three-Dimensional Convective Storm Dynamics.
  Journal of the Atmospheric Sciences, 35(6), 1070-1096.
- [DCMIP2016 Fortran implementation](kessler.f90 in https://doi.org/10.5281/zenodo.1298671)

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
"""
struct DCMIP2016KesslerMicrophysics end

const DCMIP2016KM = DCMIP2016KesslerMicrophysics

"""
$(TYPEDSIGNATURES)

Return the names of prognostic microphysical fields for the Kessler scheme.

# Fields
- `:ρqᶜˡ`: Density-weighted cloud liquid mass fraction (\$kg/m^3\$).
- `:ρqʳ`: Density-weighted rain mass fraction (\$kg/m^3\$).
"""
prognostic_field_names(::DCMIP2016KM) = (:ρqᶜˡ, :ρqʳ)

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
- `precipitation_rate`: Surface precipitation rate (\$m/s\$).
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
@inline function compute_moisture_fractions(i, j, k, grid, ::DCMIP2016KM, ρ, qᵗ, μ)
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
@inline maybe_adjust_thermodynamic_state(𝒰, ::DCMIP2016KM, μ, qᵗ, constants) = 𝒰

"""
$(TYPEDSIGNATURES)

Return `nothing`.

Rain sedimentation is handled internally by the kernel rather than through the advection interface.
"""
@inline microphysical_velocities(::DCMIP2016KM, name) = nothing

"""
$(TYPEDSIGNATURES)

Return zero tendency.

All microphysical source/sink terms are applied directly to the prognostic fields via the
`microphysics_model_update!` kernel, bypassing the standard tendency interface.
"""
@inline microphysical_tendency(i, j, k, grid, ::DCMIP2016KM, name, μ, 𝒰, constants) = zero(eltype(grid))

#####
##### Kessler scheme constants (from kessler.f90)
#####

# Clausius-Clapeyron coefficient for saturation vapor pressure
const kessler_f2x = 17.27

# Saturation adjustment coefficient: \$237.3 \cdot f2x \cdot ℒᵛ_Kessler / cᵖᵈ_Kessler\$
# where \$ℒᵛ_Kessler = 2.5 \times 10^6 J/kg\$ (latent heat of vaporization) and \$cᵖᵈ_Kessler = 1003 J/(kg \cdot K)\$
const kessler_f5 = 237.3 * kessler_f2x * 2500000.0 / 1003.0

# Kappa = \$R_d/cᵖᵈ_Kessler\$ (ratio of dry air gas constant to specific heat)
const kessler_xk = 0.2875

# Reference sea level pressure (millibars)
const kessler_psl = 1000.0

# Density of liquid water (\$kg/m^3\$)
const kessler_rhoqr = 1000.0

#####
##### Conversion between mass fraction and mixing ratio
#####
# The Kessler scheme uses mixing ratios (mass of hydrometeor / mass of dry air).
# Breeze uses mass fractions (mass of hydrometeor / total mass of moist air).
#
# Conversion:
#   r = q / (1 - q^t)  where q^t is total mass fraction
#   q = r / (1 + r^t)  where r^t is total mixing ratio
#####

"""
$(TYPEDSIGNATURES)

Convert mass fraction \$q\$ to mixing ratio \$r\$.
\$q^t\$ is the total mass fraction (sum of all moisture species).

The conversion is: \$r = q / (1 - q^t)\$
"""
@inline mass_fraction_to_mixing_ratio(q, qᵗ) = q / (1 - qᵗ)

"""
$(TYPEDSIGNATURES)

Convert mixing ratio \$r\$ to mass fraction \$q\$.
\$r^t\$ is the total mixing ratio (sum of all moisture species).

The conversion is: \$q = r / (1 + r^t)\$
"""
@inline mixing_ratio_to_mass_fraction(r, rᵗ) = r / (1 + rᵗ)


"""
$(TYPEDSIGNATURES)

Compute liquid water terminal velocity (\$m/s\$) following Klemp and Wilhelmson (1978) eq. 2.15.

Uses the three-argument form with explicit reference density.
`ρ_bottom` is the reference density at the lowest vertical level (\$kg/m^3\$).
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

This function launches a GPU kernel that processes each column independently, with rain sedimentation subcycling.

The kernel handles conversion between mass fractions (Breeze) and mixing ratios (Kessler)
internally for efficiency. Water vapor is diagnosed from \$q^v = q^t - q^{cl} - q^r\$.
"""
function microphysics_model_update!(::DCMIP2016KM, model)
    grid = model.grid
    arch = architecture(grid)
    Nz = grid.Nz
    Δt = model.clock.last_Δt

    # Skip microphysics update if timestep is zero, infinite, or invalid
    # (e.g., during model construction before any time step has been taken)
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    # Reference state - use interior() for reduced fields to get GPU-compatible arrays
    ρᵣ = interior(model.formulation.reference_state.density, 1, 1, :)
    pᵣ = interior(model.formulation.reference_state.pressure, 1, 1, :)

    # Surface pressure for Exner function
    p₀ = model.formulation.reference_state.surface_pressure

    # Thermodynamic constants for liquid-ice potential temperature conversion
    constants = model.thermodynamic_constants

    # Thermodynamic fields (liquid-ice potential temperature, NOT regular potential temperature)
    θˡⁱ  = model.formulation.thermodynamics.potential_temperature
    ρθˡⁱ = model.formulation.thermodynamics.potential_temperature_density

    # Total moisture density (prognostic variable of AtmosphereModel)
    ρqᵗ = model.moisture_density

    # Microphysical fields
    μ = model.microphysical_fields

    # Use interior() for 2D field to avoid GPU indexing issues
    precipitation_rate_data = interior(μ.precipitation_rate, :, :, 1)

    launch!(arch, grid, :xy, _kessler_microphysical_update!,
            grid, Nz, Δt, ρᵣ, pᵣ, p₀, constants, θˡⁱ, ρθˡⁱ,
            ρqᵗ, μ.ρqᶜˡ, μ.ρqʳ,
            μ.qᵛ, μ.qᶜˡ, μ.qʳ,
            precipitation_rate_data, μ.vᵗ_rain)

    return nothing
end

#####
##### GPU kernel for Kessler microphysics
#####

# This kernel processes each (i,j) column independently. The algorithm:
#
# 1. INITIALIZATION: Convert mass fractions -> mixing ratios for the entire column.
#    - Diagnose q^v = q^t - q^{cl} - q^r from total moisture and condensates.
#    - Store mixing ratios temporarily in diagnostic fields (qᵛ_field, qᶜˡ_field, qʳ_field).
#    - Compute terminal velocities and determine the CFL-limited subcycle timestep.
#
# 2. SUBCYCLING: For each subcycle timestep:
#    a. Accumulate surface precipitation.
#    b. For each vertical level (bottom to top):
#       - Compute temperature from liquid-ice potential temperature: T = Π * θ_li + ℒˡᵣ * q_l / cᵖᵐ.
#       - Rain sedimentation via upstream differencing.
#       - Autoconversion + accretion (cloud -> rain).
#       - Saturation adjustment (vapor <-> cloud).
#       - Rain evaporation (rain -> vapor in subsaturated air).
#       - Update liquid-ice potential temperature accounting for:
#         * Latent heating from phase changes (T_new = T + ℒᵛ_Kessler * Δq_l / cᵖᵈ_Kessler).
#         * Conversion back to θ_li with new liquid content: θ_li = (T - ℒˡᵣ * q_l / cᵖᵐ) / Π.
#    c. Recalculate terminal velocities for the next subcycle.
#
# 3. FINALIZATION: Convert mixing ratios -> mass fractions for the entire column.
#    - Write back to prognostic fields (ρqᵗ, ρqᶜˡ, ρqʳ).
#    - Update diagnostic fields with final mass fractions.
#
# Note: Breeze uses liquid-ice potential temperature (θ_li), NOT standard potential
# temperature (θ). The relationship is:
#   T = Π * θ_li + (ℒˡᵣ * q_l + ℒⁱᵣ * q_i) / cᵖᵐ
# For this warm-phase Kessler scheme (no ice), ice terms are zero.

@kernel function _kessler_microphysical_update!(grid, Nz, Δt, ρᵣ, pᵣ, p₀, constants, θˡⁱ, ρθˡⁱ,
                                                 ρqᵗ, ρqᶜˡ, ρqʳ,
                                                 qᵛ_field, qᶜˡ_field, qʳ_field,
                                                 precipitation_rate, vᵗ_rain)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    # Extract thermodynamic constants for liquid-ice potential temperature
    ℒˡᵣ = constants.liquid.reference_latent_heat  # Latent heat of vaporization (J/kg)

    # Reference density at lowest level for terminal velocity correction (KW eq. 2.15)
    # Used as ρˢ in: velqr = 36.34 * (qr * 0.001 * ρ)^0.1364 * sqrt(ρˢ/ρ)
    @inbounds ρ_bottom = ρᵣ[1]

    #####
    ##### PHASE 1: Convert mass fraction → mixing ratio for entire column
    #####
    # All physics calculations use mixing ratios (mass per dry air mass)
    # Diagnostic fields temporarily store mixing ratios during physics loop
    
    dt_max = Δt
    for k = 1:Nz
        @inbounds begin
            ρ = ρᵣ[k]

            # Get total moisture from prognostic field
            qᵗ = ρqᵗ[i, j, k] / ρ

            # Get condensate mass fractions from prognostic microphysical fields
            qᶜˡ = max(ρqᶜˡ[i, j, k] / ρ, zero(FT))
            qʳ  = max(ρqʳ[i, j, k] / ρ, zero(FT))

            # Ensure total moisture is at least the diagnosed condensate.
            # This prevents negative diagnosed vapor in rare inconsistent states.
            qᵗ = max(qᵗ, qᶜˡ + qʳ)

            # Diagnose water vapor: qᵛ = qᵗ - qᶜˡ - qʳ
            qᵛ = qᵗ - qᶜˡ - qʳ

            # ===== CONVERSION: mass fraction → mixing ratio =====
            rʳ = mass_fraction_to_mixing_ratio(qʳ, qᵗ)

            # Terminal velocity (m/s) - uses mixing ratio
            velqr = kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
            vᵗ_rain[i, j, k] = velqr

            # Store mixing ratios in diagnostic fields temporarily for use in physics loop
            # This avoids repeated conversion inside the subcycle loop
            rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ)
            rᶜ = mass_fraction_to_mixing_ratio(qᶜˡ, qᵗ)
            qᵛ_field[i, j, k]  = rᵛ
            qᶜˡ_field[i, j, k] = rᶜ
            qʳ_field[i, j, k]  = rʳ
        end

        # CFL check for sedimentation
        if k < Nz
            @inbounds begin
                z_k   = znode(i, j, k, grid, Center(), Center(), Center())
                z_kp1 = znode(i, j, k+1, grid, Center(), Center(), Center())
                dz = z_kp1 - z_k
                velqr = vᵗ_rain[i, j, k]
                if velqr > 0
                    dt_max = min(dt_max, 0.8 * dz / velqr)
                end
            end
        end
    end

    # Number of subcycles for rain sedimentation (CFL constraint)
    # Ensures rain doesn't fall more than 0.8 * Δz per substep
    rainsplit = max(1, ceil(Int, Δt / dt_max))
    dt0 = Δt / rainsplit

    # Initialize surface precipitation accumulator
    @inbounds precipitation_rate[i, j] = zero(FT)

    #####
    ##### PHASE 2: Subcycle through microphysics (all in mixing ratio space)
    #####
    for nt = 1:rainsplit

        # Accumulate surface precipitation (using mixing ratio stored in qʳ_field)
        @inbounds begin
            ρ_1 = ρᵣ[1]
            rʳ_1 = qʳ_field[i, j, 1]  # This is mixing ratio during physics loop
            precipitation_rate[i, j] += ρ_1 * rʳ_1 * vᵗ_rain[i, j, 1] / kessler_rhoqr
        end

        #####
        ##### Process each level (all in mixing ratio space)
        #####
        for k = 1:Nz
            @inbounds begin
                ρ = ρᵣ[k]
                p = pᵣ[k]
                θˡⁱ_k = θˡⁱ[i, j, k]

                # Read mixing ratios (stored in diagnostic fields during physics)
                rᵛ = qᵛ_field[i, j, k]
                rᶜ = qᶜˡ_field[i, j, k]
                rʳ = qʳ_field[i, j, k]

                # Current liquid mass fraction (cloud + rain) for temperature calculation
                # Convert from mixing ratio to mass fraction for thermodynamic calculation
                rᵗ = rᵛ + rᶜ + rʳ
                qᵛ_current = mixing_ratio_to_mass_fraction(rᵛ, rᵗ)
                qˡ_current = mixing_ratio_to_mass_fraction(rᶜ + rʳ, rᵗ)

                # Use Breeze's moist Exner function and moist heat capacity so that
                # θˡⁱ ↔ T is thermodynamically consistent with the model.
                q = MoistureMassFractions(qᵛ_current, qˡ_current)
                cᵖᵐ = mixture_heat_capacity(q, constants)
                Rᵐ  = mixture_gas_constant(q, constants)
                Π = (p / p₀)^(Rᵐ / cᵖᵐ)

                # Temperature from liquid-ice potential temperature state:
                # T = Π θˡⁱ + (ℒˡᵣ qˡ)/cᵖᵐ (warm phase; qⁱ = 0)
                T_k = Π * θˡⁱ_k + ℒˡᵣ * qˡ_current / cᵖᵐ

                # Also compute Kessler's pk for saturation calculation
                p_mb = p / 100
                pk = (p_mb / kessler_psl)^kessler_xk

                #####
                ##### Rain sedimentation using upstream differencing
                #####
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

                #####
                ##### Autoconversion + accretion (KW eq. 2.13a,b) - implicit formula
                #####
                rrprod = rᶜ - (rᶜ - dt0 * max(0.001 * (rᶜ - 0.001), 0)) / 
                         (1 + dt0 * 2.2 * rʳ^0.875)
                rᶜ_new = max(rᶜ - rrprod, 0)
                rʳ_new = max(rʳ + rrprod + sed, 0)

                #####
                ##### Saturation mixing ratio (KW eq. 2.11)
                #####
                pc = 3.8 / (pk^(1 / kessler_xk) * kessler_psl)
                rᵛˢ = pc * exp(kessler_f2x * (T_k - 273) / (T_k - 36))

                #####
                ##### Saturation adjustment
                #####
                prod = (rᵛ - rᵛˢ) / (1 + rᵛˢ * kessler_f5 / (T_k - 36)^2)

                #####
                ##### Rain evaporation (KW eq. 2.14a,b)
                #####
                rrr = r_k * rʳ_new
                ern_num = (1.6 + 124.9 * rrr^0.2046) * rrr^0.525
                ern_den = 2550000 * pc / (3.8 * rᵛˢ) + 540000
                subsaturation = max(rᵛˢ - rᵛ, 0)
                ern_rate = ern_num / ern_den * subsaturation / (r_k * rᵛˢ + 1e-20)
                ern = min(dt0 * ern_rate, max(-prod - rᶜ_new, 0), rʳ_new)

                #####
                ##### Apply adjustments (KW eq. 3.10)
                #####
                condensation = max(prod, -rᶜ_new)

                rᵛ_new = max(rᵛ - condensation + ern, 0)
                rᶜ_final = rᶜ_new + condensation
                rʳ_final = rʳ_new - ern

                # Update mixing ratios in diagnostic fields (still in mixing ratio space)
                qᵛ_field[i, j, k]  = rᵛ_new
                qᶜˡ_field[i, j, k] = rᶜ_final
                qʳ_field[i, j, k]  = rʳ_final

                #####
                ##### Update liquid-ice potential temperature
                #####
                # The Fortran Kessler scheme updates θ (standard potential temperature) as:
                #   θ_new = θ + ℒᵛ_Kessler * (condensation - ern) / (cᵖᵈ_Kessler * Π)
                # where ℒᵛ_Kessler = 2500000 J/kg, cᵖᵈ_Kessler = 1003 J/(kg·K), and
                # condensation and ern are in mixing ratio and represent PHASE CHANGES ONLY.
                #
                # For liquid-ice potential temperature θˡⁱ, the relationship is:
                #   T = Π * θˡⁱ + ℒˡᵣ * qˡ / cᵖᵐ
                #   θˡⁱ = (T - ℒˡᵣ * qˡ / cᵖᵐ) / Π
                #
                # The temperature change from latent heating (PHASE CHANGES ONLY) is:
                #   ΔT = ℒᵛ_Kessler * (condensation - ern) / cᵖᵈ_Kessler
                #
                # Note: We use Kessler's hardcoded constants (ℒᵛ_Kessler = 2500000, cᵖᵈ_Kessler = 1003) for 
                # the latent heating to match the DCMIP2016 configuration exactly, but use Breeze's ℒˡᵣ for 
                # the θˡⁱ definition for thermodynamic consistency.
                
                # Net phase change in mixing ratio (positive = condensation, negative = evaporation)
                net_phase_change = condensation - ern
                
                # Temperature change from latent heating using Kessler's constants
                # (same as Fortran: ΔT = 2500000/1003 * net_phase_change)
                ΔT_phase = 2500000.0 * net_phase_change / 1003.0
                T_new = T_k + ΔT_phase

                # Compute new liquid mass fraction (includes ALL changes: autoconversion, 
                # sedimentation, saturation adjustment, evaporation)
                rᵗ_new = rᵛ_new + rᶜ_final + rʳ_final
                qᵛ_new_mf = mixing_ratio_to_mass_fraction(rᵛ_new, rᵗ_new)
                qˡ_new = mixing_ratio_to_mass_fraction(rᶜ_final + rʳ_final, rᵗ_new)

                # Update moist thermodynamic properties with the new moisture state.
                q_new = MoistureMassFractions(qᵛ_new_mf, qˡ_new)
                cᵖᵐ_new = mixture_heat_capacity(q_new, constants)
                Rᵐ_new  = mixture_gas_constant(q_new, constants)
                Π_new = (p / p₀)^(Rᵐ_new / cᵖᵐ_new)

                # Convert back to liquid-ice potential temperature:
                # θˡⁱ = (T - ℒˡᵣ qˡ / cᵖᵐ) / Π
                θˡⁱ_new = (T_new - ℒˡᵣ * qˡ_new / cᵖᵐ_new) / Π_new

                # Update thermodynamics
                θˡⁱ[i, j, k]  = θˡⁱ_new
                ρθˡⁱ[i, j, k] = ρ * θˡⁱ_new
            end
        end

        # Recalculate terminal velocities for next subcycle (except last)
        if nt < rainsplit
            for k = 1:Nz
                @inbounds begin
                    ρ = ρᵣ[k]
                    rʳ = qʳ_field[i, j, k]  # Already mixing ratio
                    vᵗ_rain[i, j, k] = kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
                end
            end
        end
    end

    # Convert accumulated precipitation to average rate
    @inbounds precipitation_rate[i, j] /= rainsplit

    #####
    ##### PHASE 3: Convert mixing ratio → mass fraction for entire column
    #####
    # Write final values back to prognostic and diagnostic fields
    for k = 1:Nz
        @inbounds begin
            ρ = ρᵣ[k]

            # Read final mixing ratios
            rᵛ = qᵛ_field[i, j, k]
            rᶜ = qᶜˡ_field[i, j, k]
            rʳ = qʳ_field[i, j, k]

            # ===== CONVERSION: mixing ratio → mass fraction =====
            rᵗ = rᵛ + rᶜ + rʳ
            qᵛ  = mixing_ratio_to_mass_fraction(rᵛ, rᵗ)
            qᶜˡ = mixing_ratio_to_mass_fraction(rᶜ, rᵗ)
            qʳ  = mixing_ratio_to_mass_fraction(rʳ, rᵗ)
            qᵗ  = qᵛ + qᶜˡ + qʳ

            # Update prognostic fields (density-weighted mass fractions)
            # Note: ρqᵗ is updated because microphysics can change total moisture
            # (e.g., precipitation removes moisture from the column)
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
##### Interface stub for update_microphysical_fields!
#####

"""
$(TYPEDSIGNATURES)

Update the diagnostic mass fraction fields from the prognostic density-weighted fields.

Water vapor is diagnosed as \$q^v = q^t - q^{cl} - q^r\$.

This function is called by the general `update_state!` machinery. The main microphysics
updates are performed via the `microphysics_model_update!` kernel.
"""
@inline function update_microphysical_fields!(μ, ::DCMIP2016KM, i, j, k, grid, ρ, 𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    @inbounds begin
        μ.qᶜˡ[i, j, k] = μ.ρqᶜˡ[i, j, k] / ρ
        μ.qʳ[i, j, k]  = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k]  = qᵗ - μ.qᶜˡ[i, j, k] - μ.qʳ[i, j, k]
    end
    return nothing
end
