using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
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
using Oceananigans.Grids: znode, Center, Nothing as GridNothing
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

using DocStringExtensions: TYPEDSIGNATURES

"""
    KesslerMicrophysics

DCMIP2016 implementation of the Kessler (1969) warm-rain bulk microphysics scheme,
following Klemp and Wilhelmson (1978).

# References

- Kessler, E. (1969). On the Distribution and Continuity of Water Substance in 
  Atmospheric Circulations. Meteorological Monographs, 10(32).
- Klemp, J. B., & Wilhelmson, R. B. (1978). The Simulation of Three-Dimensional 
  Convective Storm Dynamics. Journal of the Atmospheric Sciences, 35(6), 1070-1096.
- DCMIP2016 Fortran implementation: 
  https://gitlab.in2p3.fr/ipsl/projets/dynamico/dynamico/-/blob/master/src/dcmip2016_kessler_physic.f90

# Moisture categories

This scheme represents moisture in three categories:
- Water vapor mixing ratio (`rᵛ`)
- Cloud water mixing ratio (`rᶜˡ`)
- Rain water mixing ratio(`rʳ`)

Breeze uses mass fractions, so conversions between mass fractions and mixing ratios are performed as needed. 
Also, Breeze does not track water vapor as a prognostic variable; instead, it is diagnosed from total moisture. 

Internally, the scheme uses mixing ratios (mass per unit mass of dry air) for microphysics
calculations. 

# Physical processes

1. **Autoconversion** (KW eq. 2.13a): Cloud → Rain when cloud exceeds threshold
2. **Accretion** (KW eq. 2.13b): Cloud → Rain via collection by falling rain  
3. **Saturation adjustment** (KW eq. 3.10): Vapor ↔ Cloud to maintain saturation
4. **Rain evaporation** (KW eq. 2.14): Rain → Vapor in subsaturated air
5. **Rain sedimentation** (KW eq. 2.15): Gravitational settling of rain

# Constants (from `kessler.f90`)

- `kessler_f2x = 17.27`: Clausius-Clapeyron coefficient
- `kessler_f5 = 237.3 * f2x * 2500000 / 1003`: Saturation adjustment coefficient  
- `kessler_xk = 0.2875`: Kappa (Rᵈ/cₚ)
- `kessler_psl = 1000`: Reference pressure (mb)
- `kessler_rhoqr = 1000`: Density of liquid water (kg/m³)

# Implementation notes

- Physics is applied via a GPU kernel launched from `microphysics_model_update!`
- Rain sedimentation uses subcycling to satisfy CFL constraints
- All microphysical tendencies return zero; updates are applied directly in the kernel
"""
struct KesslerMicrophysics end

const KM = KesslerMicrophysics

"""
    prognostic_field_names(::KesslerMicrophysics)

Return the names of prognostic microphysical fields for Kessler scheme:
- `ρqᶜˡ`: density-weighted cloud liquid mass fraction (kg/m³)  
- `ρqʳ`: density-weighted rain mass fraction (kg/m³)

Note: Water vapor `qᵛ` is **not** prognostic. It is diagnosed as `qᵛ = qᵗ - qᶜˡ - qʳ`,
where `qᵗ` is the total specific moisture (a prognostic variable of `AtmosphereModel`).
"""
prognostic_field_names(::KM) = (:ρqᶜˡ, :ρqʳ)

"""
    materialize_microphysical_fields(::KesslerMicrophysics, grid, boundary_conditions)

Create and return all microphysical fields for the Kessler scheme.

# Prognostic fields (density-weighted, with boundary conditions)
- `ρqᶜˡ`, `ρqʳ`: Density-weighted cloud liquid and rain mass fractions

# Diagnostic fields (mass fractions, no boundary conditions needed)
- `qᵛ`: Water vapor mass fraction, diagnosed as `qᵛ = qᵗ - qᶜˡ - qʳ`
- `qᶜˡ`, `qʳ`: Cloud liquid and rain mass fractions (kg/kg)
- `precipitation_rate`: Surface precipitation rate (m/s)
- `vᵗ_rain`: Rain terminal velocity (m/s)
"""
function materialize_microphysical_fields(::KM, grid, boundary_conditions)
    # Prognostic fields (density-weighted)
    ρqᶜˡ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᶜˡ)
    ρqʳ  = CenterField(grid, boundary_conditions=boundary_conditions.ρqʳ)

    # Diagnostic fields (mass fractions)
    qᵛ  = CenterField(grid)
    qᶜˡ = CenterField(grid)
    qʳ  = CenterField(grid)

    # Precipitation and velocity diagnostics
    precipitation_rate = Field{Center, Center, GridNothing}(grid)
    vᵗ_rain = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, qᵛ, qᶜˡ, qʳ, precipitation_rate, vᵗ_rain)
end

#####
##### Interface functions for AtmosphereModel integration
#####

"""
    compute_moisture_fractions(i, j, k, grid, ::KesslerMicrophysics, ρ, qᵗ, μ)

Compute moisture mass fractions at grid point (i, j, k) for thermodynamic state.
Water vapor is diagnosed as `qᵛ = qᵗ - qᶜˡ - qʳ`.
Returns `MoistureMassFractions(qᵛ, qˡ)` where `qˡ = qᶜˡ + qʳ` is total liquid.
"""
@inline function compute_moisture_fractions(i, j, k, grid, ::KM, ρ, qᵗ, μ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ  = μ.ρqʳ[i, j, k] / ρ
    end
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

"""
    maybe_adjust_thermodynamic_state(𝒰, ::KesslerMicrophysics, μ, qᵗ, constants)

Return thermodynamic state without adjustment. Kessler scheme performs its own
saturation adjustment internally via the kernel.
"""
@inline maybe_adjust_thermodynamic_state(𝒰, ::KM, μ, qᵗ, constants) = 𝒰

"""
    microphysical_velocities(::KesslerMicrophysics, name, μ)

Return `nothing` - rain sedimentation is handled internally by the kernel
rather than through the advection interface.
"""
@inline microphysical_velocities(::KM, name) = nothing

"""
    microphysical_tendency(i, j, k, grid, ::KesslerMicrophysics, name, μ, 𝒰, constants)

Return zero tendency. All microphysical source/sink terms are applied directly
to prognostic fields via `microphysics_model_update!` kernel, bypassing the
standard tendency interface.
"""
@inline microphysical_tendency(i, j, k, grid, ::KM, name, μ, 𝒰, constants) = zero(eltype(grid))

#####
##### Kessler scheme constants (from kessler.f90)
#####

# Clausius-Clapeyron coefficient for saturation vapor pressure
const kessler_f2x = 17.27

# Saturation adjustment coefficient: 237.3 * f2x * Lᵥ / cₚ
# where Lᵥ = 2.5e6 J/kg (latent heat of vaporization) and cₚ = 1003 J/(kg·K)
const kessler_f5 = 237.3 * kessler_f2x * 2500000.0 / 1003.0

# Kappa = Rᵈ/cₚ (ratio of dry air gas constant to specific heat)
const kessler_xk = 0.2875

# Reference sea level pressure (millibars)
const kessler_psl = 1000.0

# Density of liquid water (kg/m³)
const kessler_rhoqr = 1000.0

#####
##### Conversion between mass fraction and mixing ratio
#####
# Kessler scheme uses mixing ratio (mass of hydrometeor / mass of dry air)
# Breeze uses mass fraction (mass of hydrometeor / total mass of moist air)
# Conversion: r = q / (1 - qᵗ)  where qᵗ is total mass fraction
#             q = r / (1 + rᵗ)  where rᵗ is total mixing ratio
#####

"""
    mass_fraction_to_mixing_ratio(q, qᵗ)

Convert mass fraction `q` to mixing ratio `r`.
`qᵗ` is the total mass fraction (sum of all moisture species).

The conversion is: r = q / (1 - qᵗ)
"""
@inline mass_fraction_to_mixing_ratio(q, qᵗ) = q / (1 - qᵗ)

"""
    mixing_ratio_to_mass_fraction(r, rᵗ)

Convert mixing ratio `r` to mass fraction `q`.
`rᵗ` is the total mixing ratio (sum of all moisture species).

The conversion is: q = r / (1 + rᵗ)
"""
@inline mixing_ratio_to_mass_fraction(r, rᵗ) = r / (1 + rᵗ)

"""
    kessler_saturation_mixing_ratio(T, p)

Compute saturation vapor mixing ratio following KW eq. 2.11.
Uses temperature T (K) and pressure p (Pa).
Original Fortran implementation: qvs = pc * exp(f2x * (Π*θ - 273) / (Π*θ - 36))
where pc = 3.8 / (Π^(1/xk) * psl) = 3.8 / (p/p0 * psl) with p in suitable units.

Note: This returns mixing ratio (mass of vapor / mass of dry air), not mass fraction.
"""
@inline function kessler_saturation_mixing_ratio(T, p)
    # Convert pressure from Pa to mb for consistency with Fortran
    p_mb = p / 100
    pc = 3.8 / p_mb
    rᵛˢ = pc * exp(kessler_f2x * (T - 273) / (T - 36))
    return rᵛˢ
end

"""
    kessler_terminal_velocity(rʳ, ρ, ρˢ)

Compute liquid water terminal velocity (m/s) following KW eq. 2.15.
Uses three-argument form with explicit reference density. ρˢ is surface air density (kg/m³).
"""
@inline function kessler_terminal_velocity(rʳ, ρ, ρˢ)
    rhalf = sqrt(ρˢ / ρ)
    return 36.34 * (rʳ * 0.001 * ρ)^0.1364 * rhalf
end

#####
##### Main update function - launches GPU kernel
#####

"""
    microphysics_model_update!(::KM, model)

Apply Kessler microphysics to the model. This function launches a GPU kernel
that processes each column independently, with rain sedimentation subcycling.

The kernel handles conversion between mass fractions (Breeze) and mixing ratios (Kessler)
internally for efficiency. Water vapor is diagnosed from `qᵛ = qᵗ - qᶜˡ - qʳ`.
"""
function microphysics_model_update!(::KM, model)
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

    # Thermodynamic fields
    θ  = model.formulation.thermodynamics.potential_temperature
    ρθ = model.formulation.thermodynamics.potential_temperature_density
    T  = model.temperature

    # Total moisture density (prognostic variable of AtmosphereModel)
    ρqᵗ = model.moisture_density

    # Microphysical fields
    μ = model.microphysical_fields

    # Use interior() for 2D field to avoid GPU indexing issues
    precipitation_rate_data = interior(μ.precipitation_rate, :, :, 1)

    launch!(arch, grid, :xy, _kessler_microphysical_update!,
            grid, Nz, Δt, ρᵣ, pᵣ, θ, ρθ, T,
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
# 1. INITIALIZATION: Convert mass fractions → mixing ratios for entire column
#    - Diagnose qᵛ = qᵗ - qᶜˡ - qʳ from total moisture and condensates
#    - Store mixing ratios temporarily in diagnostic fields (qᵛ_field, qᶜˡ_field, qʳ_field)
#    - Compute terminal velocities and determine CFL-limited subcycle timestep
#
# 2. SUBCYCLING: For each subcycle timestep:
#    a. Accumulate surface precipitation
#    b. For each vertical level (bottom to top):
#       - Rain sedimentation via upstream differencing
#       - Autoconversion + accretion (cloud → rain)
#       - Saturation adjustment (vapor ↔ cloud)
#       - Rain evaporation (rain → vapor in subsaturated air)
#       - Update potential temperature for latent heating
#    c. Recalculate terminal velocities for next subcycle
#
# 3. FINALIZATION: Convert mixing ratios → mass fractions for entire column
#    - Write back to prognostic fields (ρqᵗ, ρqᶜˡ, ρqʳ)
#    - Update diagnostic fields with final mass fractions

@kernel function _kessler_microphysical_update!(grid, Nz, Δt, ρᵣ, pᵣ, θ, ρθ, T,
                                                 ρqᵗ, ρqᶜˡ, ρqʳ,
                                                 qᵛ_field, qᶜˡ_field, qʳ_field,
                                                 precipitation_rate, vᵗ_rain)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    # Surface density for terminal velocity calculation (KW eq. 2.15 correction factor)
    @inbounds ρˢ = ρᵣ[1]

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
            qᶜˡ = ρqᶜˡ[i, j, k] / ρ
            qʳ  = ρqʳ[i, j, k] / ρ

            # Diagnose water vapor: qᵛ = qᵗ - qᶜˡ - qʳ
            qᵛ = qᵗ - qᶜˡ - qʳ

            # ===== CONVERSION: mass fraction → mixing ratio =====
            rʳ = mass_fraction_to_mixing_ratio(qʳ, qᵗ)

            # Terminal velocity (m/s) - uses mixing ratio
            velqr = kessler_terminal_velocity(rʳ, ρ, ρˢ)
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
                θ_k = θ[i, j, k]

                # Exner function and temperature
                p_mb = p / 100
                pk = (p_mb / kessler_psl)^kessler_xk
                T_k = pk * θ_k

                # Read mixing ratios (stored in diagnostic fields during physics)
                rᵛ = qᵛ_field[i, j, k]
                rᶜ = qᶜˡ_field[i, j, k]
                rʳ = qʳ_field[i, j, k]

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
                θ_new = θ_k + 2500000 / (1003 * pk) * (condensation - ern)

                rᵛ_new = max(rᵛ - condensation + ern, 0)
                rᶜ_final = rᶜ_new + condensation
                rʳ_final = rʳ_new - ern

                # Update mixing ratios in diagnostic fields (still in mixing ratio space)
                qᵛ_field[i, j, k]  = rᵛ_new
                qᶜˡ_field[i, j, k] = rᶜ_final
                qʳ_field[i, j, k]  = rʳ_final

                # Update thermodynamics
                θ[i, j, k]  = θ_new
                ρθ[i, j, k] = ρ * θ_new
                T[i, j, k]  = pk * θ_new
            end
        end

        # Recalculate terminal velocities for next subcycle (except last)
        if nt < rainsplit
            for k = 1:Nz
                @inbounds begin
                    ρ = ρᵣ[k]
                    rʳ = qʳ_field[i, j, k]  # Already mixing ratio
                    vᵗ_rain[i, j, k] = kessler_terminal_velocity(rʳ, ρ, ρˢ)
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
    update_microphysical_fields!(μ, ::KesslerMicrophysics, i, j, k, grid, ρ, 𝒰, constants)

Update diagnostic mass fraction fields from prognostic density-weighted fields.
Water vapor is diagnosed as `qᵛ = qᵗ - qᶜˡ - qʳ`.
This is called by the general `update_state!` machinery. The main microphysics
updates are performed via `microphysics_model_update!` kernel.
"""
@inline function update_microphysical_fields!(μ, ::KM, i, j, k, grid, ρ, 𝒰, constants)
    qᵗ = total_specific_moisture(𝒰)
    @inbounds begin
        μ.qᶜˡ[i, j, k] = μ.ρqᶜˡ[i, j, k] / ρ
        μ.qʳ[i, j, k]  = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k]  = qᵗ - μ.qᶜˡ[i, j, k] - μ.qʳ[i, j, k]
    end
    return nothing
end