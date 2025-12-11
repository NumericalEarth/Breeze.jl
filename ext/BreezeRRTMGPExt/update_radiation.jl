#####
##### Update radiation fluxes from model state
#####
#
# Type ownership:
#   RRTMGP types (external, cannot modify):
#     - GrayAtmosphericState: atmospheric state arrays (t_lay, p_lay, t_lev, p_lev, z_lev, t_sfc)
#     - NoScatLWRTE, NoScatSWRTE: longwave/shortwave RTE solvers
#     - FluxLW, FluxSW: flux storage (flux_up, flux_dn, flux_net, flux_dn_dir)
#
#   Breeze types (internal, can modify):
#     - GrayRadiationModel: wrapper containing RRTMGP solvers and Oceananigans flux fields
#     - SingleColumnGrid type alias
#

using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ

using Breeze: dry_air_gas_constant
using Breeze.AtmosphereModels: ReferenceState
using Breeze.Thermodynamics: adiabatic_hydrostatic_pressure

import Breeze.AtmosphereModels: update_radiation!

"""
    update_radiation!(radiation::GrayRadiationModel, model)

Update the radiative fluxes from the current model state.

This function:
1. Updates the RRTMGP atmospheric state from model fields (T, p)
2. Computes the solar zenith angle from the model clock and grid location
3. Solves the longwave and shortwave RTE
4. Copies the fluxes to Oceananigans fields for output

Sign convention: positive flux = upward, negative flux = downward.
"""
function update_radiation!(radiation::GrayRadiationModel, model)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    constants = model.thermodynamic_constants

    rrtmgp_state = radiation.atmospheric_state
    surface_temperature = radiation.surface_temperature

    # Update RRTMGP atmospheric state from model fields
    update_rrtmgp_atmospheric_state!(rrtmgp_state, model, surface_temperature, constants)

    # Update solar zenith angle from clock
    update_solar_zenith_angle!(radiation.shortwave_solver, grid, clock)

    # Solve longwave RTE (RRTMGP external call)
    solve_lw!(radiation.longwave_solver, rrtmgp_state)

    # Solve shortwave RTE (only if sun is above horizon)
    cos_θz = radiation.shortwave_solver.bcs.cos_zenith[1]
    if cos_θz > 0
        solve_sw!(radiation.shortwave_solver, rrtmgp_state)
    else
        # Sun below horizon - zero shortwave fluxes
        radiation.shortwave_solver.flux.flux_up .= 0
        radiation.shortwave_solver.flux.flux_dn .= 0
        radiation.shortwave_solver.flux.flux_net .= 0
        radiation.shortwave_solver.flux.flux_dn_dir .= 0
    end

    # Copy RRTMGP flux arrays to Oceananigans fields with sign convention
    copy_fluxes_to_fields!(radiation, grid)

    return nothing
end

#####
##### Update RRTMGP atmospheric state from model fields
#####

"""
    update_rrtmgp_atmospheric_state!(rrtmgp_state, model, surface_temperature, constants)

Update the RRTMGP `GrayAtmosphericState` arrays from model fields.

# Physics notes

For radiation, we need temperature and pressure at both cell centers (layers)
and cell faces (levels).

**Temperature**: We use the actual temperature field `T` computed from the model state.
This is the temperature that matters for emission and absorption.

**Pressure**: In the anelastic approximation, pressure perturbations are negligible
compared to the reference (hydrostatic) pressure. We use the reference state pressure
`reference_state.pressure` at cell centers, and interpolate/extrapolate to faces.
The reference pressure is computed via `adiabatic_hydrostatic_pressure(z, p₀, θ₀)`.

# RRTMGP array layout
- RRTMGP uses (nlay, ncol) for layer values (at cell centers)
- RRTMGP uses (nlev, ncol) for level values (at cell faces)
- Layer 1 is at the bottom (surface), layer nlay is at the top
"""
function update_rrtmgp_atmospheric_state!(rrtmgp_state::GrayAtmosphericState,
                                          model, surface_temperature, constants)
    grid = model.grid
    arch = architecture(grid)
    
    # Temperature field (actual temperature from model state)
    T = model.temperature

    # Reference state provides the hydrostatic pressure profile
    # In the anelastic approximation, pressure ≈ reference pressure
    reference_state = model.formulation.reference_state
    p₀ = reference_state.surface_pressure  # Pressure at z = 0
    θ₀ = reference_state.potential_temperature  # Reference potential temperature

    # Unpack RRTMGP state arrays for kernels
    t_lay = rrtmgp_state.t_lay
    p_lay = rrtmgp_state.p_lay
    t_lev = rrtmgp_state.t_lev
    p_lev = rrtmgp_state.p_lev
    z_lev = rrtmgp_state.z_lev
    t_sfc = rrtmgp_state.t_sfc

    # Update layer values (at cell centers)
    # Temperature: actual temperature field
    # Pressure: reference hydrostatic pressure
    launch!(arch, grid, :xyz, _update_layer_values!, t_lay, p_lay, grid, T, reference_state)

    # Update level values (at cell faces)
    # Temperature: interpolated from cell centers
    # Pressure: interpolated from reference state, using surface_pressure at z=0
    launch!(arch, grid, :xyz, _update_level_values!,
            t_lev, p_lev, z_lev, grid, T, reference_state, p₀, θ₀, constants)

    # Surface skin temperature (for longwave emission from ground)
    t_sfc[1] = surface_temperature

    return nothing
end

@kernel function _update_layer_values!(t_lay, p_lay, grid, T, reference_state)
    i, j, k = @index(Global, NTuple)

    # Layer values are at cell centers
    # RRTMGP expects (k, col) indexing; for single column, col=1
    # For 3D grids, we'd need to linearize (i, j) → col
    @inbounds begin
        # Actual temperature from model state
        t_lay[k, 1] = T[i, j, k]

        # Reference pressure (hydrostatic background)
        # In anelastic models, pressure perturbations are negligible for radiation
        p_lay[k, 1] = reference_state.pressure[i, j, k]
    end
end

@kernel function _update_level_values!(t_lev, p_lev, z_lev, grid, T, reference_state,
                                       p₀, θ₀, constants)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    nlev = Nz + 1

    # Level values are at cell faces (Nz+1 faces for Nz cells)
    # The kernel runs for k = 1:Nz, so we handle:
    #   - k=1: bottom face (face index 1) and interior face (face index 2)
    #   - k=2:Nz-1: interior faces (face index k+1)
    #   - k=Nz: top face (face index Nz+1) and interior face (face index Nz)

    @inbounds begin
        # --- Bottom face (face index 1, at z = 0) ---
        if k == 1
            z_bottom = znode(i, j, 1, grid, Center(), Center(), Face())

            # Pressure at z=0: use the reference state's surface pressure directly
            # This is exact by definition of the reference state
            p_lev[1, 1] = p₀

            # Temperature at bottom face: extrapolate from first cell center
            # Use the first layer temperature as approximation for near-surface air
            t_lev[1, 1] = T[i, j, 1]

            # Altitude
            z_lev[1, 1] = z_bottom
        end

        # --- Top face (face index Nz+1) ---
        if k == Nz
            z_top = znode(i, j, nlev, grid, Center(), Center(), Face())

            # Pressure at top: use the same adiabatic hydrostatic formula
            # that defines the reference state, for consistency
            p_lev[nlev, 1] = adiabatic_hydrostatic_pressure(z_top, p₀, θ₀, constants)

            # Temperature at top face: use last layer temperature
            t_lev[nlev, 1] = T[i, j, Nz]

            # Altitude
            z_lev[nlev, 1] = z_top
        end

        # --- Interior faces (face indices 2 to Nz) ---
        # These fall between cell centers k and k+1
        if k < Nz
            kface = k + 1
            z_face = znode(i, j, kface, grid, Center(), Center(), Face())

            # Pressure: geometric mean of adjacent cell centers
            # This is equivalent to log-linear interpolation, which is appropriate
            # for the exponential-like hydrostatic pressure profile
            p_below = reference_state.pressure[i, j, k]
            p_above = reference_state.pressure[i, j, k + 1]
            p_lev[kface, 1] = sqrt(p_below * p_above)

            # Temperature: arithmetic mean of adjacent cell centers
            T_below = T[i, j, k]
            T_above = T[i, j, k + 1]
            t_lev[kface, 1] = (T_below + T_above) / 2

            # Altitude
            z_lev[kface, 1] = z_face
        end
    end
end

#####
##### Update solar zenith angle
#####

"""
    update_solar_zenith_angle!(sw_solver, grid, clock)

Update the solar zenith angle in the shortwave solver from the model clock.

Uses the datetime from `clock.time` and the grid's location (latitude/longitude)
to compute the cosine of the solar zenith angle via celestial mechanics.
"""
function update_solar_zenith_angle!(sw_solver, grid, clock)
    datetime = clock.time

    if datetime isa DateTime
        cos_θz = cos_solar_zenith_angle(grid, datetime)
        sw_solver.bcs.cos_zenith[1] = max(cos_θz, 0)  # Clamp to positive (sun above horizon)
    else
        # If clock.time is not a DateTime, use a default (overhead sun)
        sw_solver.bcs.cos_zenith[1] = 0.5
    end

    return nothing
end

#####
##### Copy RRTMGP fluxes to Oceananigans fields
#####

"""
    copy_fluxes_to_fields!(radiation::GrayRadiationModel, grid)

Copy RRTMGP flux arrays to Oceananigans ZFaceFields.

Applies sign convention: positive = upward, negative = downward.
For the non-scattering shortwave solver, only the direct beam flux is computed.
"""
function copy_fluxes_to_fields!(radiation::GrayRadiationModel, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)
    
    # Unpack flux arrays from RRTMGP solvers
    lw_flux_up = radiation.longwave_solver.flux.flux_up
    lw_flux_dn = radiation.longwave_solver.flux.flux_dn
    sw_flux_dn_dir = radiation.shortwave_solver.flux.flux_dn_dir

    # Unpack Oceananigans output fields
    ℐ_lw_up = radiation.upwelling_longwave_flux
    ℐ_lw_dn = radiation.downwelling_longwave_flux
    ℐ_sw_dn = radiation.downwelling_shortwave_flux

    launch!(arch, grid, :xyz, _copy_fluxes_kernel!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)

    # Handle top face (k = Nz + 1) separately since kernel only goes to Nz
    ℐ_lw_up[1, 1, Nz + 1] = lw_flux_up[Nz + 1, 1]
    ℐ_lw_dn[1, 1, Nz + 1] = -lw_flux_dn[Nz + 1, 1]
    ℐ_sw_dn[1, 1, Nz + 1] = -sw_flux_dn_dir[Nz + 1, 1]

    return nothing
end

@kernel function _copy_fluxes_kernel!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, 
                                      lw_flux_up, lw_flux_dn, sw_flux_dn_dir, grid)
    i, j, k = @index(Global, NTuple)

    # RRTMGP uses (nlev, ncol), we use (i, j, k) for ZFaceField
    # Sign convention: upwelling positive, downwelling negative
    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, 1]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, 1]  # Negate for downward
        ℐ_sw_dn[i, j, k] = -sw_flux_dn_dir[k, 1]  # Negate for downward
    end
end

# Default no-op for models without radiation
update_radiation!(::Nothing, model) = nothing
