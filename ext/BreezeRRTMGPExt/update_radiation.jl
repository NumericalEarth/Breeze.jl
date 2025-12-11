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

    # Update RRTMGP atmospheric state from model fields
    update_atmospheric_state!(radiation.atmospheric_state, model, 
                              radiation.surface_temperature, constants)

    # Update solar zenith angle from clock
    update_solar_zenith_angle!(radiation.shortwave_solver, grid, clock)

    # Solve longwave RTE (RRTMGP external call)
    solve_lw!(radiation.longwave_solver, radiation.atmospheric_state)

    # Solve shortwave RTE (only if sun is above horizon)
    cos_θz = radiation.shortwave_solver.bcs.cos_zenith[1]
    if cos_θz > 0
        solve_sw!(radiation.shortwave_solver, radiation.atmospheric_state)
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
    update_atmospheric_state!(as::GrayAtmosphericState, model, surface_temperature, constants)

Update the RRTMGP atmospheric state arrays from model fields.

Maps Oceananigans fields to RRTMGP array format:
- RRTMGP uses (nlay, ncol) for layer values
- RRTMGP uses (nlev, ncol) for level values  
- Layer 1 is at the bottom (surface), layer nlay is at the top
"""
function update_atmospheric_state!(as::GrayAtmosphericState, model, surface_temperature, constants)
    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    
    T = model.temperature
    reference_state = model.formulation.reference_state

    # Extract physical constants
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)

    # Update layer values (cell centers)
    launch!(arch, grid, :xyz, _update_layer_values!, as, grid, T, reference_state)

    # Update level values (cell faces)
    launch!(arch, grid, :xyz, _update_level_values!, as, grid, T, reference_state, g, Rᵈ)

    # Update surface temperature (single value for now)
    as.t_sfc[1] = surface_temperature

    return nothing
end

@kernel function _update_layer_values!(as, grid, T, reference_state)
    i, j, k = @index(Global, NTuple)

    # RRTMGP expects (k, col) indexing, but for single column col=1
    # For 3D grids, we'd need to linearize (i, j) → col
    # For now, support single column (i=j=1)
    @inbounds begin
        as.t_lay[k, 1] = T[i, j, k]
        as.p_lay[k, 1] = reference_state.pressure[i, j, k]
    end
end

@kernel function _update_level_values!(as, grid, T, reference_state, g, Rᵈ)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    nlev = Nz + 1

    # Level values are at cell faces
    # Need to handle bottom (k=1), interior (k=2:Nz), and top (k=Nz+1) differently
    @inbounds begin
        # Level k corresponds to face k
        # Bottom face (k=1): extrapolate downward from first layer center
        if k == 1
            z_center = znode(i, j, 1, grid, Center(), Center(), Center())
            z_face = znode(i, j, 1, grid, Center(), Center(), Face())
            Δz = z_face - z_center  # negative (going down)
            T_center = T[i, j, 1]
            p_center = reference_state.pressure[i, j, 1]
            
            as.p_lev[1, 1] = p_center * exp(-Δz * g / (Rᵈ * T_center))
            as.t_lev[1, 1] = T_center  # Use surface layer temperature
            as.z_lev[1, 1] = z_face
        end

        # Top face (k=Nz+1): extrapolate upward from last layer center
        if k == Nz
            z_center = znode(i, j, Nz, grid, Center(), Center(), Center())
            z_face = znode(i, j, nlev, grid, Center(), Center(), Face())
            Δz = z_face - z_center  # positive (going up)
            T_center = T[i, j, Nz]
            p_center = reference_state.pressure[i, j, Nz]
            
            as.p_lev[nlev, 1] = p_center * exp(-Δz * g / (Rᵈ * T_center))
            as.t_lev[nlev, 1] = T_center  # Use top layer temperature
            as.z_lev[nlev, 1] = z_face
        end

        # Interior faces (k+1 for faces 2 to Nz): interpolate between adjacent centers
        if k < Nz
            kface = k + 1
            # Geometric mean for pressure (log-linear interpolation)
            p_below = reference_state.pressure[i, j, k]
            p_above = reference_state.pressure[i, j, k + 1]
            as.p_lev[kface, 1] = sqrt(p_below * p_above)
            
            # Arithmetic mean for temperature
            T_below = T[i, j, k]
            T_above = T[i, j, k + 1]
            as.t_lev[kface, 1] = (T_below + T_above) / 2
            
            # Altitude at face
            as.z_lev[kface, 1] = znode(i, j, kface, grid, Center(), Center(), Face())
        end
    end
end

#####
##### Update solar zenith angle
#####

"""
    update_solar_zenith_angle!(sw_solver, grid, clock)

Update the solar zenith angle in the shortwave solver from the model clock.
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
    
    lw_flux = radiation.longwave_solver.flux
    sw_flux = radiation.shortwave_solver.flux

    launch!(arch, grid, :xyz, _copy_fluxes_kernel!,
            radiation.upwelling_longwave_flux,
            radiation.downwelling_longwave_flux,
            radiation.downwelling_shortwave_flux,
            lw_flux, sw_flux, grid)

    # Handle top face (k = Nz + 1) separately since kernel only goes to Nz
    Nz = size(grid, 3)
    radiation.upwelling_longwave_flux[1, 1, Nz + 1] = lw_flux.flux_up[Nz + 1, 1]
    radiation.downwelling_longwave_flux[1, 1, Nz + 1] = -lw_flux.flux_dn[Nz + 1, 1]
    radiation.downwelling_shortwave_flux[1, 1, Nz + 1] = -sw_flux.flux_dn_dir[Nz + 1, 1]

    return nothing
end

@kernel function _copy_fluxes_kernel!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux, sw_flux, grid)
    i, j, k = @index(Global, NTuple)

    # RRTMGP uses (nlev, ncol), we use (i, j, k) for ZFaceField
    # Sign convention: upwelling positive, downwelling negative
    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux.flux_up[k, 1]
        ℐ_lw_dn[i, j, k] = -lw_flux.flux_dn[k, 1]  # Negate for downward
        # For NoScatSWRTE, only direct beam is computed
        ℐ_sw_dn[i, j, k] = -sw_flux.flux_dn_dir[k, 1]  # Negate for downward
    end
end

# Default no-op for models without radiation
update_radiation!(::Nothing, model) = nothing
