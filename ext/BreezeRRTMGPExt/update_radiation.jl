#####
##### Update radiation fluxes from model state
#####

"""
    update_radiation!(radiation::GrayRadiationModel, model)

Update the radiative fluxes from the current model state.

This function:
1. Updates the RRTMGP atmospheric state from model fields (T, p)
2. Computes the solar zenith angle from the model clock and grid location
3. Solves the longwave and shortwave RTE
4. Copies the fluxes to Oceananigans fields for output
"""
function Breeze.AtmosphereModels.update_radiation!(radiation::GrayRadiationModel, model)
    grid = model.grid
    clock = model.clock

    # Update atmospheric state from model fields
    update_atmospheric_state!(radiation.atmospheric_state, model, radiation.surface_temperature)

    # Update solar zenith angle from clock
    update_solar_zenith_angle!(radiation.shortwave_solver, grid, clock)

    # Solve longwave RTE
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

    # Copy fluxes to Oceananigans fields
    copy_fluxes_to_fields!(radiation, grid)

    return nothing
end

"""
    update_atmospheric_state!(as::GrayAtmosphericState, model, surface_temperature)

Update the RRTMGP atmospheric state arrays from model fields.
"""
function update_atmospheric_state!(as::GrayAtmosphericState, model, surface_temperature)
    grid = model.grid
    FT = eltype(grid)
    nlay = size(grid, 3)
    nlev = nlay + 1

    # Get temperature and pressure from model
    T = model.temperature
    reference_state = model.formulation.reference_state

    # For single column grid, copy data to RRTMGP format
    # RRTMGP uses (nlay, ncol) with layer 1 at the bottom (surface)
    # Oceananigans uses (1, 1, Nz) with k=1 at the bottom

    # Layer values (at cell centers)
    for k in 1:nlay
        as.t_lay[k, 1] = T[1, 1, k]
        as.p_lay[k, 1] = reference_state.pressure[1, 1, k]
    end

    # Level values (at cell faces)
    # Need to interpolate temperature to faces
    z = znodes(grid, Center())
    zf = znodes(grid, Face())

    for k in 1:nlev
        # Pressure at levels - use reference state
        # For anelastic model, pressure is hydrostatic
        if k == 1
            # Bottom level - extrapolate from first layer
            as.p_lev[k, 1] = reference_state.pressure[1, 1, 1] * 
                             exp((zf[k] - z[1]) * FT(9.81) / (FT(287) * T[1, 1, 1]))
        elseif k == nlev
            # Top level - extrapolate from last layer
            as.p_lev[k, 1] = reference_state.pressure[1, 1, nlay] * 
                             exp((zf[k] - z[nlay]) * FT(9.81) / (FT(287) * T[1, 1, nlay]))
        else
            # Interior levels - interpolate
            as.p_lev[k, 1] = (reference_state.pressure[1, 1, k-1] + reference_state.pressure[1, 1, k]) / 2
        end

        # Temperature at levels - simple interpolation
        if k == 1
            as.t_lev[k, 1] = T[1, 1, 1]  # Use surface layer temperature
        elseif k == nlev
            as.t_lev[k, 1] = T[1, 1, nlay]  # Use top layer temperature
        else
            as.t_lev[k, 1] = (T[1, 1, k-1] + T[1, 1, k]) / 2
        end

        # Altitude at levels
        as.z_lev[k, 1] = zf[k]
    end

    # Surface temperature
    as.t_sfc[1] = surface_temperature

    return nothing
end

"""
    update_solar_zenith_angle!(sw_solver, grid, clock)

Update the solar zenith angle in the shortwave solver from the model clock.
"""
function update_solar_zenith_angle!(sw_solver, grid, clock)
    # Get datetime from clock
    datetime = clock.time

    if datetime isa DateTime
        # Compute cosine of solar zenith angle
        cos_θz = cos_solar_zenith_angle(grid, datetime)
        sw_solver.bcs.cos_zenith[1] = max(cos_θz, 0)  # Clamp to positive (sun above horizon)
    else
        # If clock.time is not a DateTime, use a default
        sw_solver.bcs.cos_zenith[1] = 0.5
    end

    return nothing
end

"""
    copy_fluxes_to_fields!(radiation::GrayRadiationModel, grid)

Copy RRTMGP flux arrays to Oceananigans ZFaceFields.
"""
function copy_fluxes_to_fields!(radiation::GrayRadiationModel, grid::SingleColumnGrid)
    nlay = size(grid, 3)
    nlev = nlay + 1

    # Get RRTMGP flux arrays
    lw_flux = radiation.longwave_solver.flux
    sw_flux = radiation.shortwave_solver.flux

    # Copy to Oceananigans fields
    # RRTMGP uses (nlev, ncol), Oceananigans uses (1, 1, Nz+1) for ZFaceField
    for k in 1:nlev
        radiation.upwelling_longwave_flux[1, 1, k] = lw_flux.flux_up[k, 1]
        radiation.downwelling_longwave_flux[1, 1, k] = lw_flux.flux_dn[k, 1]
        radiation.upwelling_shortwave_flux[1, 1, k] = sw_flux.flux_up[k, 1]
        radiation.downwelling_shortwave_flux[1, 1, k] = sw_flux.flux_dn[k, 1]
    end

    return nothing
end

# Default no-op for models without radiation
Breeze.AtmosphereModels.update_radiation!(::Nothing, model) = nothing

