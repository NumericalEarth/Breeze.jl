"""
RadiativeTransferModel type and constructor for integrating RRTMGP with Breeze.
"""

using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: RectilinearGrid, CPU, GPU
using ClimaComms
using Adapt: adapt

import RRTMGP: RRTMGPSolver, RRTMGPGridParams, GrayRadiation, update_lw_fluxes!, update_sw_fluxes!
import RRTMGP.Parameters: RRTMGPParameters
import RRTMGP.AtmosphericStates: GrayAtmosphericState, GrayOpticalThicknessSchneider2004
import RRTMGP.BCs: LwBCs, SwBCs
import RRTMGP.Fluxes: FluxLW, FluxSW
import RRTMGP.Optics: TwoStream

using ..grid_conversion: create_climacomms_context, grid_to_columns,
                         compute_pressure_levels, compute_layer_pressures,
                         compute_level_temperatures, compute_layer_temperatures,
                         compute_level_altitudes, reshape_to_columns, reshape_from_columns

"""
    SurfaceProperties{FT, FTA1D, FTA2D}

Surface boundary condition properties for radiative transfer.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SurfaceProperties{FT, FTA1D, FTA2D}
    "Surface temperature `[K]`; `(ncol,)`"
    surface_temperature::FTA1D
    "Surface emissivity; `(1, ncol)` or scalar"
    surface_emissivity::FTA2D
    "Surface albedo for direct radiation; `(1, ncol)` or scalar"
    surface_albedo_direct::FTA2D
    "Surface albedo for diffuse radiation; `(1, ncol)` or scalar"
    surface_albedo_diffuse::FTA2D
    "Cosine of solar zenith angle; `(ncol,)`"
    cos_zenith::FTA1D
    "Top of atmosphere solar flux `[W/m²]`; `(ncol,)`"
    toa_solar_flux::FTA1D
    "Top of atmosphere longwave flux `[W/m²]`; `(ncol,)` (optional, defaults to 0)"
    toa_longwave_flux::FTA1D
end

"""
    RadiativeTransferModel{FT, RM, Solver, FluxLW, FluxSW, SfcProps}

Model for computing radiative transfer using RRTMGP.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct RadiativeTransferModel{FT, RM, Solver, FluxLW, FluxSW, SfcProps}
    solver::Solver  # RRTMGPSolver
    flux_lw::FluxLW  # Longwave fluxes
    flux_sw::FluxSW  # Shortwave fluxes
    surface_properties::SfcProps  # Surface boundary condition properties
    grid_params::RRTMGPGridParams  # RRTMGP grid parameters
    atmospheric_state::GrayAtmosphericState  # Current atmospheric state
end

"""
    RadiativeTransferModel(
        grid::RectilinearGrid;
        surface_temperature = nothing,
        surface_emissivity = 0.98,
        surface_albedo_direct = 0.1,
        surface_albedo_diffuse = 0.1,
        cos_zenith = 0.5,
        toa_solar_flux = 1360.0,
        toa_longwave_flux = 0.0,
        optical_thickness_params = nothing
    )

Construct a `RadiativeTransferModel` for gray atmosphere radiation.

# Arguments
- `grid`: Breeze grid
- `surface_temperature`: Surface temperature array `(nx, ny)` or `nothing` to extract from model
- `surface_emissivity`: Surface emissivity (scalar or array)
- `surface_albedo_direct`: Surface albedo for direct radiation (scalar or array)
- `surface_albedo_diffuse`: Surface albedo for diffuse radiation (scalar or array)
- `cos_zenith`: Cosine of solar zenith angle (scalar or array)
- `toa_solar_flux`: Top of atmosphere solar flux in W/m² (scalar or array)
- `toa_longwave_flux`: Top of atmosphere longwave flux in W/m² (scalar or array, defaults to 0)
- `optical_thickness_params`: Gray optical thickness parameters (defaults to Schneider2004)
"""
function RadiativeTransferModel(
    grid::RectilinearGrid;
    surface_temperature = nothing,
    surface_emissivity = 0.98,
    surface_albedo_direct = 0.1,
    surface_albedo_diffuse = 0.1,
    cos_zenith = 0.5,
    toa_solar_flux = 1360.0,
    toa_longwave_flux = 0.0,
    optical_thickness_params = nothing
)
    FT = eltype(grid)
    arch = grid.architecture
    
    # Create ClimaComms context
    context = create_climacomms_context(arch)
    
    # Get grid dimensions
    nlay, ncol = grid_to_columns(grid)
    nlev = nlay + 1
    
    # Create RRTMGP grid parameters
    grid_params = RRTMGPGridParams(FT; context, nlay, ncol)
    
    # Create RRTMGP parameters
    # Note: RRTMGPParameters(FT) requires ClimaParams extension
    # For now, we'll create it manually with default values
    # In practice, this should use the ClimaParams extension if available
    # Try to use the extension if available, otherwise create manually
    if isdefined(RRTMGP.Parameters, :RRTMGPParameters) && 
       hasmethod(RRTMGPParameters, Tuple{Type{FT}})
        params = RRTMGPParameters(FT)
    else
        # Fallback: create parameters manually (values from RRTMGP defaults)
        params = RRTMGPParameters{FT}(
            grav = FT(9.80665),
            molmass_dryair = FT(0.028965),
            molmass_water = FT(0.018015),
            gas_constant = FT(8.314462618),
            kappa_d = FT(0.2854),
            Stefan = FT(5.670374419e-8),
            avogad = FT(6.02214076e23)
        )
    end
    
    # Set up optical thickness parameters
    if optical_thickness_params === nothing
        otp = GrayOpticalThicknessSchneider2004(FT)
    else
        otp = optical_thickness_params
    end
    
    # Create surface properties
    # For now, create placeholder arrays - these will be updated when we have model state
    DA = ClimaComms.array_type(ClimaComms.device(context))
    
    # Latitude array (placeholder - will need to be set properly)
    lat = DA{FT}(zeros(FT, ncol))
    
    # Create initial atmospheric state (will be updated from model)
    # We need pressure levels - for now create placeholder
    p_lev = DA{FT}(undef, nlev, ncol)
    p_lay = DA{FT}(undef, nlay, ncol)
    t_lev = DA{FT}(undef, nlev, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    z_lev = DA{FT}(undef, nlev, ncol)
    t_sfc = DA{FT}(undef, ncol)
    
    # Initialize with zeros for now - will be updated
    fill!(p_lev, zero(FT))
    fill!(p_lay, zero(FT))
    fill!(t_lev, zero(FT))
    fill!(t_lay, zero(FT))
    fill!(z_lev, zero(FT))
    fill!(t_sfc, zero(FT))
    
    atmospheric_state = GrayAtmosphericState{FT, typeof(t_sfc), typeof(p_lev), typeof(otp)}(
        lat,
        p_lay,
        p_lev,
        t_lay,
        t_lev,
        z_lev,
        t_sfc,
        otp
    )
    
    # Create surface property arrays
    sfc_T = if surface_temperature === nothing
        DA{FT}(undef, ncol)  # Will be extracted from model
    else
        # Reshape 2D to 1D column format
        nx, ny = size(grid)[1:2]
        sfc_T_2d = surface_temperature
        sfc_T_1d = DA{FT}(undef, ncol)
        for j in 1:ny, i in 1:nx
            icol = (j - 1) * nx + i
            sfc_T_1d[icol] = sfc_T_2d[i, j]
        end
        sfc_T_1d
    end
    
    # Surface emissivity - expand scalar to array if needed
    sfc_emis = if surface_emissivity isa Number
        DA{FT}(fill(FT(surface_emissivity), 1, ncol))  # (1, ncol) for gray atmosphere
    else
        # Assume it's already in the right format
        DA{FT}(surface_emissivity)
    end
    
    # Surface albedos - expand scalars to arrays if needed
    sfc_alb_dir = if surface_albedo_direct isa Number
        DA{FT}(fill(FT(surface_albedo_direct), 1, ncol))
    else
        DA{FT}(surface_albedo_direct)
    end
    
    sfc_alb_diff = if surface_albedo_diffuse isa Number
        DA{FT}(fill(FT(surface_albedo_diffuse), 1, ncol))
    else
        DA{FT}(surface_albedo_diffuse)
    end
    
    # Solar zenith angle
    cos_zen = if cos_zenith isa Number
        DA{FT}(fill(FT(cos_zenith), ncol))
    else
        DA{FT}(cos_zenith)
    end
    
    # TOA fluxes
    toa_sw = if toa_solar_flux isa Number
        DA{FT}(fill(FT(toa_solar_flux), ncol))
    else
        DA{FT}(toa_solar_flux)
    end
    
    toa_lw = if toa_longwave_flux isa Number
        DA{FT}(fill(FT(toa_longwave_flux), ncol))
    else
        DA{FT}(toa_longwave_flux)
    end
    
    surface_properties = SurfaceProperties(
        sfc_T,
        sfc_emis,
        sfc_alb_dir,
        sfc_alb_diff,
        cos_zen,
        toa_sw,
        toa_lw
    )
    
    # Create boundary conditions
    inc_flux_lw = nothing  # No incoming longwave at TOA by default
    bcs_lw = LwBCs(sfc_emis, inc_flux_lw)
    
    inc_flux_diffuse_sw = nothing
    bcs_sw = SwBCs(cos_zen, toa_sw, sfc_alb_dir, inc_flux_diffuse_sw, sfc_alb_diff)
    
    # Create RRTMGP solver
    radiation_method = GrayRadiation()
    solver = RRTMGPSolver(
        grid_params,
        radiation_method,
        params,
        bcs_lw,
        bcs_sw,
        atmospheric_state;
        op_lw = TwoStream(grid_params),
        op_sw = TwoStream(grid_params)
    )
    
    # Get flux fields from solver
    flux_lw = solver.lws.flux
    flux_sw = solver.sws.flux
    
    return RadiativeTransferModel{FT, typeof(radiation_method), typeof(solver), 
                                  typeof(flux_lw), typeof(flux_sw), typeof(surface_properties)}(
        solver,
        flux_lw,
        flux_sw,
        surface_properties,
        grid_params,
        atmospheric_state
    )
end

"""
    update_atmospheric_state!(rtm::RadiativeTransferModel, model::AtmosphereModel)

Update the atmospheric state in the radiative transfer model from the current
Breeze atmosphere model state.
"""
function update_atmospheric_state!(rtm::RadiativeTransferModel, model)
    grid = model.grid
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    nlev = nlay + 1
    
    FT = eltype(grid)
    
    # Extract reference pressure
    p_ref = model.formulation.reference_state.pressure
    
    # Extract temperature
    T = model.temperature
    
    # Compute pressure levels and layers
    p_lev = compute_pressure_levels(p_ref, grid)
    p_lay = compute_layer_pressures(p_lev)
    
    # Compute temperature levels and layers
    t_lev = compute_level_temperatures(T, grid)
    t_lay = compute_layer_temperatures(T, grid)
    
    # Compute altitude levels
    z_lev = compute_level_altitudes(grid)
    
    # Extract surface temperature
    sfc_T = extract_surface_temperature(T, grid)
    
    # Reshape to column format
    p_lev_cols = reshape_to_columns(p_lev, grid)
    p_lay_cols = reshape_to_columns(p_lay, grid)
    t_lev_cols = reshape_to_columns(t_lev, grid)
    t_lay_cols = reshape_to_columns(t_lay, grid)
    z_lev_cols = reshape_to_columns(z_lev, grid)
    
    # Reshape surface temperature
    sfc_T_cols = similar(sfc_T, ncol)
    for j in 1:ny, i in 1:nx
        icol = (j - 1) * nx + i
        sfc_T_cols[icol] = sfc_T[i, j]
    end
    
    # Update atmospheric state arrays
    # Convert CPU arrays to the appropriate array type for RRTMGP
    # RRTMGP arrays are created using ClimaComms.array_type which matches the device
    as = rtm.atmospheric_state
    DA = typeof(as.p_lev)
    
    # Convert CPU arrays to RRTMGP array type if needed
    # For GPU, we need to adapt arrays to the device
    # Use Adapt.jl to convert arrays to the appropriate type
    p_lev_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), p_lev_cols)
    p_lay_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), p_lay_cols)
    t_lev_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), t_lev_cols)
    t_lay_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), t_lay_cols)
    z_lev_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), z_lev_cols)
    t_sfc_rrtmgp = adapt(ClimaComms.device(rtm.grid_params.context), sfc_T_cols)
    
    # Copy data to atmospheric state arrays
    copyto!(as.p_lev, p_lev_rrtmgp)
    copyto!(as.p_lay, p_lay_rrtmgp)
    copyto!(as.t_lev, t_lev_rrtmgp)
    copyto!(as.t_lay, t_lay_rrtmgp)
    copyto!(as.z_lev, z_lev_rrtmgp)
    copyto!(as.t_sfc, t_sfc_rrtmgp)
    
    # Update surface temperature in surface properties
    copyto!(rtm.surface_properties.surface_temperature, t_sfc_rrtmgp)
    
    return nothing
end

"""
    _update_radiative_fluxes!(rtm::RadiativeTransferModel, model::AtmosphereModel)

Update radiative fluxes by solving the radiative transfer equation
with the current atmospheric state.
"""
function _update_radiative_fluxes!(rtm::RadiativeTransferModel, model)
    # Update atmospheric state from model
    update_atmospheric_state!(rtm, model)
    
    # Solve for longwave and shortwave fluxes using RRTMGP's update methods
    update_lw_fluxes!(rtm.solver)
    update_sw_fluxes!(rtm.solver)
    
    return nothing
end

"""
    compute_radiative_heating_rate(rtm::RadiativeTransferModel, grid, i, j, k)

Compute the radiative heating rate at grid point (i, j, k) from flux differences.
The heating rate is computed as:
    hr = g * (flux_net[k+1] - flux_net[k]) / (cp * Δp)

where flux_net is the net radiative flux (longwave + shortwave).
"""
function compute_radiative_heating_rate(rtm::RadiativeTransferModel, grid, i, j, k)
    # Get net fluxes
    flux_net_lw = rtm.flux_lw.flux_net
    flux_net_sw = rtm.flux_sw.flux_net
    
    # Combine longwave and shortwave
    flux_net_total = flux_net_lw .+ flux_net_sw
    
    # Convert grid indices to column index
    nx, ny, nz = size(grid)
    icol = (j - 1) * nx + i
    
    # Get flux at levels k and k+1 (fluxes are at levels, k is cell index)
    # Level k corresponds to bottom of cell k
    # Level k+1 corresponds to top of cell k
    flux_bottom = flux_net_total[k, icol]
    flux_top = flux_net_total[k+1, icol]
    
    # Get pressure difference
    p_ref = grid  # We'll need to pass this properly
    # For now, use reference pressure from atmospheric state
    p_lev = rtm.atmospheric_state.p_lev
    Δp = p_lev[k+1, icol] - p_lev[k, icol]
    
    # Get constants
    g = 9.81  # gravitational acceleration (m/s²)
    cp = 1004.0  # specific heat capacity (J/kg/K)
    
    # Compute heating rate
    hr = g * (flux_top - flux_bottom) / (cp * Δp)
    
    return hr
end

