#####
##### GrayRadiativeTransferModel: Gray atmosphere radiative transfer model
#####
##### Uses the O'Gorman and Schneider (2008) optical thickness parameterization
##### with RRTMGP's two-stream or no-scattering solvers.
#####
##### Note: GrayRadiativeTransferModel struct is defined in Breeze.AtmosphereModels.radiation_interface
##### This extension provides the constructor and update methods.
#####

using Oceananigans.Grids: znodes

import Breeze: GrayRadiativeTransferModel

"""
    $(TYPEDSIGNATURES)

Construct a gray atmosphere radiative transfer model for the given grid.

Uses the O'Gorman and Schneider (2008) optical thickness parameterization.

# Keyword Arguments
- `latitude`: Latitude in degrees. If `nothing` (default), extracted from grid y-coordinate.
              Can be a scalar (constant for all columns) or a 2D field.
- `surface_temperature`: Surface temperature in Kelvin (default: 300).
                         Can be a scalar or 2D field.
- `surface_emissivity`: Surface emissivity, 0-1 (default: 0.98). Scalar.
- `surface_albedo`: Surface albedo, 0-1 (default: 0.1). Can be scalar or 2D field.
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361)
"""
function GrayRadiativeTransferModel(grid, constants;
                                    stefan_boltzmann_constant = 5.670374419e-8,
                                    avogadro_number = 6.02214076e23,
                                    optical_thickness = nothing,
                                    latitude = nothing,
                                    surface_temperature = 300,
                                    surface_emissivity = 0.98,
                                    surface_albedo = 0.1,
                                    solar_constant = 1361)

    FT = eltype(grid)
    
    # Default optical thickness parameterization
    if isnothing(optical_thickness)
        optical_thickness = GrayOpticalThicknessOGorman2008(FT)
    end
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny

    # Set up RRTMGP grid parameters
    context = rrtmgp_context(arch)
    DA = ClimaComms.array_type(context.device)
    grid_params = RRTMGPGridParams(FT; context, nlay=Nz, ncol=Nc)

    # Create RRTMGP parameters with default values
    kappa_d = constants.dry_air.heat_capacity / constants.dry_air.molar_mass
    params = RRTMGPParameters(;
        grav = FT(constants.gravitational_acceleration),
        molmass_dryair = FT(constants.dry_air.molar_mass),
        molmass_water = FT(constants.vapor.molar_mass),
        gas_constant = FT(constants.molar_gas_constant),
        kappa_d = FT(kappa_d),
        Stefan = FT(stefan_boltzmann_constant),
        avogad = FT(avogadro_number),
    )

    # Allocate RRTMGP arrays: (nlay/nlev, ncol)
    # Note: RRTMGP uses "lat" internally for its GrayAtmosphericState struct
    rrtmgp_latitude = DA{FT}(undef, Nc)
    p_lay = DA{FT}(undef, Nz, Nc)
    t_lay = DA{FT}(undef, Nz, Nc)
    t_lev = DA{FT}(undef, Nz+1, Nc)
    p_lev = DA{FT}(undef, Nz+1, Nc)
    z_lev = DA{FT}(undef, Nz+1, Nc)
    t_sfc = DA{FT}(undef, Nc)

    rrtmgp_latitude .= 0
    set_latitude!(rrtmgp_latitude, latitude, grid)
    
    # Set altitude at cell faces (fixed, doesn't change during simulation)
    # z_lev has shape (Nz+1, Nc) - broadcast z nodes across all columns
    zf = znodes(grid, Face())
    z_lev .= reshape(collect(FT, zf), Nz+1, 1)

    atmospheric_state = GrayAtmosphericState(rrtmgp_latitude, p_lay, p_lev, t_lay, t_lev, z_lev, t_sfc, optical_thickness)

    # Create boundary conditions for longwave
    sfc_emis = DA{FT}(undef, 1, Nc)
    sfc_emis .= FT(surface_emissivity)
    inc_flux_lw = nothing  # No incident longwave flux at TOA

    # Create longwave solver
    longwave_solver = NoScatLWRTE(grid_params; params, sfc_emis, inc_flux = inc_flux_lw)

    # Create boundary conditions for shortwave (will be updated with solar zenith angle)
    cos_zenith = DA{FT}(undef, Nc)
    cos_zenith .= FT(0.5)  # Placeholder, updated during solve
    toa_flux = DA{FT}(undef, Nc)
    toa_flux .= FT(solar_constant)
    sfc_alb_direct = DA{FT}(undef, 1, Nc)
    sfc_alb_diffuse = DA{FT}(undef, 1, Nc)
    inc_flux_diffuse = nothing  # No incident diffuse flux at TOA

    # Initialize albedo (will be updated if it's a field)
    if surface_albedo isa Number
        sfc_alb_direct .= FT(surface_albedo)
        sfc_alb_diffuse .= FT(surface_albedo)
    end

    # Create shortwave solver
    shortwave_solver = NoScatSWRTE(grid_params;
                                   cos_zenith,
                                   toa_flux,
                                   sfc_alb_direct,
                                   inc_flux_diffuse,
                                   sfc_alb_diffuse)

    # Create Oceananigans fields to store fluxes for output/plotting
    upwelling_longwave_flux = ZFaceField(grid)
    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)  # Direct beam only

    # Store surface properties (can be scalar or field)
    if surface_temperature isa Number
        surface_temperature = ConstantField(convert(FT, surface_temperature))
    end

    if surface_albedo isa Number
        surface_albedo = ConstantField(convert(FT, surface_albedo))
    end

    return GrayRadiativeTransferModel(longwave_solver,
                         shortwave_solver,
                         atmospheric_state,
                         optical_thickness,
                         latitude,
                         upwelling_longwave_flux,
                         downwelling_longwave_flux,
                         downwelling_shortwave_flux,
                         surface_temperature,
                         convert(FT, surface_emissivity),
                         surface_albedo,
                         convert(FT, solar_constant))
end

function set_latitude!(rrtmgp_latitude, ::Nothing, grid::SingleColumnGrid)
    # TODO: could launch kernel over xy to support ensembles of columns
    φ = ynode(1, 1, 1, grid, Center(), Center(), Center())
    rrtmgp_latitude .= φ
    return nothing
end

function set_latitude!(rrtmgp_latitude, latitude::Number, grid)
    FT = eltype(grid)
    rrtmgp_latitude .= FT(latitude)
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Create an RRTMGP-compatible ClimaComms context from an Oceananigans architecture.
"""
function rrtmgp_context(arch::CPU)
    device = Threads.nthreads() > 1 ? ClimaComms.CPUMultiThreaded() : ClimaComms.CPUSingleThreaded()
    return ClimaComms.context(device)
end

# GPU support would go here
# function rrtmgp_context(arch::GPU)
#     return ClimaComms.context(ClimaComms.CUDADevice())
# end
