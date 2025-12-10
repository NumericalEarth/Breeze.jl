#####
##### GrayRadiation: Gray atmosphere radiative transfer model
#####
##### Uses the O'Gorman and Schneider (2008) optical thickness parameterization
##### with RRTMGP's two-stream or no-scattering solvers.
#####

const SingleColumnGrid = Oceananigans.Grids.AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

"""
    GrayRadiationModel{LW, SW, AS, FT, F}

Gray atmosphere radiative transfer model using RRTMGP.

# Fields
- `longwave_solver`: RRTMGP longwave RTE solver
- `shortwave_solver`: RRTMGP shortwave RTE solver
- `atmospheric_state`: RRTMGP GrayAtmosphericState
- `upwelling_longwave_flux`: ZFaceField for upwelling LW flux (W/m²)
- `downwelling_longwave_flux`: ZFaceField for downwelling LW flux (W/m²)
- `downwelling_shortwave_flux`: ZFaceField for direct-beam SW flux (W/m²)
- `surface_temperature`: Surface temperature (K)
- `surface_emissivity`: Surface emissivity (0-1)
- `surface_albedo`: Surface albedo (0-1)
- `solar_constant`: Top-of-atmosphere solar flux (W/m²)

!!! note "Shortwave radiation"
    The gray atmosphere uses a non-scattering shortwave approximation, so only
    the direct beam flux (`downwelling_shortwave_flux`) is computed. There is
    no diffuse shortwave or upwelling shortwave in this model.
"""
struct GrayRadiationModel{LW, SW, AS, FT, F}
    longwave_solver :: LW
    shortwave_solver :: SW
    atmospheric_state :: AS
    upwelling_longwave_flux :: F
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F  # Direct beam only for no-scattering solver
    surface_temperature :: FT
    surface_emissivity :: FT
    surface_albedo :: FT
    solar_constant :: FT
end

"""
    GrayRadiation(grid;
                  surface_temperature = 300,
                  surface_emissivity = 0.98,
                  surface_albedo = 0.1,
                  solar_constant = 1361)

Construct a gray atmosphere radiative transfer model for the given grid.

Uses the O'Gorman and Schneider (2008) optical thickness parameterization.

# Arguments
- `grid`: Oceananigans grid (currently only single-column grids are supported)

# Keyword Arguments
- `surface_temperature`: Surface temperature in Kelvin (default: 300)
- `surface_emissivity`: Surface emissivity, 0-1 (default: 0.98)
- `surface_albedo`: Surface albedo, 0-1 (default: 0.1)
- `solar_constant`: Top-of-atmosphere solar flux in W/m² (default: 1361)
"""
function Breeze.GrayRadiation(grid::SingleColumnGrid;
                               surface_temperature = 300,
                               surface_emissivity = 0.98,
                               surface_albedo = 0.1,
                               solar_constant = 1361)

    FT = eltype(grid)
    arch = Oceananigans.architecture(grid)
    nlay = size(grid, 3)
    nlev = nlay + 1
    ncol = 1  # Single column

    # Set up RRTMGP grid parameters
    context = rrtmgp_context(arch)
    DA = ClimaComms.array_type(context.device)
    grid_params = RRTMGPGridParams(FT; context, nlay, ncol)

    # Create RRTMGP parameters with default values
    params = RRTMGPParameters(;
        grav = FT(9.80665),
        molmass_dryair = FT(0.028964),
        molmass_water = FT(0.018015),
        gas_constant = FT(8.3144598),
        kappa_d = FT(2/7),
        Stefan = FT(5.670374419e-8),
        avogad = FT(6.02214076e23)
    )

    # Create optical thickness parameterization
    otp = GrayOpticalThicknessOGorman2008(FT)

    # Create atmospheric state (will be updated from model fields)
    atmospheric_state = create_gray_atmospheric_state(grid, otp, DA)

    # Create boundary conditions for longwave
    sfc_emis = DA{FT}(undef, 1, ncol)
    sfc_emis .= FT(surface_emissivity)
    inc_flux_lw = nothing  # No incident longwave flux at TOA

    # Create longwave solver
    longwave_solver = NoScatLWRTE(grid_params; params, sfc_emis, inc_flux = inc_flux_lw)

    # Create boundary conditions for shortwave (will be updated with solar zenith angle)
    cos_zenith = DA{FT}(undef, ncol)
    cos_zenith .= FT(0.5)  # Placeholder, updated during solve
    toa_flux = DA{FT}(undef, ncol)
    toa_flux .= FT(solar_constant)
    sfc_alb_direct = DA{FT}(undef, 1, ncol)
    sfc_alb_direct .= FT(surface_albedo)
    sfc_alb_diffuse = DA{FT}(undef, 1, ncol)
    sfc_alb_diffuse .= FT(surface_albedo)
    inc_flux_diffuse = nothing  # No incident diffuse flux at TOA

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

    return GrayRadiationModel(longwave_solver,
                              shortwave_solver,
                              atmospheric_state,
                              upwelling_longwave_flux,
                              downwelling_longwave_flux,
                              downwelling_shortwave_flux,
                              FT(surface_temperature),
                              FT(surface_emissivity),
                              FT(surface_albedo),
                              FT(solar_constant))
end

"""
    rrtmgp_context(arch)

Create an RRTMGP-compatible ClimaComms context from an Oceananigans architecture.
"""
function rrtmgp_context(arch::Oceananigans.CPU)
    device = Threads.nthreads() > 1 ? ClimaComms.CPUMultiThreaded() : ClimaComms.CPUSingleThreaded()
    return ClimaComms.context(device)
end

# GPU support would go here
# function rrtmgp_context(arch::Oceananigans.GPU)
#     return ClimaComms.context(ClimaComms.CUDADevice())
# end

"""
    create_gray_atmospheric_state(grid, otp, DA)

Create an RRTMGP GrayAtmosphericState for the given grid.
The arrays are allocated but not initialized - they will be filled
from model fields during update_radiation!.
"""
function create_gray_atmospheric_state(grid::SingleColumnGrid, otp, DA)
    FT = eltype(grid)
    nlay = size(grid, 3)
    nlev = nlay + 1
    ncol = 1

    # Get latitude from grid (y-coordinate for single column)
    φ = grid.yᵃᶜᵃ[1]

    # Allocate arrays in RRTMGP layout: (nlay/nlev, ncol)
    lat = DA{FT}(undef, ncol)
    lat .= FT(φ)

    p_lay = DA{FT}(undef, nlay, ncol)
    p_lev = DA{FT}(undef, nlev, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    t_lev = DA{FT}(undef, nlev, ncol)
    z_lev = DA{FT}(undef, nlev, ncol)
    t_sfc = DA{FT}(undef, ncol)

    return GrayAtmosphericState(lat, p_lay, p_lev, t_lay, t_lev, z_lev, t_sfc, otp)
end

Base.summary(radiation::GrayRadiationModel) = "GrayRadiationModel"

function Base.show(io::IO, radiation::GrayRadiationModel)
    print(io, summary(radiation), "\n",
          "├── surface_temperature: ", radiation.surface_temperature, " K\n",
          "├── surface_emissivity: ", radiation.surface_emissivity, "\n",
          "├── surface_albedo: ", radiation.surface_albedo, "\n",
          "└── solar_constant: ", radiation.solar_constant, " W/m²")
end

