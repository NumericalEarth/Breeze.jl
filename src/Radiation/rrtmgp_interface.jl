using RRTMGP
using RRTMGP: RRTMGPGridParams
using RRTMGP.AngularDiscretizations
using RRTMGP.Fluxes
using RRTMGP.Optics
using RRTMGP.GrayUtils: setup_gray_as_pr_grid
using RRTMGP.Sources
using RRTMGP.BCs
using RRTMGP.Parameters: RRTMGPParameters
using RRTMGP.Optics: GrayOpticalThicknessSchneider2004

using Oceananigans
using Oceananigans: RectilinearGrid
using Oceananigans: field
using ClimaParams
using ClimaComms

import RRTMGP.Parameters.RRTMGPParameters
import RRTMGP.AtmosphericStates
import RRTMGP.RTE: NoScatLWRTE, TwoStreamLWRTE, NoScatSWRTE, TwoStreamSWRTE

function latitude(grid::RectilinearGrid; lat_center=0, planet_radius=6_371_000, unit=:degrees)
    FT = eltype(grid)
    DA = array_type(grid.architecture)
    Nx = grid.Nx

    y = ynodes(grid, Center())
    lat = lat_center .+ y / planet_radius * (unit == :degrees ? 180/π : 1)
    lat = DA{FT}(repeat(lat', Nx, 1))

    return lat
end

function RRTMGP.RRTMGPGridParams(grid::RectilinearGrid; isothermal_boundary_layer::Bool = false)
    FT = eltype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Ncols = Nz * Ny
    context = context_from_arch(grid.architecture)
    return RRTMGPGridParams{FT, typeof(context)}(context, Nz, Ncols, isothermal_boundary_layer)
end

function RRTMGP.AtmosphericStates.GrayAtmosphericState(
    grid::RectilinearGrid;
    temperature::Field,
    pressure::Field,
    otp,
    lat_center=0, 
    planet_radius=6_371_000,
)
    FT = eltype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Ncols = Nx * Ny
    lat = reshape(latitude(grid; lat_center, planet_radius), Ncols)
    p_lay = reshape(interior(pressure), Nz, Ncols)
    p_lev = reshape(interior(Field(@at (Center, Center, Face) pressure)), Nz + 1, Ncols)
    t_lay = reshape(interior(temperature), Nz, Ncols)
    t_lev = reshape(interior(Field(@at (Center, Center, Face) temperature)), Nz + 1, Ncols)
    z_lev = reshape(view(repeat(znodes(grid, Face()), Nx, Ny, 1), :, :, :), Nz + 1, Ncols)
    t_sfc = t_lev[1, :]

    return GrayAtmosphericState{FT, AbstractVector{FT}, AbstractMatrix{FT}, typeof(otp)}(
        lat, 
        p_lay, 
        p_lev, 
        t_lay, 
        t_lev, 
        z_lev, 
        t_sfc, 
        otp
    )
end

function RRTMGP.RTE.NoScatLWRTE(
    grid::RectilinearGrid; 
    sfc_emission, 
    lw_inc_flux
)
    grid_params = RRTMGPGridParams(grid)
    params = RRTMGPParameters(eltype(grid))
    (; context) = grid_params
    op = OneScalar(grid_params)
    src = SourceLWNoScat(grid_params; params)
    bcs = LwBCs(sfc_emission, lw_inc_flux)
    fluxb = FluxLW(grid_params)
    flux = FluxLW(grid_params)
    ad = AngularDiscretization(grid_params, 1)
    return NoScatLWRTE(context, op, src, bcs, fluxb, flux, ad)
end

function RRTMGP.RTE.TwoStreamLWRTE(
    grid::RectilinearGrid; 
    sfc_emission, 
    lw_inc_flux
)
    FT = eltype(grid)
    Nx, Ny = grid.Nx, grid.Ny
    Ncols = Nx * Ny
    grid_params = RRTMGPGridParams(grid)
    params = RRTMGPParameters(eltype(grid))
    (; context) = grid_params
    op = TwoStream(grid_params)
    src = SourceLW2Str(grid_params; params)
    if lw_inc_flux !== nothing
        lw_inc_flux = reshape(lw_inc_flux, 1, Ncols)
    end
    bcs = LwBCs(reshape(sfc_emission, 1, Ncols), lw_inc_flux)
    fluxb = FluxLW(grid_params)
    flux = FluxLW(grid_params)
    return TwoStreamLWRTE(context, op, src, bcs, fluxb, flux)
end

function RRTMGP.RTE.NoScatSWRTE(
    grid::RectilinearGrid;
    cos_zenith,
    toa_flux,
    sfc_alb_direct,
    inc_flux_diffuse,
    sfc_alb_diffuse,
)
    grid_params = RRTMGP.RRTMGPGridParams(grid)
    (; context) = grid_params
    op = OneScalar(grid_params)
    bcs = SwBCs(cos_zenith, toa_flux, sfc_alb_direct, inc_flux_diffuse, sfc_alb_diffuse)
    fluxb = FluxSW(grid_params)
    flux = FluxSW(grid_params)
    return NoScatSWRTE(context, op, bcs, fluxb, flux)
end

function RRTMGP.RTE.TwoStreamSWRTE(
    grid::RectilinearGrid;
    cos_zenith,
    toa_flux,
    sfc_alb_direct,
    inc_flux_diffuse,
    sfc_alb_diffuse,
)
    grid_params = RRTMGPGridParams(grid)
    (; context) = grid_params
    op = TwoStream(grid_params)
    src = SourceSW2Str(grid_params)
    bcs = SwBCs(cos_zenith, toa_flux, sfc_alb_direct, inc_flux_diffuse, sfc_alb_diffuse)
    fluxb = FluxSW(grid_params)
    flux = FluxSW(grid_params)
    return TwoStreamSWRTE(context, op, src, bcs, fluxb, flux)
end

function context_from_arch(arch)
    if arch isa CPU
        dev = Threads.nthreads() > 1 ?
              ClimaComms.CPUMultiThreaded() :
              ClimaComms.CPUSingleThreaded()
        return ClimaComms.context(dev)    
    elseif arch isa GPU
        return ClimaComms.context(ClimaComms.CUDADevice())
    end
end

function gray_test_t_p_profiles(grid::RectilinearGrid; p0, pe)
    # get data types and context from grid
    FT = eltype(grid)
    context = context_from_arch(grid.architecture)
    DA = array_type(grid.architecture)

    # derive latitude from grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    ncol = grid.Nx * grid.Ny
    if ncol == 1
        lat = DA{FT}([0])                   
    else
        lat = DA{FT}(range(FT(-90), FT(90), length = ncol))
    end
    
    # get test case temperature and pressure profiles
    param_set = RRTMGPParameters(FT)
    otp = GrayOpticalThicknessSchneider2004(FT)
    gray_as = setup_gray_as_pr_grid(context, Nz, lat, FT(p0), FT(pe), otp, param_set, DA)
    p_lay_as_array = reshape(gray_as.p_lay', Nx, Ny, Nz)
    t_lay_as_array = reshape(gray_as.t_lay', Nx, Ny, Nz)

    # create Fields from test case temperature and pressure profiles
    pressure = CenterField(grid)
    temperature = CenterField(grid)
    set!(pressure, p_lay_as_array)
    set!(temperature, t_lay_as_array)

    return pressure, temperature
end


