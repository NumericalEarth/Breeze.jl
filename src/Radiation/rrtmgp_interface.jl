using RRTMGP
using RRTMGP: RRTMGPGridParams
using RRTMGP.AngularDiscretizations
using RRTMGP.Fluxes
using RRTMGP.Optics
using RRTMGP.GrayUtils
using RRTMGP.Sources
using RRTMGP.BCs
using ClimaParams

import RRTMGP.Parameters.RRTMGPParameters
import RRTMGP.AtmosphericStates
import RRTMGP.RTE: NoScatLWRTE, TwoStreamLWRTE, NoScatSWRTE, TwoStreamSWRTE

function RRTMGP.RRTMGPGridParams(grid::RectilinearGrid; isothermal_boundary_layer::Bool = false)
    nlay = grid.Nz
    ncols = grid.Nx * grid.Ny
    context = context_from_arch(grid.architecture)
    return RRTMGPGridParams(context, nlay, ncols, isothermal_boundary_layer)
end

function RRTMGP.AtmosphericStates.GrayAtmosphericState(
    grid::RectilinearGrid;
    temperature::Field,
    pressure::Field,
    lat::Field,
    otp,
)
    lat = grid.lat
    nlay = grid.Nz
    ncols = grid.Nx * grid.Ny
    p_lay = reshape(interior(pressure), nlay, ncols)
    p_lev = reshape(interior(Field(@at (Center, Center, Face) pressure)), nlay + 1, ncols)
    t_lay = reshape(interior(temperature), nlay, ncols)
    t_lev = reshape(interior(Field(@at (Center, Center, Face) temperature)), nlay + 1, ncols)
    z_lev = znodes(grid, Face())
    t_sfc = t_lev[1, :]
    otp = otp

    return GrayAtmosphericState(lat, p_lay, p_lev, t_lay, t_lev, z_lev, t_sfc, otp)
end

function RRTMGP.RTE.NoScatLWRTE(
    grid::RectilinearGrid; 
    sfc_emis, 
    inc_flux
)
    grid_params = RRTMGPGridParams(grid)
    params = RRTMGPParameters(eltype(grid))
    (; context) = grid_params
    op = OneScalar(grid_params)
    src = SourceLWNoScat(grid_params; params)
    bcs = LwBCs(sfc_emis, inc_flux)
    fluxb = FluxLW(grid_params)
    flux = FluxLW(grid_params)
    ad = AngularDiscretization(grid_params, 1)
    return NoScatLWRTE(context, op, src, bcs, fluxb, flux, ad)
end

function RRTMGP.RTE.TwoStreamLWRTE(
    grid::RectilinearGrid; 
    sfc_emis, 
    inc_flux
)
    grid_params = RRTMGPGridParams(grid)
    params = RRTMGPParameters(eltype(grid))
    (; context) = grid_params
    op = TwoStream(grid_params)
    src = SourceLW2Str(grid_params; params)
    bcs = LwBCs(sfc_emis, inc_flux)
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
