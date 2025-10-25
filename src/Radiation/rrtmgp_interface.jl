using RRTMGP
using RRTMGP: RRTMGPGridParams
using RRTMGP.AngularDiscretizations
using RRTMGP.Fluxes
using RRTMGP.Optics
using RRTMGP.GrayUtils: setup_gray_as_pr_grid
using RRTMGP.Sources
using RRTMGP.BCs
using RRTMGP.Parameters: RRTMGPParameters
using RRTMGP.Optics: GrayOpticalThicknessOGorman2008

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans: RectilinearGrid
using Oceananigans: field
using ClimaParams
using ClimaComms

using Breeze: ncols, array_type, CPU, GPU

# Import RRTMGP types for extensions
import RRTMGP.Parameters.RRTMGPParameters
import RRTMGP.AtmosphericStates
import RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE

"""
    RRTMGPGridParams(grid; isothermal_boundary_layer=false)

Construct RRTMGP grid parameters and a compute context compatible with the
grid's architecture. `Ncols = Nx * Ny` follows RRTMGP's column-major layout.
"""
function RRTMGP.RRTMGPGridParams(grid::RectilinearGrid; isothermal_boundary_layer::Bool = false)
    FT = eltype(grid)
    Nz = size(grid, 3)
    context = context_from_arch(grid.architecture)
    return RRTMGPGridParams{FT, typeof(context)}(context, Nz, ncols(grid), isothermal_boundary_layer)
end

"""
    AtmosphericStates.GrayAtmosphericState(grid; temperature, pressure, otp,
                                           lat_center=0, planet_radius=6_371_000)

Build a gray-atmosphere state from Oceananigans `Field`s, reshaping data into
the column-major layout used by RRTMGP. Shapes:
- `lat`: (Ncols)
- `p_lay`, `t_lay`: (Nz, Ncols)
- `p_lev`, `t_lev`, `z_lev`: (Nz+1, Ncols)
"""
function RRTMGP.AtmosphericStates.GrayAtmosphericState(
    grid :: RectilinearGrid;
    latitude :: AbstractArray, 
)
    DA = array_type(grid.architecture)
    FT = eltype(grid)
    Nz = size(grid, 3)
    Ncols = ncols(grid)

    # Default optical thickness parameters from O'Gorman 2008
    otp = GrayOpticalThicknessOGorman2008(FT)
    
    # Default: reshape the provided fields into RRTMGP's column layout
    lat = reshape(latitude, Ncols)
    p_lay = DA{FT}(undef, Nz, Ncols)
    p_lev = DA{FT}(undef, Nz + 1, Ncols)
    t_lay = DA{FT}(undef, Nz, Ncols)
    t_lev = DA{FT}(undef, Nz + 1, Ncols)
    z_lev = DA{FT}(undef, Nz + 1, Ncols)
    t_sfc = DA{FT}(undef, Ncols)

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

"""
    RTE.TwoStreamLWRTE(grid; sfc_emission, lw_inc_flux)

Construct the two-stream longwave RTE. Arrays are reshaped to RRTMGP's column
layout internally:
- `sfc_emission`: (Nbnd, Nx, Ny) → (Nbnd, Ncols)
- `lw_inc_flux`: nothing or (Nbnd, Nx, Ny) → (Nbnd, Ncols)
"""
function RRTMGP.RTE.TwoStreamLWRTE(
    grid::RectilinearGrid; 
    sfc_emission,
    lw_inc_flux,
    isothermal_boundary_layer
)
    Ncols = ncols(grid)
    Nbnd = size(sfc_emission, 1)
    grid_params = RRTMGPGridParams(grid; isothermal_boundary_layer)
    params = RRTMGPParameters(eltype(grid))
    # TODO!: need helpers here for reshaping fields to column layout
    sfc_emis = reshape(sfc_emission, Nbnd, Ncols)
    lw_inc_flux = lw_inc_flux isa Nothing ? nothing : reshape(lw_inc_flux, Nbnd, Ncols)

    return TwoStreamLWRTE(grid_params; params, sfc_emis, inc_flux=lw_inc_flux)
end

"""
    RTE.TwoStreamSWRTE(grid; 
                       cos_zenith, 
                       toa_flux, 
                       sfc_alb_direct,
                       inc_flux_diffuse, 
                       sfc_alb_diffuse, 
                       inc_flux_diffuse, 
                       isothermal_boundary_layer)

Construct the two-stream shortwave RTE. Arrays are reshaped to column layout:
- `cos_zenith`, `toa_flux`: (Nx, Ny) → (Ncols)
- `sfc_alb_direct`, `sfc_alb_diffuse`: (Nbnd, Nx, Ny) → (Nbnd, Ncols)
- `inc_flux_diffuse`: nothing or (Nbnd, Nx, Ny) → (Nbnd, Ncols)
- `isothermal_boundary_layer`: Bool
"""
function RRTMGP.RTE.TwoStreamSWRTE(
    grid::RectilinearGrid;
    cos_zenith,
    toa_flux,
    sfc_alb_direct,
    sfc_alb_diffuse,
    inc_flux_diffuse,
    isothermal_boundary_layer
)
    Ncols = ncols(grid)
    Nbnd = size(sfc_alb_direct, 1)
    grid_params = RRTMGPGridParams(grid; isothermal_boundary_layer)

    # TODO!: need helpers here for reshaping fields to column layout
    cosz = reshape(cos_zenith, Ncols)
    toa = reshape(toa_flux, Ncols)
    alb_dir = reshape(sfc_alb_direct, Nbnd, Ncols)
    alb_diff = reshape(sfc_alb_diffuse, Nbnd, Ncols)
    inc_flux_diffuse = inc_flux_diffuse isa Nothing ? nothing : reshape(inc_flux_diffuse, Nbnd, Ncols)

    return TwoStreamSWRTE(
        grid_params; 
        cos_zenith=cosz, 
        toa_flux=toa, 
        sfc_alb_direct=alb_dir, 
        inc_flux_diffuse=inc_flux_diffuse,
        sfc_alb_diffuse=alb_diff,
    )
end

"""
    update_atmospheric_state!(as::GrayAtmosphericState, temperature::Field, pressure::Field)

Update the atmospheric state with new temperature and pressure fields.
"""
function update_atmospheric_state!(as::RRTMGP.AtmosphericStates.GrayAtmosphericState, temperature::Field, pressure::Field)
    grid = temperature.grid
    Nz = size(grid, 3)
    Ncols = ncols(grid)

    # TODO!: need helpers here for reshaping fields to column layout
    as.p_lay .= reshape(PermutedDimsArray(interior(pressure), (3, 1, 2)), Nz, Ncols)
    as.p_lev .= reshape(PermutedDimsArray(interior(Field(@at (Center, Center, Face) pressure)), (3, 1, 2)), Nz + 1, Ncols)
    as.t_lay .= reshape(PermutedDimsArray(interior(temperature), (3, 1, 2)), Nz, Ncols)
    as.t_lev .= reshape(PermutedDimsArray(interior(Field(@at (Center, Center, Face) temperature)), (3, 1, 2)), Nz + 1, Ncols)
    as.t_sfc .= view(as.t_lev, 1, :)

    return nothing
end

"""
    context_from_arch(arch)

Return a `ClimaComms` execution context compatible with the provided
Oceananigans architecture (CPU single/multithreaded or GPU).
"""
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
