module RRTMGPExt

using Breeze
using RRTMGP

import Breeze.RadiativeTransfer: RadiativeTransferModel

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

function context_from_arch(arch::CPU)
    dev = Threads.nthreads() > 1 ? ClimaComms.CPUMultiThreaded() : ClimaComms.CPUSingleThreaded()
    return ClimaComms.context(dev)    
end

columns(grid) = grid.Nx * grid.Ny

context_from_arch(arch::GPU) = ClimaComms.context(ClimaComms.CUDADevice())

struct RRTMGPRTESolver{AS, LW, SW}
    atmosphere_state :: as
    longwave_solver :: LW
    shortwave_solver :: SW
end

function RadiativeTransferModel(grid, optical_thickness::GrayOpticalThicknessOGorman2008;
                                isothermal_boundary_layer::Bool = false,
                                surface_emissivity = 0.98,
                                direct_surface_albedo = 0.15,
                                diffuse_surface_albedo = 0.15,
                                zenith_angle = 60,
                                incoming_shortwave = 1_361,
                                incoming_longwave = 0)

    state = GrayAtmosphericState(grid, latitude, optical_thickness)

    lw_params = (
        sfc_emission = DA(sfc_emissivity),
        lw_inc_flux = nothing,
        isothermal_boundary_layer = isothermal_boundary_layer,
    )

    sw_params = (
        cos_zenith = DA(cos.(FT(π / 180) .* zenith_angle)),
        toa_flux = DA(toa_sw_flux_inc),
        sfc_alb_direct = DA(sfc_albedo_direct),
        inc_flux_diffuse = nothing,
        sfc_alb_diffuse = DA(sfc_albedo_diffuse),
        isothermal_boundary_layer = isothermal_boundary_layer,
    )

    longwave_solver = TwoStreamLWRTE(grid; lw_params...)
    shortwave_solver = TwoStreamSWRTE(grid; sw_params...)

    solver = RRTMGPRTESolver(state, longwave_solver, shortwave_solver)

    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)

    return RadiativeTransferModel(solver,
                                  downwelling_longwave_flux,
                                  downwelling_shortwave_flux,
                                  zenith_angle,
                                  surface_emissivity,
                                  direct_surface_albedo,
                                  diffuse_surface_albedo,
                                  incoming_shortwave,
                                  incoming_longwave)
end


"""
    RRTMGPGridParams(grid; isothermal_boundary_layer=false)

Construct RRTMGP grid parameters and a compute context compatible with the
grid's architecture. `Ncols = Nx * Ny` follows RRTMGP's column-major layout.
"""
function RRTMGP.RRTMGPGridParams(grid; isothermal_boundary_layer::Bool = false)
    FT = eltype(grid)
    Nz = size(grid, 3)
    context = context_from_arch(grid.architecture)
    Ncols = columns(grid)
    return RRTMGPGridParams{FT, typeof(context)}(context, Nz, Ncols, isothermal_boundary_layer)
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
function RRTMGP.AtmosphericStates.GrayAtmosphericState(grid, latitude, otp)
    DA = array_type(grid.architecture)
    FT = eltype(grid)
    Nz = size(grid, 3)
    Ncols = ncols(grid)

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


    # Default: reshape the provided fields into RRTMGP's column layout

"""
    RTE.TwoStreamLWRTE(grid; sfc_emission, lw_inc_flux)

Construct the two-stream longwave RTE. Arrays are reshaped to RRTMGP's column
layout internally:
- `sfc_emission`: (Nbnd, Nx, Ny) → (Nbnd, Ncols)
- `lw_inc_flux`: nothing or (Nbnd, Nx, Ny) → (Nbnd, Ncols)
"""
function RRTMGP.RTE.TwoStreamLWRTE(
    grid::AbstractGrid,
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

#=
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
=#

end # module

