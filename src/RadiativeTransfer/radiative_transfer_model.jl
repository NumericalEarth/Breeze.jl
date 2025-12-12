using RRTMGP.AtmosphericStates: GrayAtmosphericState
using RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!

using Breeze: field, ZFaceField, RectilinearGrid, set!, ncols, array_type
using Breeze.Radiation: AbstractRadiationModel, update_atmospheric_state!

"""
GrayRadiativeTransferModelModel stores state and solver handles for a gray-band radiative
transfer setup.
Field conventions (array shapes):
- cos_zenith_angle: (Nx, Ny) — cos of zenith angle on the horizontal grid
- sfc_emissivity: (Nbnd=1, Nx, Ny) — surface emissivity per band
- sfc_albedo_direct: (Nbnd=1, Nx, Ny) — direct-beam surface albedo per band
- sfc_albedo_diffuse: (Nbnd=1, Nx, Ny) — diffuse surface albedo per band
- toa_sw_flux_inc: (Nx, Ny) — incoming shortwave flux at TOA
"""

"""
    GrayRadiativeTransferModelModel(grid; 
                       temperature,
                       pressure,
                       zenith_angle, 
                       sfc_emissivity, 
                       sfc_albedo_direct,
                       sfc_albedo_diffuse, 
                       toa_sw_flux_inc)

Construct a gray-band model using a precomputed atmospheric state.
Inputs may be scalars or arrays and are normalized to device arrays.
"""
function GrayRadiativeTransferModelModel(
    grid;
    zenith_angle,
    sfc_emissivity,
    sfc_albedo_direct,
    sfc_albedo_diffuse,
    toa_sw_flux_inc,
    latitude,
    isothermal_boundary_layer=false,
)
    DA = array_type(grid.architecture)
    FT = eltype(grid)

    # Create atmospheric state from provided Fields (generic library behavior)
    # Allocated the required memory for the atmospheric state inside the
    # RRTMGP.jl atmospheric state
    rrtmgp_atmospheric_state = GrayAtmosphericState(
        grid;
        latitude=latitude,
    )

    # Build solvers using high-level wrappers 
    # They reshape internally for RRTMGP.jl compatibility to match the column layout
    # used by RRTMGP.jl
    SLVLW = TwoStreamLWRTE
    SLVSW = TwoStreamSWRTE
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
    rrtmgp_solver_lw = SLVLW(grid; lw_params...)
    rrtmgp_solver_sw = SLVSW(grid; sw_params...)

    # Create Fields for radiative fluxes to store the radiative fluxes from
    # the RRTMGP.jl solvers for native compatibility with Oceananigans operators
    downwelling_longwave_flux = ZFaceField(grid)
    downwelling_shortwave_flux = ZFaceField(grid)

    return GrayRadiativeTransferModelModel(
        downwelling_longwave_flux,
        downwelling_shortwave_flux,
        sw_params.cos_zenith,
        lw_params.sfc_emission,
        sw_params.sfc_alb_direct,
        sw_params.sfc_alb_diffuse,
        sw_params.toa_flux,
        rrtmgp_atmospheric_state,
        rrtmgp_solver_lw,
        rrtmgp_solver_sw,
    )
end

"""
    (model::GrayRadiativeTransferModelModel)(temperature::Field, pressure::Field)

Update the radiative fluxes for the given `GrayRadiativeTransferModelModel` by running the
longwave and shortwave two-stream solvers with the current atmospheric state
and boundary conditions.
"""
function (model::GrayRadiativeTransferModelModel)(temperature::Field, pressure::Field)
    # Update the atmospheric state inside the RRTMGP.jl atmospheric state
    # This is needed before running the solvers to make sure the atmospheric state is updated
    # before computing the radiative fluxes
    update_atmospheric_state!(model.rrtmgp_atmospheric_state, temperature, pressure)

    # Compute the radiative fluxes inside the RRTMGP.jl solvers
    solve_lw!(model.rrtmgp_solver_lw, model.rrtmgp_atmospheric_state)
    solve_sw!(model.rrtmgp_solver_sw, model.rrtmgp_atmospheric_state)

    # Update the radiative flux Field objects with the radiative fluxes from the RRTMGP.jl solvers
    Nx, Ny, Nz = size(model.downwelling_longwave_flux.grid)
    set!(model.downwelling_longwave_flux, permutedims(reshape(model.rrtmgp_solver_lw.flux.flux_net, Nz+1, Nx, Ny), (2, 3, 1)))
    set!(model.downwelling_shortwave_flux, permutedims(reshape(model.rrtmgp_solver_sw.flux.flux_net, Nz+1, Nx, Ny), (2, 3, 1)))

    return model.downwelling_longwave_flux, model.downwelling_shortwave_flux
end
