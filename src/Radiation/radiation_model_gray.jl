using RRTMGP.Optics: GrayOpticalThicknessSchneider2004
using RRTMGP.AtmosphericStates: GrayAtmosphericState
using RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!

using Oceananigans
using Oceananigans: field
using Oceananigans.Architectures: array_type
using Oceananigans: RectilinearGrid

mutable struct GrayRadiationModel{FT, DA, SLVLW, SLVSW, AS, OTP} <: AbstractRadiationModel
    atmospheric_state :: AS
    slv_lw :: SLVLW
    slv_sw :: SLVSW
    optical_properties :: OTP
    cos_zenith_angle :: DA{FT}
    sfc_emissivity :: DA{FT}
    sfc_albedo_direct :: DA{FT}
    sfc_albedo_diffuse :: DA{FT}
    sw_toa_flux_inc :: DA{FT}
    sw_inc_flux_diffuse :: Union{Nothing, DA{FT}}
    lw_toa_inc_flux :: Union{Nothing, DA{FT}}
end

function GrayRadiationModel(
    grid; 
    temperature, 
    pressure,
    cos_zenith_angle,
    sfc_emissivity,
    sfc_albedo_direct,
    sfc_albedo_diffuse,
    sw_flux_inc_toa,
    sw_flux_inc_toa_diffusive,
    lw_flux_inc_toa,
    optical_properties=GrayOpticalThicknessSchneider2004(FT),
    lat_center=0
)  
    # We assemble objects required for RRTMGP to work.
    atmospheric_state = GrayAtmosphericState(
        grid; 
        temperature, 
        pressure, 
        otp=optical_properties, 
        lat_center=lat_center
    )
    SLVLW = TwoStreamLWRTE
    SLVSW = TwoStreamSWRTE
    lw_params = (
        sfc_emission = sfc_emissivity, 
        lw_inc_flux = lw_flux_inc_toa,
    )
    sw_params = (
        cos_zenith = cos_zenith_angle, 
        toa_flux = sw_flux_inc_toa,
        sfc_alb_direct = sfc_albedo_diffuse,
        inc_flux_diffuse = sw_flux_inc_toa_diffusive,
        sfc_alb_diffuse = sfc_albedo_diffuse,
    )
    slv_lw = SLVLW(grid; lw_params...)
    slv_sw = SLVSW(grid; sw_params...)
    
    if sfc_emissivity isa DA

    elseif sfc_emissivity isa AbstractFloat

    end

    return GrayRadiationModel(
        atmospheric_state,
        slv_lw, 
        slv_sw,
        optical_properties,
        cos_zenith_angle,
        sfc_emission,
        sfc_albedo_direct,
        sfc_albedo_diffuse,
        sw_flux_inc_toa, 
        sw_flux_inc_toa_diffusive,
        lw_flux_inc_toa,
    )
end

function GrayRadiationModel(
    grid; 
    temperature :: Field, 
    pressure :: Field,
    zenith_angle :: FT,
    sfc_emissivity :: FT,
    sfc_albedo_direct :: FT,
    sfc_albedo_diffuse :: FT,
    sw_flux_inc_toa :: FT,
    sw_flux_inc_toa_diffusive :: Union{Nothing, FT},
    lw_flux_inc_toa :: Union{Nothing, FT},
    optical_properties=GrayOpticalThicknessSchneider2004(FT),
    lat_center=0
)
    FT = eltype(grid)
    DA = device_array(grid.architecture)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Nbnd = 1


    sfc_emission = DA{FT}(undef, Nbnd, Nx, Ny)
    sfc_alb_direct = DA{FT}(undef, Nbnd, Nx, Ny)
    sfc_alb_diffuse = DA{FT}(undef, Nbnd, Nx, Ny)
    cos_zenith = DA{FT}(undef, Nx, Ny)
    toa_flux = DA{FT}(undef, Nx, Ny)
    lw_toa_inc_flux = nothing
    inc_flux_diffuse = nothing
    fill!(sfc_emission, FT(sfc_emissivity))
    fill!(sfc_alb_direct, FT(albedo_direct))
    fill!(sfc_alb_diffuse, FT(albedo_diffuse))
    fill!(cos_zenith, FT(cos(deg2rad * zenith_angle)))
    fill!(toa_flux, FT(sw_inc_flux))
end

function (rad::GrayRadiationModel)(::Val{:ρe}, temperature, pressure)
    update_atmospheric_state(rad, temperature, pressure)
    solve_lw!(rad.slv_lw, rad.atmospheric_state)
    solve_sw!(rad.slv_sw, rad.atmospheric_state)    
end
