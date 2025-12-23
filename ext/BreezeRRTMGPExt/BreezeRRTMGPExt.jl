module BreezeRRTMGPExt

using Breeze
using RRTMGP: RRTMGP

using Dates: DateTime
using DocStringExtensions: TYPEDSIGNATURES

# Oceananigans imports
using Oceananigans.Architectures: architecture, CPU
using Oceananigans.Fields: ZFaceField

# RRTMGP imports (external types - cannot modify)
#   GrayAtmosphericState: atmospheric state arrays (t_lay, p_lay, t_lev, p_lev, z_lev, t_sfc)
#   NoScatLWRTE, NoScatSWRTE: longwave/shortwave RTE solvers  
#   FluxLW, FluxSW: flux storage (flux_up, flux_dn, flux_net, flux_dn_dir)
#   RRTMGPParameters: physical constants for RRTMGP

using RRTMGP: RRTMGPGridParams
using RRTMGP.RTE: NoScatLWRTE, NoScatSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!
using RRTMGP.Parameters: RRTMGPParameters

using ClimaComms: ClimaComms

using Breeze.CelestialMechanics: cos_solar_zenith_angle

const SingleColumnGrid = RectilinearGrid{<:Any, <:Flat, <:Flat, <:Bounded}
const DateTimeClock = Clock{DateTime}

"""
    default_rrtmgp_parameters(FT=Float64)

Return `RRTMGPParameters` with sensible default values for Earth's atmosphere.
"""
function default_rrtmgp_parameters(::Type{FT}=Float64) where FT
    return RRTMGPParameters(
        grav           = FT(9.80665),        # m/s²
        molmass_dryair = FT(0.028964),       # kg/mol
        molmass_water  = FT(0.018015),       # kg/mol
        gas_constant   = FT(8.3144598),      # J/(mol·K)
        kappa_d        = FT(1004.64/0.028964), # J/(kg·K) / (kg/mol) = J/(mol·K)
        Stefan         = FT(5.670374419e-8), # W m⁻² K⁻⁴
        avogad         = FT(6.02214076e23),  # mol⁻¹
    )
end

"""
    RRTMGPParameters(constants::ThermodynamicConstants)

Construct `RRTMGPParameters` from Breeze's `ThermodynamicConstants`.
"""
function RRTMGPParameters(constants::ThermodynamicConstants{FT}) where FT
    ϰᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass

    return RRTMGPParameters(
        grav           = FT(constants.gravitational_acceleration),
        molmass_dryair = FT(constants.dry_air.molar_mass),
        molmass_water  = FT(constants.vapor.molar_mass),
        gas_constant   = FT(constants.molar_gas_constant),
        kappa_d        = FT(ϰᵈ),
        Stefan         = FT(5.670374419e-8),  # W m⁻² K⁻⁴
        avogad         = FT(6.02214076e23),   # mol⁻¹
    )
end

include("gray_radiative_transfer_model.jl")
include("clear_sky_radiative_transfer_model.jl")

end # module

