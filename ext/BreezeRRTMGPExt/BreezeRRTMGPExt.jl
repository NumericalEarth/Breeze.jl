module BreezeRRTMGPExt

using Breeze

using Breeze.Thermodynamics: ThermodynamicConstants
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

import RRTMGP.Parameters: RRTMGPParameters

using ClimaComms: ClimaComms

using Breeze.CelestialMechanics: cos_solar_zenith_angle

const SingleColumnGrid = RectilinearGrid{<:Any, <:Flat, <:Flat, <:Bounded}
const DateTimeClock = Clock{DateTime}

"""
    RRTMGPParameters(constants::ThermodynamicConstants)

Construct `RRTMGPParameters` from Breeze's `ThermodynamicConstants`.
"""
function RRTMGPParameters(constants::ThermodynamicConstants{FT};
                          stefan_bolzmann_constant = 5.670374419e-8,  # W m⁻² K⁻⁴
                          avogadro_number = 6.02214076e23) where FT  # mol⁻¹

    ϰᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass

    return RRTMGPParameters(
        grav           = convert(FT, constants.gravitational_acceleration),
        molmass_dryair = convert(FT, constants.dry_air.molar_mass),
        molmass_water  = convert(FT, constants.vapor.molar_mass),
        gas_constant   = convert(FT, constants.molar_gas_constant),
        kappa_d        = convert(FT, ϰᵈ),
        Stefan         = convert(FT, stefan_bolzmann_constant),  # W m⁻² K⁻⁴
        avogad         = convert(FT, avogadro_number),   # mol⁻¹
    )
end

include("gray_radiative_transfer_model.jl")
include("clear_sky_radiative_transfer_model.jl")

end # module

