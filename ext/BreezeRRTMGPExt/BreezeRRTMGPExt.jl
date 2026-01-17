module BreezeRRTMGPExt

using Breeze

using Breeze.AtmosphereModels: GrayOptics, ClearSkyOptics, AllSkyOptics, ConstantRadiusParticles
using Breeze.Thermodynamics: ThermodynamicConstants
using RRTMGP: RRTMGP

using Dates: AbstractDateTime, DateTime, Millisecond
using DocStringExtensions: TYPEDSIGNATURES

# Oceananigans imports
using Oceananigans.Architectures: architecture, CPU, GPU
using Oceananigans.Fields: ZFaceField

# RRTMGP imports (external types - cannot modify)
#   GrayAtmosphericState: atmospheric state arrays (t_lay, p_lay, t_lev, p_lev, z_lev, t_sfc)
#   NoScatLWRTE, NoScatSWRTE: longwave/shortwave RTE solvers
#   FluxLW, FluxSW: flux storage (flux_up, flux_dn, flux_net, flux_dn_dir)
#   RRTMGPParameters: physical constants for RRTMGP

using RRTMGP: RRTMGPGridParams
using RRTMGP.Parameters: Parameters, RRTMGPParameters
using RRTMGP.RTE: NoScatLWRTE, NoScatSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!

using ClimaComms: ClimaComms

using Breeze.CelestialMechanics: cos_solar_zenith_angle

const SingleColumnGrid = RectilinearGrid{<:Any, <:Flat, <:Flat, <:Bounded}
const DateTimeClock = Clock{DateTime}

"""
    RRTMGPParameters(constants::ThermodynamicConstants)

Construct `RRTMGPParameters` from Breeze's `ThermodynamicConstants`.
"""
function Parameters.RRTMGPParameters(constants::ThermodynamicConstants{FT};
                                     stefan_bolzmann_constant = 5.670374419e-8,  # W m⁻² K⁻⁴
                                     avogadro_number = 6.02214076e23) where FT  # mol⁻¹

    ϰᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass

    return Parameters.RRTMGPParameters(
        grav           = convert(FT, constants.gravitational_acceleration),
        molmass_dryair = convert(FT, constants.dry_air.molar_mass),
        molmass_water  = convert(FT, constants.vapor.molar_mass),
        gas_constant   = convert(FT, constants.molar_gas_constant),
        kappa_d        = convert(FT, ϰᵈ),
        Stefan         = convert(FT, stefan_bolzmann_constant),  # W m⁻² K⁻⁴
        avogad         = convert(FT, avogadro_number),   # mol⁻¹
    )
end

"""
$(TYPEDSIGNATURES)

Create an RRTMGP-compatible ClimaComms context from an Oceananigans architecture.
"""
function rrtmgp_context(arch::CPU)
    device = Threads.nthreads() > 1 ? ClimaComms.CPUMultiThreaded() : ClimaComms.CPUSingleThreaded()
    return ClimaComms.context(device)
end

function rrtmgp_context(arch::GPU)
    return ClimaComms.context(ClimaComms.CUDADevice())
end

compute_datetime(dt::AbstractDateTime, epoch) = dt
compute_datetime(t::Number, epoch::AbstractDateTime) = epoch + Millisecond(round(Int, 1000t))
# When epoch is nothing and time is numeric, we can't compute datetime (used for fixed zenith angle)
compute_datetime(t::Number, epoch::Nothing) = nothing

include("gray_radiative_transfer_model.jl")
include("rrtmgp_shared_utilities.jl")
include("clear_sky_radiative_transfer_model.jl")
include("all_sky_radiative_transfer_model.jl")

end # module
