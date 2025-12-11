module BreezeRRTMGPExt

using Breeze
using RRTMGP: RRTMGP

using Dates: DateTime

# Oceananigans imports
using Oceananigans.Architectures: architecture, CPU
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat, Bounded, topology, znodes, znode
using Oceananigans.Fields: ZFaceField

# RRTMGP imports (external types - cannot modify)
#   GrayAtmosphericState: atmospheric state arrays (t_lay, p_lay, t_lev, p_lev, z_lev, t_sfc)
#   NoScatLWRTE, NoScatSWRTE: longwave/shortwave RTE solvers  
#   FluxLW, FluxSW: flux storage (flux_up, flux_dn, flux_net, flux_dn_dir)
#   RRTMGPParameters: physical constants for RRTMGP
using RRTMGP: RRTMGPGridParams
using RRTMGP.AtmosphericStates: GrayAtmosphericState, GrayOpticalThicknessOGorman2008
using RRTMGP.RTE: NoScatLWRTE, NoScatSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!
using RRTMGP.Parameters: RRTMGPParameters

using ClimaComms: ClimaComms

include("solar_zenith_angle.jl")
include("gray_radiation.jl")
include("update_radiation.jl")

end # module

