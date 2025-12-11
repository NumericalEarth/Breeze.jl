module BreezeRRTMGPExt

using Breeze
using RRTMGP: RRTMGP
using Dates: Dates, DateTime

# Oceananigans imports
using Oceananigans.Architectures: architecture, CPU
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat, Bounded, topology, znodes
using Oceananigans.Fields: ZFaceField

# RRTMGP imports
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

