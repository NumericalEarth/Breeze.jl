module BreezeRRTMGPExt

using Breeze
using RRTMGP
using Dates

# Oceananigans imports
using Oceananigans
using Oceananigans.Grids: Flat, Bounded, topology
using Oceananigans.Fields: ZFaceField, CenterField, interior

# RRTMGP imports
using RRTMGP: RRTMGPGridParams
using RRTMGP.AtmosphericStates: GrayAtmosphericState, GrayOpticalThicknessOGorman2008
using RRTMGP.Fluxes: FluxLW, FluxSW
using RRTMGP.RTE: NoScatLWRTE, NoScatSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!
using RRTMGP.Parameters: RRTMGPParameters

using ClimaComms

import Breeze: GrayRadiation
import Breeze.AtmosphereModels: update_radiation!

include("solar_zenith_angle.jl")
include("gray_radiation.jl")
include("update_radiation.jl")

end # module

