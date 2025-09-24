module Microphysics

import CloudMicrophysics

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Fields: ZeroField

import Oceananigans.Fields: CenterField

const CM1 = CloudMicrophysics.Microphysics1M
const CMNe = CloudMicrophysics.MicrophysicsNonEq
const CMP = CloudMicrophysics.Parameters

export AbstractMicrophysics,
       
include("interface.jl")
include("process_rates.jl")
include("sedimentation_velocities.jl")
include("microphysics_1m.jl")

end # module
