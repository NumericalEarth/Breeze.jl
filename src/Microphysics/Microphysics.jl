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
include("interface_sedimentation_velocities.jl")
include("microphysics_1m_rates.jl")
include("microphysics_1m.jl")

end # module
