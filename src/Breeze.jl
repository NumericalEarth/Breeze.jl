"""
Finite volume GPU and CPU large eddy simulations (LES) of atmospheric flows.
The abstractions, design, and finite volume engine are based on Oceananigans.
"""
module Breeze

export MoistAirBuoyancy, AtmosphereThermodynamics, ReferenceConstants

include("Thermodynamics/Thermodynamics.jl")
using .Thermodynamics

include("MoistAirBuoyancies.jl")
using .MoistAirBuoyancies

include("AtmosphereModels/AtmosphereModels.jl")
using .AtmosphereModels

end # module Breeze
