module Thermodynamics

export ThermodynamicConstants, ReferenceState, IdealGas,
       CondensedPhase,
       mixture_gas_constant, mixture_heat_capacity

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

include("thermodynamics_constants.jl")
include("vapor_saturation.jl")
include("reference_states.jl")
include("dynamic_states.jl")

end # module
