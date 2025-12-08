module Thermodynamics

export ThermodynamicConstants, ReferenceState, IdealGas,
       CondensedPhase,
       mixture_gas_constant, mixture_heat_capacity,
       liquid_latent_heat, ice_latent_heat

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

include("thermodynamics_constants.jl")
include("vapor_saturation.jl")
include("reference_states.jl")
include("dynamic_states.jl")

end # module
