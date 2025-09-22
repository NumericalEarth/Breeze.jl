module Microphysics

export temperature,
       saturation_adjustment_residual,
       specific_volume,
       HeightReferenceThermodynamicState

using ..Thermodynamics:
    AnelasticThermodynamicState,
    BoussinesqThermodynamicState,
    ThermodynamicState,
    mixture_heat_capacity,
    mixture_gas_constant,
    exner_function,
    reference_pressure

import ..Thermodynamics: condensate_specific_humidity

const HeightReferenceThermodynamicState = ThermodynamicState

include("nothing_microphysics.jl")
include("saturation_adjustment.jl")

end # module
