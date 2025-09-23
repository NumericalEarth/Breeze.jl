module Microphysics

export temperature,
       specific_volume

using ..Thermodynamics:
    mixture_heat_capacity,
    mixture_gas_constant,
    exner_function,
    reference_pressure

import ..Thermodynamics: condensate_specific_humidity, temperature

include("nothing_microphysics.jl")
#include("saturation_adjustment.jl")

end # module
