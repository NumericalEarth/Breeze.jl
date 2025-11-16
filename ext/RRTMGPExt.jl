module RRTMGPExt

import Breeze
import RRTMGP

# Load the RadiativeTransfer module when RRTMGP is available
include(joinpath(@__DIR__, "..", "src", "RadiativeTransfer", "RadiativeTransfer.jl"))

using .RadiativeTransfer

# Re-export RadiativeTransferModel in Breeze module
const RadiativeTransferModel = RadiativeTransfer.RadiativeTransferModel

# Make functions available to AtmosphereModels module by adding them to the module
# This allows the functions to be called from dynamics_kernel_functions.jl
Breeze.AtmosphereModels.eval(quote
    const _radiative_heating_rate = $(RadiativeTransfer._radiative_heating_rate)
    const _update_radiative_fluxes! = $(RadiativeTransfer._update_radiative_fluxes!)
end)

# Export RadiativeTransferModel from Breeze
Breeze.eval(quote
    export RadiativeTransferModel
end)

end # module

