using Breeze.AtmosphereModels: AtmosphereModels
using Reactant: TracedRNumber
using Reactant.Ops: julia_callback

# The reduction is part of the compiled program, so its Boolean cannot be inspected
# while tracing. Check it in a runtime callback instead. Only the Boolean crosses
# into Julia; the moisture fields stay on the device, and the check has no derivative.
function AtmosphereModels.validate_total_moisture(valid::TracedRNumber{Bool})
    julia_callback(AtmosphereModels.validate_total_moisture, (), valid; has_side_effect=true)
    return nothing
end
