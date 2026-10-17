using Breeze.AtmosphereModels: AtmosphereModels
using Reactant: TracedRNumber
using Reactant.Ops: julia_callback

# Runtime callbacks are closure `cfunction`s, which need LLVM trampolines. Julia provides
# them on x86 but not, for example, on Apple silicon.
const runtime_callbacks_supported = Sys.ARCH in (:x86_64, :i686)

# The reduction is part of the compiled program, so its Boolean cannot be inspected
# while tracing. Check it in a runtime callback instead. Only the Boolean crosses
# into Julia; the moisture fields stay on the device, and the check has no derivative.
# Where runtime callbacks are unavailable, skip the check rather than fail to compile.
function AtmosphereModels.validate_total_moisture(valid::TracedRNumber{Bool})
    if runtime_callbacks_supported
        julia_callback(AtmosphereModels.validate_total_moisture, (), valid; has_side_effect=true)
    else
        @warn "Skipping the traced total moisture check: runtime callbacks are unavailable on $(Sys.ARCH)." maxlog=1
    end
    return nothing
end
