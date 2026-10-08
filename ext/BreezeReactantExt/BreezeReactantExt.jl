module BreezeReactantExt

using Breeze

include("Timesteppers.jl")

Breeze.Utils.initialize_on_construction!(::ReactantState, x, args...) = nothing
include("initialization.jl")

end # module
