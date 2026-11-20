module BreezeReactantExt

using Breeze
using Reactant: Reactant

const ConcreteReactantArray = Union{Reactant.AnyConcretePJRTArray, Reactant.AnyConcreteIFRTArray}

include("Timesteppers.jl")
include("initialization.jl")
include("Solvers.jl")
include("Fields.jl")

end # module
