using Reactant: @jit
using Oceananigans: ReactantState
using Oceananigans.Grids: LatitudeLongitudeGrid
using Oceananigans.Fields: set!
using Breeze.AtmosphereModels: AtmosphereModels

# Workaround: on LatitudeLongitudeGrid + ReactantState, eager KA kernel compilation
# rejects getindex on the grid's ConcretePJRTArray metric vectors.
# For CCC × FCC (velocity set!), we @jit to trace through MLIR where all
# arrays are TracedRArray and interpolation operators work correctly.
# See https://github.com/NumericalEarth/Breeze.jl/issues/543

const ReactantLatLonModel = AtmosphereModel{<:Any, <:Any, <:ReactantState, <:Any, <:LatitudeLongitudeGrid}

function AtmosphereModels.set_product!(dest::Field{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}, a, b)
    @jit set!(dest, a * b)
end
