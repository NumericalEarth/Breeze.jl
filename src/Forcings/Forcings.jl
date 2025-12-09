module Forcings

export
    geostrophic_forcings,
    SubsidenceForcing,
    # Internal types and functions used by AtmosphereModels
    UGeostrophicForcing,
    VGeostrophicForcing,
    materialize_geostrophic_forcing,
    materialize_subsidence_forcing,
    compute_forcing!

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face

include("geostrophic_forcings.jl")
include("subsidence_forcings.jl")

end


