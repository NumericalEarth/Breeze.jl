module Forcings

export
    geostrophic_forcings,
    SubsidenceForcing,
    GeostrophicForcing

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face
using Oceananigans.Forcings: materialize_forcing, MultipleForcings

using ..AtmosphereModels: AtmosphereModels, materialize_atmosphere_model_forcing, compute_forcing!

include("geostrophic_forcings.jl")
include("subsidence_forcing.jl")

#####
##### Extension of materialize_forcing with context argument
#####

# Fallback: standard forcings don't need context
AtmosphereModels.materialize_atmosphere_model_forcing(forcing, field, name, model_field_names, context) =
    materialize_forcing(forcing, field, name, model_field_names)

# Handle tuples of forcings (multiple forcings on the same field)
# Uses Oceananigans' MultipleForcings to sum contributions
function AtmosphereModels.materialize_atmosphere_model_forcing(forcings::Tuple, field, name, model_field_names, context)
    materialized = Tuple(materialize_atmosphere_model_forcing(f, field, name, model_field_names, context) for f in forcings)
    return MultipleForcings(materialized)
end

# Handle compute_forcing! for MultipleForcings
AtmosphereModels.compute_forcing!(mf::MultipleForcings) = compute_forcing!(mf.forcings)

# Fallback for other forcing types - do nothing
AtmosphereModels.compute_forcing!(forcing) = nothing

end
