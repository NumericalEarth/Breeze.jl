module Forcings

export
    geostrophic_forcings,
    SubsidenceForcing,
    GeostrophicForcing,
    SpecificForcing

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face
using Oceananigans.Forcings: materialize_forcing, MultipleForcings

using ..AtmosphereModels: AtmosphereModels, materialize_atmosphere_model_forcing,
                          compute_forcing!, is_density_tendency_forcing, wrap_specific_forcing

include("geostrophic_forcings.jl")
include("subsidence_forcing.jl")
include("specific_forcing.jl")

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
function AtmosphereModels.compute_forcing!(mf::MultipleForcings)
    for forcing in mf.forcings
        compute_forcing!(forcing)
    end
    return nothing
end

#####
##### Density-tendency forcing trait: identifies Breeze forcings that already produce F_{ρϕ}
##### so the atmosphere-model dispatch can reject misuse under specific keys.
#####

AtmosphereModels.is_density_tendency_forcing(::SubsidenceForcing) = true
AtmosphereModels.is_density_tendency_forcing(::GeostrophicForcing) = true

#####
##### Specific-key wrapping: build a SpecificForcing for a user value supplied under a
##### specific name (`θ`, `u`, ...). Tuples recurse; density-tendency forcings error.
#####

function AtmosphereModels.wrap_specific_forcing(value, density_name)
    if is_density_tendency_forcing(value)
        msg = string("Forcing of type ", nameof(typeof(value)),
                     " produces a density-weighted tendency F_{ρϕ}; ",
                     "supply it under the density-weighted key `", density_name,
                     "` rather than its specific counterpart. ",
                     "Auto-wrapping it in SpecificForcing would multiply by ρ a second time.")
        throw(ArgumentError(msg))
    end
    return SpecificForcing(value)
end

AtmosphereModels.wrap_specific_forcing(values::Tuple, density_name) =
    map(v -> wrap_specific_forcing(v, density_name), values)

end
