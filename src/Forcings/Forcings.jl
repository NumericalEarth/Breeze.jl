module Forcings

export
    geostrophic_forcings,
    SubsidenceForcing,
    UGeostrophicForcing,
    VGeostrophicForcing,
    compute_forcing!

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Average, Field, set!, compute!
using Oceananigans.Grids: Center, Face
using Oceananigans.Forcings: materialize_forcing, MultipleForcings

import Oceananigans.Forcings: materialize_forcing

include("geostrophic_forcings.jl")
include("subsidence_forcings.jl")

#####
##### Extension of materialize_forcing with context argument
#####

# Fallback: standard forcings don't need context
materialize_atmosphere_model_forcing(forcing, field, name, model_field_names, context) =
    materialize_forcing(forcing, field, name, model_field_names)

# Handle tuples of forcings (multiple forcings on the same field)
# Uses Oceananigans' MultipleForcings to sum contributions
function materialize_atmosphere_model_forcing(forcings::Tuple, field, name, model_field_names, context)
    materialized = Tuple(materialize_atmosphere_model_forcing(f, field, name, model_field_names, context) for f in forcings)
    return MultipleForcings(materialized)
end


# Subsidence forcing
function materialize_atmosphere_model_forcing(forcing::SubsidenceForcing, field, name, model_field_names, context)
    grid = field.grid
    
    # Strip the rho prefix to get the specific field name (e.g., :rhou -> :u)
    specific_name = strip_density_prefix(name)
    
    if !haskey(context.specific_fields, specific_name)
        error("SubsidenceForcing is not supported for field $name. " *
              "Available specific fields are: $(keys(context.specific_fields))")
    end
    
    averaged_field = context.specific_fields[specific_name]
    return materialize_subsidence_forcing(forcing, grid, context.reference_density, averaged_field)
end

"""
    strip_density_prefix(name::Symbol)

Convert a density-weighted field name to its specific counterpart by removing
the rho prefix. For example, :rhou -> :u
"""
function strip_density_prefix(name::Symbol)
    s = string(name)
    if startswith(s, "ρ")
        return Symbol(s[nextind(s, 1):end])
    end
    return name
end

# Handle compute_forcing! for MultipleForcings
compute_forcing!(mf::MultipleForcings) = compute_forcing!(mf.forcings)

end
