module Microphysics

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Advection: div_Uc, Centered
using Oceananigans.Architectures: device, architecture
using Oceananigans.Fields: ZeroField

import Oceananigans.Fields: CenterField

#####
##### Generic fallbacks for microphysics
#####

@inline microphysics_rhs(i, j, k, grid, ::Nothing, val_tracer_name, clock, fields) = zero(grid)

"""
    update_tendencies!(mp, model)

Update prognostic tendencies after they have been computed.
"""
update_tendencies!(mp, model) = nothing

"""
    update_microphysics_state!(mp, model)

Update microphysical state variables. Called at the end of update_state!.
"""
update_microphsyics_state!(mp, model) = nothing

@inline microphysics_drift_velocity(mp, val_tracer_name) = nothing
@inline microphysics_auxiliary_fields(mp) = NamedTuple()

"""
    AbstractMicrophysics

Abstract type for microphysical models. To define a microphysical relationship
the following functions must have methods defined where `MicrophysicsModel`
is a subtype of `AbstractMicrophysics`:

  - `(mp::MicrophysicsModel)(i, j, k, grid, ::Val{:tracer_name}, clock, fields)` which
     returns the microphysical reaction for for each tracer.

  - `required_microphysics_tracers(::MicrophysicsModel)` which returns a tuple of
     required `tracer_names`.

  - `required_microphysics_auxiliary_fields(::MicrophysicsModel)` which returns
     a tuple of required auxiliary fields.

  - `microphysics_auxiliary_fields(mp::MicrophysicsModel)` which returns a `NamedTuple`
     of the models auxiliary fields.

  - `microphysics_drift_velocity(mp::MicrophysicsModel, ::Val{:tracer_name})` which
     returns a velocity fields (i.e. a `NamedTuple` of fields with keys `u`, `v` & `w`)
     for each tracer.

  - `update_microphysics_state!(mp::MicrophysicsModel, model)` (optional) to update the
      model state.
"""
abstract type AbstractMicrophysics end

# Returns the forcing for discrete form models
@inline microphysics_transition(i, j, k, grid, mp, val_tracer_name, clock, fields) =
    mp(i, j, k, grid, val_tracer_name, clock, fields)

@inline microphysics_transition(i, j, k, grid, ::Nothing, val_tracer_name, clock, fields) = zero(grid)

# Required for when a model is defined but not for all tracers
@inline (mp::AbstractMicrophysics)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

"""
    AbstractContinuousFormMicrophysics

Abstract type for microphysical models with continuous form microphysical reaction
functions. To define a microphysical relationship the following functions must have methods
defined where `MicrophysicsModel` is a subtype of `AbstractContinuousFormMicrophysics`:

  - `(mp::MicrophysicsModel)(::Val{:tracer_name}, x, y, z, t, tracers..., auxiliary_fields...)`
     which returns the microphysical reaction for for each tracer.

  - `required_microphysics_tracers(::MicrophysicsModel)` which returns a tuple of
     required tracer names.

  - `required_microphysics_auxiliary_fields(::MicrophysicsModel)` which returns
     a tuple of required auxiliary fields.

  - `microphysics_auxiliary_fields(mp::MicrophysicsModel)` which returns a `NamedTuple`
     of the models auxiliary fields

  - `microphysics_drift_velocity(mp::MicrophysicsModel, ::Val{:tracer_name})` which
     returns "additional" velocity fields modeling, for example, sinking particles

  - `update_microphysics_state!(mp::MicrophysicsModel, model)` (optional) to update the
     model state
"""
abstract type AbstractContinuousFormMicrophysics <: AbstractMicrophysics end

@inline extract_microphysics_fields(i, j, k, grid, fields, names::NTuple{1}) =
    @inbounds tuple(fields[names[1]][i, j, k])

@inline extract_microphysics_fields(i, j, k, grid, fields, names::NTuple{2}) =
    @inbounds (fields[names[1]][i, j, k],
               fields[names[2]][i, j, k])

@inline extract_microphysics_fields(i, j, k, grid, fields, names::NTuple{N}) where N =
    @inbounds ntuple(n -> fields[names[n]][i, j, k], Val(N))

"""Return the microphysical forcing for `val_tracer_name` for continuous form when model is called."""
@inline function microphysics_transition(i, j, k, grid, mp::AbstractContinuousFormMicrophysics,
                                           val_tracer_name, clock, fields)

    names_to_extract = tuple(required_microphysics_tracers(mp)...,
                             required_microphysics_auxiliary_fields(mp)...)

    fields_ijk = extract_microphysics_fields(i, j, k, grid, fields, names_to_extract)

    x = xnode(i, j, k, grid, Center(), Center(), Center())
    y = ynode(i, j, k, grid, Center(), Center(), Center())
    z = znode(i, j, k, grid, Center(), Center(), Center())

    return mp(val_tracer_name, x, y, z, clock.time, fields_ijk...)
end

@inline (mp::AbstractContinuousFormMicrophysics)(val_tracer_name, x, y, z, t, fields...) = zero(t)

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

add_microphysics_tracer(tracers::Tuple, name, grid) = tuple(tracers..., name)
add_microphysics_tracer(tracers::NamedTuple, name, grid) = merge(tracers, (; name => CenterField(grid)))

@inline function has_microphysics_tracers(fields, required_fields, grid)
    user_specified_tracers = [name in tracernames(fields) for name in required_fields]

    if !all(user_specified_tracers) && any(user_specified_tracers)
        throw(ArgumentError("The microphysical model you have selected requires $required_fields.\n" *
                            "You have specified some but not all of these as tracers so may be attempting\n" *
                            "to use them for a different purpose. Please either specify all of the required\n" *
                            "fields, or none and allow them to be automatically added."))

    elseif !any(user_specified_tracers)
        for field_name in required_fields
            fields = add_microphysics_tracer(fields, field_name, grid)
        end
    else
        fields = fields
    end

    return fields
end

"""
    validate_microphysics(tracers, auxiliary_fields, mp, grid, clock)

Ensure that `tracers` contains microphysical tracers and `auxiliary_fields`
contains microphysical auxiliary fields.
"""
@inline function validate_microphysics(tracers, auxiliary_fields, mp, grid, clock)
    req_tracers = required_microphysics_tracers(mp)
    tracers = has_microphysics_tracers(tracers, req_tracers, grid)
    req_auxiliary_fields = required_microphysics_auxiliary_fields(mp)

    all(field ∈ tracernames(auxiliary_fields) for field in req_auxiliary_fields) ||
        error("$(req_auxiliary_fields) must be among the list of auxiliary fields to use $(typeof(mp).name.wrapper)")

    # Return tracers and aux fields so that users may overload and
    # define their own special auxiliary fields
    return tracers, auxiliary_fields
end

const AbstractMPOrNothing = Union{Nothing, AbstractMicrophysics}
required_microphysics_tracers(::AbstractMPOrNothing) = ()
required_microphysics_auxiliary_fields(::AbstractMPOrNothing) = ()

end # module
