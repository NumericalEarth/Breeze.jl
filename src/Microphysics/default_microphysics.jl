abstract type AbstractMicrophysics end

"""
    DefaultMicrophysics()

Placeholder microphysics scheme that leaves all tendencies unchanged. This is
useful for dry runs or when the governing equations already account for moist
processes externally.
"""
struct DefaultMicrophysics <: AbstractMicrophysics end

function update_microphysics_drift_velocity!(::AbstractMicrophysics, state)
    return nothing
end

microphysics_drift_velocity(::AbstractMicrophysics, ::Val) = nothing

microphysics_condensate_names(::AbstractMicrophysics) = ()

microphysics_transition(::AbstractMicrophysics, ::Val, args...) = 0.0

microphysics_auxiliary_fields(::AbstractMicrophysics) = NamedTuple{(), Tuple{}}()

