abstract type AbstractMicrophysics end

struct DefaultMicrophysics <: AbstractMicrophysics end

function (::DefaultMicrophysics)(args...; kwargs...) end

required_microphysics_tracers(::DefaultMicrophysics) = ()

required_microphysics_auxiliary_fields(::DefaultMicrophysics) = ()
