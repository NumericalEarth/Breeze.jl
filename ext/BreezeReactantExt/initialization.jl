using Breeze.AtmosphereModels: AtmosphereModels, specific_prognostic_moisture
using Reactant: TracedRNumber, AbstractConcreteArray

# Outside a compiled program, reducing a view of a concrete Reactant array indexes it element by element, which
# is disallowed; reduce a host copy instead. Inside a compiled program the data are traced and reduce in place.
AtmosphereModels.data_minimum(data::SubArray{<:Any, <:Any, <:AbstractConcreteArray}) =
    minimum(view(Array(parent(data)), parentindices(data)...))

# Inside a compiled program the check's Boolean is a device value: it can neither be
# inspected while tracing nor raise at execution time. Mark an invalid state by setting
# its moisture to NaN through a select, which keeps the program differentiable.
function AtmosphereModels.reject_invalid_moisture!(model, valid::TracedRNumber{Bool})
    FT = eltype(model.grid)
    for field in (model.moisture_density, specific_prognostic_moisture(model))
        parent(field) .= ifelse.(valid, parent(field), convert(FT, NaN))
    end
    return nothing
end
