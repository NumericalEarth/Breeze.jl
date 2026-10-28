using Breeze.AtmosphereModels: AtmosphereModels, specific_prognostic_moisture
using Reactant: TracedRNumber

# Inside a compiled program the check's Boolean is a device value: it can neither be inspected while
# tracing nor raise at run time. The moisture of an invalid state becomes NaN instead, through a select,
# which keeps the program differentiable and leaves the NaN in the run where it shows.
AtmosphereModels.validate_total_moisture(::TracedRNumber{Bool}) = nothing

function AtmosphereModels.invalidate_moisture!(model, valid::TracedRNumber{Bool})
    FT = eltype(model.grid)
    for field in (model.moisture_density, specific_prognostic_moisture(model))
        parent(field) .= ifelse.(valid, parent(field), convert(FT, NaN))
    end
    return nothing
end
