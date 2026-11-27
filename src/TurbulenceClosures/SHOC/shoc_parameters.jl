function closure_components end
is_calibratable(name::Symbol) = startswith(string(name), "ℂ") || name === :Ri⁻ || name === :Ri⁺

function closure_scalar_fields(components)
    names = ()
    scalars = ()
    for component in components
        names = (names..., component_scalar_names(component)...)
        scalars = (scalars..., component_scalars(component)...)
    end
    return names, scalars
end

function closure_scalars(closure)
    names, scalars = closure_scalar_fields(closure_components(closure))
    return NamedTuple{names}(scalars)
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the empirical coefficients of every component of `closure` in one flat `NamedTuple`,
in the order of the components. Which coefficients there are depends on the components, and
every combination of components has them collected here.
"""
function closure_parameters(closure)
    scalars = closure_scalars(closure)
    names = filter(is_calibratable, keys(scalars))
    return NamedTuple{names}(map(name -> scalars[name], names))
end

function with_parameters end
