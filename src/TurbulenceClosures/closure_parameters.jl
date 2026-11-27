abstract type AbstractSubgridDistributionComponent end

component_constructor(::Type{T}) where T = Base.typename(T).wrapper

@inline map_scalars(f, scalar::Number) = f(scalar)

@inline function map_scalars(f, component::T) where T
    fields = map(name -> map_scalars(f, getfield(component, name)), fieldnames(T))
    return component_constructor(T)(fields...)
end

@inline component_scalars(scalar::Number) = tuple(scalar)

@inline function component_scalars(component::T) where T
    scalars = ()
    for name in fieldnames(T)
        scalars = (scalars..., component_scalars(getfield(component, name))...)
    end
    return scalars
end

component_scalar_names(component) = component_scalar_names(component, nothing)
component_scalar_names(::Number, name) = tuple(name)

function component_scalar_names(component::T, name) where T
    names = ()
    for field in fieldnames(T)
        names = (names..., component_scalar_names(getfield(component, field), field)...)
    end
    return names
end

@inline rebuild_component(::Number, scalars, n) = scalars[n], n + 1

@inline function rebuild_component(component::T, scalars, n) where T
    fields = ()
    for name in fieldnames(T)
        field, n = rebuild_component(getfield(component, name), scalars, n)
        fields = (fields..., field)
    end
    return component_constructor(T)(fields...), n
end

@inline map_scalars(f, choice::Val) = choice
@inline component_scalars(::Val) = ()
component_scalar_names(::Val, name) = ()
@inline rebuild_component(choice::Val, scalars, n) = choice, n

@inline component_scalars(sf::TurbulentRichardsonStabilityFunction) = stability_coefficients(sf)
component_scalar_names(::TurbulentRichardsonStabilityFunction, name) = TURBULENT_RICHARDSON_COEFFICIENTS
@inline function rebuild_component(::TurbulentRichardsonStabilityFunction, scalars, n)
    m = length(TURBULENT_RICHARDSON_COEFFICIENTS)
    return TurbulentRichardsonStabilityFunction(ntuple(i -> scalars[n + i - 1], Val(m))...), n + m
end
@inline map_scalars(f, sf::TurbulentRichardsonStabilityFunction) = TurbulentRichardsonStabilityFunction(map(f, stability_coefficients(sf))...)

convert_eltype(::Type{FT}, component::AbstractSubgridDistributionComponent) where FT =
    map_scalars(scalar -> convert(FT, scalar), component)

function keyword_scalars(Component, FT, defaults::NamedTuple, kw)
    names = keys(defaults)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("$(setdiff(keys(kw), names)) are not fields of $(Component); its fields are $(names)."))
    values = merge(defaults, NamedTuple(kw))
    return map(value -> convert(FT, value), Tuple(values))
end

component_from_keywords(Component, FT, defaults::NamedTuple, kw) = Component{FT}(keyword_scalars(Component, FT, defaults, kw)...)

component_from_keywords(Component, FT, defaults::NamedTuple, kw, structural...) =
    Component(keyword_scalars(Component, FT, defaults, kw)..., structural...)

Base.summary(component::AbstractSubgridDistributionComponent) = string(nameof(typeof(component)), type_parameter_summary(component))

type_parameter_summary(component) = isempty(component_scalars(component)) ? "" :
                                    string("{", typeof(first(component_scalars(component))), "}")

function Base.show(io::IO, component::AbstractSubgridDistributionComponent)
    print(io, summary(component))
    names = component_scalar_names(component)
    scalars = component_scalars(component)
    isempty(names) && return print(io, "()")
    print(io, ":")
    for (name, scalar) in zip(names, scalars)
        print(io, '\n', "    ", name, " = ", prettysummary(scalar))
    end
end

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
