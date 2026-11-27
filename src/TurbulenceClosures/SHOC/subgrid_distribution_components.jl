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

@inline component_scalars(sf::TurbulentRichardsonStabilityFunction) = TurbulenceClosures.stability_coefficients(sf)
component_scalar_names(::TurbulentRichardsonStabilityFunction, name) = TurbulenceClosures.TURBULENT_RICHARDSON_COEFFICIENTS
@inline function rebuild_component(::TurbulentRichardsonStabilityFunction, scalars, n)
    m = length(TurbulenceClosures.TURBULENT_RICHARDSON_COEFFICIENTS)
    return TurbulentRichardsonStabilityFunction(ntuple(i -> scalars[n + i - 1], Val(m))...), n + m
end
@inline map_scalars(f, sf::TurbulentRichardsonStabilityFunction) = TurbulentRichardsonStabilityFunction(map(f, TurbulenceClosures.stability_coefficients(sf))...)

TurbulenceClosures.convert_eltype(::Type{FT}, component::AbstractSubgridDistributionComponent) where FT =
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
