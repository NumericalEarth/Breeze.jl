#####
##### Components of SHOC
#####
##### `SimplifiedHigherOrderClosure` is assembled from components: concretely typed, immutable
##### structs whose scalar fields all have one type, and whose methods are scalar functions that
##### the column sweeps call level by level. A component holds the empirical coefficients (written
##### ℂ) of the relation it implements, and the bounds of that relation.
#####

"""
Supertype of the components of [`SimplifiedHigherOrderClosure`](@ref) defined with it.
"""
abstract type AbstractSHOCComponent end

# The constructor of the type of a component, without its type parameters
component_constructor(::Type{T}) where T = Base.typename(T).wrapper

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `component` with `f` applied to each of its scalars: its `Number` fields, and those of the
components it holds. Fields without scalars are kept.
"""
@inline map_scalars(f, scalar::Number) = f(scalar)

@inline function map_scalars(f, component::T) where T
    fields = map(name -> map_scalars(f, getfield(component, name)), fieldnames(T))
    return component_constructor(T)(fields...)
end

# The scalars of a component, as a flat tuple in field order
@inline component_scalars(scalar::Number) = tuple(scalar)

@inline function component_scalars(component::T) where T
    scalars = ()
    for name in fieldnames(T)
        scalars = (scalars..., component_scalars(getfield(component, name))...)
    end
    return scalars
end

# The names of the scalars of a component, in the order of `component_scalars`
component_scalar_names(component) = component_scalar_names(component, nothing)
component_scalar_names(::Number, name) = tuple(name)

function component_scalar_names(component::T, name) where T
    names = ()
    for field in fieldnames(T)
        names = (names..., component_scalar_names(getfield(component, field), field)...)
    end
    return names
end

# The component rebuilt with its scalars taken, in order, from `scalars` starting at index `n`.
# Returns the component and the index of the next unused scalar.
@inline rebuild_component(::Number, scalars, n) = scalars[n], n + 1

@inline function rebuild_component(component::T, scalars, n) where T
    fields = ()
    for name in fieldnames(T)
        field, n = rebuild_component(getfield(component, name), scalars, n)
        fields = (fields..., field)
    end
    return component_constructor(T)(fields...), n
end

# The blending rule of a mixing length is structural, not an SHOC coefficient
const MixingLengthBlending = Union{HardMinimum, HarmonicBlend}
@inline component_scalars(::MixingLengthBlending) = ()
component_scalar_names(::MixingLengthBlending, name) = ()
@inline rebuild_component(blending::MixingLengthBlending, scalars, n) = blending, n

# A `Val` field is a structural choice (such as `ScalarVarianceTransport`'s `component_third_moment`), not a coefficient
@inline map_scalars(f, choice::Val) = choice
@inline component_scalars(::Val) = ()
component_scalar_names(::Val, name) = ()
@inline rebuild_component(choice::Val, scalars, n) = choice, n

# The coefficients of the stability functions of the turbulent Richardson number are its fourteen
# values; the cached logarithms are rebuilt from them
const TRSF = TurbulentRichardsonStabilityFunction
@inline component_scalars(sf::TRSF) = TurbulenceClosures.stability_coefficients(sf)
component_scalar_names(::TRSF, name) = TurbulenceClosures.TURBULENT_RICHARDSON_COEFFICIENTS
@inline function rebuild_component(::TRSF, scalars, n)
    m = length(TurbulenceClosures.TURBULENT_RICHARDSON_COEFFICIENTS)
    return TRSF(ntuple(i -> scalars[n + i - 1], Val(m))...), n + m
end
@inline map_scalars(f, sf::TRSF) = TRSF(map(f, TurbulenceClosures.stability_coefficients(sf))...)

TurbulenceClosures.convert_eltype(::Type{FT}, component::AbstractSHOCComponent) where FT =
    map_scalars(scalar -> convert(FT, scalar), component)

# Every scalar a keyword argument: the constructor of the components below. The scalars come first; the
# `structural` fields (a clipping policy) follow them.
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

Base.summary(component::AbstractSHOCComponent) = string(nameof(typeof(component)), type_parameter_summary(component))

type_parameter_summary(component) = isempty(component_scalars(component)) ? "" :
                                    string("{", typeof(first(component_scalars(component))), "}")

function Base.show(io::IO, component::AbstractSHOCComponent)
    print(io, summary(component))
    names = component_scalar_names(component)
    scalars = component_scalars(component)
    isempty(names) && return print(io, "()")
    print(io, ":")
    for (name, scalar) in zip(names, scalars)
        print(io, '\n', "    ", name, " = ", prettysummary(scalar))
    end
end
