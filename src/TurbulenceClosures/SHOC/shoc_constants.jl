#####
##### Components of SHOC, and the constants they share
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

TurbulenceClosures.convert_eltype(::Type{FT}, component::AbstractSHOCComponent) where FT =
    map_scalars(scalar -> convert(FT, scalar), component)

# Every scalar a keyword argument: the constructor of the components below
function component_from_keywords(Component, FT, defaults::NamedTuple, kw)
    names = keys(defaults)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("$(setdiff(keys(kw), names)) are not fields of $(Component); its fields are $(names)."))
    values = merge(defaults, NamedTuple(kw))
    return Component{FT}(map(value -> convert(FT, value), Tuple(values))...)
end

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

"""
$(TYPEDEF)

The physical constants that SHOC is handed by its host (E3SM's `shr_const`), the reference values
of `shoc.F90`, and the bounds and safeguards that several components or the column sweeps
themselves use: the bounds on turbulent kinetic energy, the floors of the surface stress and of
the surface source of turbulent kinetic energy, and the scale of the stratification for the
clipping policy. None is an empirical coefficient of the closure; all are dimensional.

$(TYPEDFIELDS)
"""
struct SHOCConstants{FT} <: AbstractSHOCComponent
    "Gravitational acceleration, m s⁻²"
    ggr :: FT
    "Dry air gas constant, J kg⁻¹ K⁻¹"
    rgas :: FT
    "Water vapor gas constant, J kg⁻¹ K⁻¹"
    rv :: FT
    "Specific heat of dry air, J kg⁻¹ K⁻¹"
    cp :: FT
    "Latent heat of vaporization, J kg⁻¹"
    lcond :: FT
    "`rv / rgas - 1` (E3SM `zvir`)"
    eps :: FT
    "Von Kármán constant"
    vk :: FT
    "Reference temperature, K"
    basetemp :: FT
    "Reference pressure, Pa"
    basepres :: FT
    "Pressure above which the column stability is integrated, Pa"
    troppres :: FT
    "Minimum turbulent kinetic energy, m² s⁻²"
    mintke :: FT
    "Maximum turbulent kinetic energy, m² s⁻²"
    maxtke :: FT
    "Minimum wind speed of the lowest level in the implicit surface stress, m s⁻¹ (`wsmin`)"
    wsmin :: FT
    "Minimum surface drag coefficient of the implicit surface stress, kg m⁻² s⁻¹ (`ksrfmin`)"
    ksrfmin :: FT
    "Minimum friction velocity in the surface source of turbulent kinetic energy, m s⁻¹ (literal `0.01` of `tke_srf_flux_term`)"
    tke_flux_ustar_min :: FT
    "Magnitude of the squared buoyancy frequency that scales the smoothing of its bounds, s⁻²; `HardClipping` ignores it"
    brunt_scale :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SHOCConstants{FT}` with the values of E3SM (`shr_const` and `shoc.F90`).
"""
function SHOCConstants(FT::DataType = Oceananigans.defaults.FloatType;
                       ggr = 9.80616,
                       rgas = 287.042,
                       rv = 461.505,
                       cp = 1004.64,
                       lcond = 2.501e6,
                       eps = rv / rgas - 1,
                       vk = 0.4,
                       basetemp = 300,
                       basepres = 100000,
                       troppres = 80000,
                       mintke = 0.0004,
                       maxtke = 50,
                       wsmin = 1,
                       ksrfmin = 1e-4,
                       tke_flux_ustar_min = 0.01,
                       brunt_scale = 1e-6)

    return SHOCConstants{FT}(ggr, rgas, rv, cp, lcond, eps, vk, basetemp, basepres, troppres, mintke, maxtke,
                             wsmin, ksrfmin, tke_flux_ustar_min, brunt_scale)
end

# Kernels receive the components through `adapt`: on `ReactantState` every traced scalar must be
# replaced by its device counterpart, which needs the fields to be recursed into.
Adapt.@adapt_structure SHOCConstants
