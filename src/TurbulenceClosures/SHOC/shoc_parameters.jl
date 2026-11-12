#####
##### The coefficients of a `SimplifiedHigherOrderClosure`, as one flat view
#####
##### The empirical coefficients (written ℂ, as elsewhere in Breeze) are fields of the component
##### that uses them. `closure_parameters` collects them in one `NamedTuple`, and `with_parameters`
##### rebuilds a closure with some of them replaced, whichever components they belong to. The
##### names are unique across the components of a closure (the constructor checks).
#####

"""
The fields of [`SimplifiedHigherOrderClosure`](@ref) that hold a component, in the order of
[`closure_parameters`](@ref).
"""
const COMPONENT_SLOTS = (:mixing_length, :turbulence_time_scale, :tke_update, :buoyancy_source, :second_moments,
                         :third_moment, :stratification_number_bound, :skewness, :assumed_pdf, :pdf_saturation, :boundary_layer_depth, :clipping, :constants)

closure_components(closure) = map(slot -> getfield(closure, slot), COMPONENT_SLOTS)

# Empirical coefficients are written ℂ; the transition scales of the Richardson number of
# `RationalStabilityFunction` are coefficients too.
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

function check_unique_scalar_names(components)
    names, _ = closure_scalar_fields(components)
    repeated = unique(filter(name -> count(==(name), names) > 1, collect(names)))
    isempty(repeated) ||
        throw(ArgumentError("The components of a SimplifiedHigherOrderClosure must not share a field name, " *
                            "but $(Tuple(repeated)) occur in more than one of " *
                            join(map(summary, components), ", ") * "."))
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return every scalar of every component of `closure` in one `NamedTuple`: the empirical
coefficients, and the constants, tolerances and bounds.
"""
function closure_scalars(closure)
    names, scalars = closure_scalar_fields(closure_components(closure))
    return NamedTuple{names}(scalars)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the empirical coefficients of every component of `closure` in one flat `NamedTuple`,
in the order of the components. Which coefficients there are depends on the components, and
every combination of components has them collected here.

The coefficients of a component that another closure also uses, such as
[`GradientLimitedMixingLength`](@ref) or the [`RationalStabilityFunction`](@ref) of
[`RichardsonStabilityFunctions`](@ref), are the copies this closure holds: coefficients of
SHOC like the others, to be calibrated within it and not tied to the values of
[`TKEBasedTurbulenceClosure`](@ref).

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure()
keys(closure_parameters(closure))

# output
(:ℂˡ₁, :ℂˡ₂, :ℂˡ₃, :ℂˡ₄, :ℂˡ₅, :ℂᵘ, :ℂᶜ, :ℂᵘₛ, :ℂᶜₛ, :ℂᵝ₁, :ℂᵝ₂, :ℂᵝ₃, :ℂᵝ₄, :ℂᴰ₁, :ℂᴰ₂, :ℂᴰ₃, :ℂᴰ₄, :ℂᴰ₅, :ℂᴾ, :ℂᵛ₁, :ℂᵛ₂, :ℂᵛ₃, :ℂʷ₁, :ℂᶠ, :ℂᶠ₂, :ℂᶠ₃, :ℂᶠ₄, :ℂᶠ₅, :ℂʷ₂, :ℂʷ₃, :ℂʷ₄, :ℂʷ₅, :ℂʷ₆, :ℂʷ₇, :ℂʷ₈, :ℂʷ₉, :ℂʷ₁₀, :ℂʷ₁₁, :ℂᵍ₁, :ℂᵍ₂, :ℂᵍ₃, :ℂᵍ₄, :ℂᴴ₁, :ℂᴴ₂, :ℂᴴ₃, :ℂᴴ₄, :ℂᴴ₅, :ℂᴴ₆, :ℂᴴ₇)
```

With the mixing length and the stability functions that `TKEBasedTurbulenceClosure` also uses:

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(mixing_length = GradientLimitedMixingLength(),
                                       turbulence_time_scale = RichardsonStabilityFunctions(RationalStabilityFunction()))
keys(closure_parameters(closure))

# output
(:ℂˢ, :ℂⁱ, :ℂᵘ⁻, :ℂᵘ⁰, :ℂᵘ⁺, :ℂᶜ⁻, :ℂᶜ⁰, :ℂᶜ⁺, :ℂᵉ⁻, :ℂᵉ⁰, :ℂᵉ⁺, :ℂᴰ⁻, :ℂᴰ⁰, :ℂᴰ⁺, :Ri⁻, :Ri⁺, :ℂᵗ, :ℂᴾ, :ℂᵛ₁, :ℂᵛ₂, :ℂᵛ₃, :ℂʷ₁, :ℂᶠ, :ℂᶠ₂, :ℂᶠ₃, :ℂᶠ₄, :ℂᶠ₅, :ℂʷ₂, :ℂʷ₃, :ℂʷ₄, :ℂʷ₅, :ℂʷ₆, :ℂʷ₇, :ℂʷ₈, :ℂʷ₉, :ℂʷ₁₀, :ℂʷ₁₁, :ℂᵍ₁, :ℂᵍ₂, :ℂᵍ₃, :ℂᵍ₄, :ℂᴴ₁, :ℂᴴ₂, :ℂᴴ₃, :ℂᴴ₄, :ℂᴴ₅, :ℂᴴ₆, :ℂᴴ₇)
```
"""
function closure_parameters(closure)
    scalars = closure_scalars(closure)
    names = filter(is_calibratable, keys(scalars))
    return NamedTuple{names}(map(name -> scalars[name], names))
end

"""
The empirical coefficients of the components of SHOC that have a dimension: the eddy turnover
time scale `ℂˡ₅` (s) of [`BlendedMixingLength`](@ref), the depth `ℂᶠ₅` (m) of the convective
velocity scale of [`DowngradientSecondMoments`](@ref), and the least depths `ℂᴴ₆` (s, over the
friction velocity) and `ℂᴴ₇` (m) of [`BulkRichardsonBoundaryLayerDepth`](@ref).
"""
const DIMENSIONAL_COEFFICIENTS = (:ℂˡ₅, :ℂᶠ₅, :ℂᴴ₆, :ℂᴴ₇)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the names of the coefficients of `closure` that a calibration may train: those of
[`closure_parameters`](@ref) except the dimensional ones (`DIMENSIONAL_COEFFICIENTS`) and the
coefficients `ℂᴴ` of the boundary layer depth, which is a diagnostic.

Not every name returned can be determined independently of the others: the five `ℂᴰ` of
[`ColumnStabilityDamping`](@ref) enter through one product, `ℂᶠ` of
[`DowngradientSecondMoments`](@ref) only multiplies `ℂᶠ₂` and `ℂᶠ₃`, and `ℂˡ₂` of
[`BlendedMixingLength`](@ref) only divides by `ℂˡ₁`.
"""
trainable_parameter_names(closure) =
    filter(name -> name ∉ DIMENSIONAL_COEFFICIENTS && !startswith(string(name), "ℂᴴ"), keys(closure_parameters(closure)))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a copy of `closure` in which the scalars named by the keyword arguments are replaced,
whichever components hold them. Any name of [`closure_scalars`](@ref) may be given. The
replacements are promoted with the retained scalars, so that passing scalars of another `Real`
type (for example traced scalars during automatic differentiation) returns a closure whose
components all have the promoted scalar type.

```jldoctest
using Breeze

closure = with_parameters(SimplifiedHigherOrderClosure(); ℂᶜ=0.2, ℂʷ₂=6)
closure.turbulence_time_scale.ℂᶜ, closure.third_moment.ℂʷ₂

# output
(0.2, 6.0)
```
"""
function with_parameters(closure::SimplifiedHigherOrderClosure; kw...)
    scalars = closure_scalars(closure)
    names = keys(scalars)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("with_parameters: $(setdiff(keys(kw), names)) are not scalars of the components of " *
                            "$(summary(closure)); they are $(names)."))

    replaced = map(name -> haskey(kw, name) ? kw[name] : scalars[name], names)
    promoted = promote(replaced...)

    n = 1
    components = ()
    for slot in COMPONENT_SLOTS
        component, n = rebuild_component(getfield(closure, slot), promoted, n)
        components = (components..., component)
    end

    return SimplifiedHigherOrderClosure(components...,
                                        closure.surface_temperature_flux,
                                        closure.surface_moisture_flux,
                                        closure.surface_x_momentum_flux,
                                        closure.surface_y_momentum_flux,
                                        closure.friction_velocity,
                                        closure.host_grid_spacing)
end

"""
The empirical coefficients of the default components, each with the name it has in E3SM's
`shoc.F90` (`nothing` where `shoc.F90` has only a literal).
"""
const SHOC_COEFFICIENT_NAMES = (ℂᵘ = :Ckm, ℂᶜ = :Ckh, ℂᵘₛ = :Ckm_s, ℂᶜₛ = :Ckh_s,
                                ℂˡ₁ = :length_fac, ℂˡ₂ = nothing, ℂˡ₃ = nothing, ℂˡ₄ = nothing, ℂˡ₅ = :tscale,
                                ℂᵛ₁ = :thl2tune, ℂᵛ₂ = :qw2tune, ℂᵛ₃ = :qwthl2tune,
                                ℂʷ₁ = :w2tune, ℂʷ₂ = :c_diag_3rd_mom,
                                ℂᵝ₁ = :lambda_low, ℂᵝ₂ = :lambda_high, ℂᵝ₃ = :lambda_slope, ℂᵝ₄ = :lambda_thresh,
                                ℂᴰ₁ = :Ck, ℂᴰ₂ = :Cs, ℂᴾ = :Ck_sh, ℂᶠ = :a_const,
                                ℂᵍ₁ = :w2_1, ℂᵍ₂ = nothing, ℂᵍ₃ = nothing, ℂᵍ₄ = nothing,
                                ℂᴴ₁ = :ricr, ℂᴴ₂ = :fac, ℂᴴ₃ = :fak, ℂᴴ₄ = :betam,
                                ℂᴰ₃ = nothing, ℂᴰ₄ = nothing, ℂᴰ₅ = nothing,
                                ℂᶠ₂ = nothing, ℂᶠ₃ = nothing, ℂᶠ₄ = nothing, ℂᶠ₅ = :z_const,
                                ℂʷ₃ = nothing, ℂʷ₄ = nothing, ℂʷ₅ = nothing, ℂʷ₆ = nothing, ℂʷ₇ = nothing,
                                ℂʷ₈ = nothing, ℂʷ₉ = nothing, ℂʷ₁₀ = nothing, ℂʷ₁₁ = nothing,
                                ℂᴴ₅ = :sffrac, ℂᴴ₆ = nothing, ℂᴴ₇ = nothing)

"""
The run-time tunables of `shoc_init` in E3SM's `shoc.F90`, by their coefficient name in the
default components.
"""
const SHOC_INIT_TUNABLE_NAMES = (ℂᵛ₁ = :thl2tune, ℂᵛ₂ = :qw2tune, ℂᵛ₃ = :qwthl2tune, ℂʷ₁ = :w2tune,
                                 ℂˡ₁ = :length_fac, ℂʷ₂ = :c_diag_3rd_mom,
                                 ℂᵝ₁ = :lambda_low, ℂᵝ₂ = :lambda_high, ℂᵝ₃ = :lambda_slope, ℂᵝ₄ = :lambda_thresh,
                                 ℂᶜ = :Ckh, ℂᵘ = :Ckm, ℂᶜₛ = :Ckh_s, ℂᵘₛ = :Ckm_s)

"""
$(TYPEDEF)

A flat, read-only snapshot of the numbers that E3SM `shoc.F90` depends on, taken from a closure
with the default components: `SHOCParameters(closure)`. The closure does not
store it; to change a coefficient use [`with_parameters`](@ref) on the closure. The coefficients
are documented with the component that holds them.
"""
struct SHOCParameters{FT}
    ℂᵘ :: FT
    ℂᶜ :: FT
    ℂᵘₛ :: FT
    ℂᶜₛ :: FT
    ℂˡ₁ :: FT
    ℂˡ₂ :: FT
    ℂˡ₃ :: FT
    ℂˡ₄ :: FT
    ℂˡ₅ :: FT
    ℂᵛ₁ :: FT
    ℂᵛ₂ :: FT
    ℂᵛ₃ :: FT
    ℂʷ₁ :: FT
    ℂʷ₂ :: FT
    ℂᵝ₁ :: FT
    ℂᵝ₂ :: FT
    ℂᵝ₃ :: FT
    ℂᵝ₄ :: FT
    ℂᴰ₁ :: FT
    ℂᴰ₂ :: FT
    ℂᴾ :: FT
    ℂᶠ :: FT
    ℂᵍ₁ :: FT
    ℂᵍ₂ :: FT
    ℂᵍ₃ :: FT
    ℂᵍ₄ :: FT
    ℂᴴ₁ :: FT
    ℂᴴ₂ :: FT
    ℂᴴ₃ :: FT
    ℂᴴ₄ :: FT
    ℂᴰ₃ :: FT
    ℂᴰ₄ :: FT
    ℂᴰ₅ :: FT
    ℂᶠ₂ :: FT
    ℂᶠ₃ :: FT
    ℂᶠ₄ :: FT
    ℂᶠ₅ :: FT
    ℂʷ₃ :: FT
    ℂʷ₄ :: FT
    ℂʷ₅ :: FT
    ℂʷ₆ :: FT
    ℂʷ₇ :: FT
    ℂʷ₈ :: FT
    ℂʷ₉ :: FT
    ℂʷ₁₀ :: FT
    ℂʷ₁₁ :: FT
    ℂᴴ₅ :: FT
    ℂᴴ₆ :: FT
    ℂᴴ₇ :: FT
    ggr :: FT
    rgas :: FT
    rv :: FT
    cp :: FT
    lcond :: FT
    eps :: FT
    vk :: FT
    basetemp :: FT
    basepres :: FT
    troppres :: FT
    ustar_min :: FT
    pblmaxp :: FT
    maximum_skewness :: FT
    maxlen :: FT
    minlen :: FT
    maxtke :: FT
    mintke :: FT
    tinyw :: FT
    maxiso :: FT
    temp_crit :: FT
    pbl_trans :: FT
    wsmin :: FT
    ksrfmin :: FT
    tke_flux_ustar_min :: FT
    ufmin :: FT
    wtke_ustar_min :: FT
    reset_third_moment :: FT
    kbfs_min :: FT
    thl_tol :: FT
    rt_tol :: FT
    w_tol :: FT
    Tl_min :: FT
    plume_fraction_min :: FT
    plume_fraction_max :: FT
    plume_variance_ratio_max :: FT
end

function SHOCParameters(closure::SimplifiedHigherOrderClosure)
    scalars = closure_scalars(closure)
    names = fieldnames(SHOCParameters)
    all(name -> haskey(scalars, name), names) ||
        throw(ArgumentError("SHOCParameters is a snapshot of the default components, but $(summary(closure)) " *
                            "has no $(filter(name -> !haskey(scalars, name), names))."))
    return SHOCParameters(promote(map(name -> scalars[name], names)...)...)
end

Base.eltype(::SHOCParameters{FT}) where FT = FT
Base.summary(::SHOCParameters{FT}) where FT = "SHOCParameters{$FT}"

function Base.show(io::IO, p::SHOCParameters)
    print(io, summary(p), ":")
    for name in keys(SHOC_COEFFICIENT_NAMES)
        print(io, '\n', "    ", name, " = ", prettysummary(getfield(p, name)))
    end
end
