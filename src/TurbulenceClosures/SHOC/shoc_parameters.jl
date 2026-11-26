#####
##### The coefficients of a `SimplifiedHigherOrderClosure`, as one flat view
#####
##### The empirical coefficients (written ℂ, as elsewhere in Breeze) are fields of the component
##### that uses them. `closure_parameters` collects them in one `NamedTuple`, and `with_parameters`
##### rebuilds a closure with some of them replaced, whichever components they belong to. The
##### names are unique across the components of a closure: `closure_scalars` collects them in a `NamedTuple`,
##### which throws on a repeated name.
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The components of `closure` that hold its coefficients, in the order of [`closure_parameters`](@ref).
"""
function closure_components end

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

Each closure holds its own copy of the coefficients of its components: those of a component that
another closure also uses are coefficients of this closure like the others.

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure()
keys(closure_parameters(closure))

# output
(:ℂˡ₁, :ℂˡ₂, :ℂˡ₃, :ℂˡ₄, :ℂˡ₅, :ℂᵘ, :ℂᶜ, :ℂᵝ₁, :ℂᵝ₂, :ℂᵝ₃, :ℂᵝ₄, :ℂᴰ₁, :ℂᴰ₂, :ℂᴰ₃, :ℂᴰ₄, :ℂᴰ₅, :ℂᴾ, :ℂᵛ₁, :ℂᵛ₂, :ℂᵛ₃, :ℂʷ₁, :ℂʷ₂, :ℂʷ₃, :ℂʷ₄, :ℂʷ₅, :ℂʷ₆, :ℂʷ₇, :ℂʷ₈, :ℂʷ₉, :ℂʷ₁₀, :ℂʷ₁₁, :ℂᵍ₁, :ℂᵍ₂, :ℂᵍ₃, :ℂᵍ₄, :ℂᵐ)
```
"""
function closure_parameters(closure)
    scalars = closure_scalars(closure)
    names = filter(is_calibratable, keys(scalars))
    return NamedTuple{names}(map(name -> scalars[name], names))
end

"""
The empirical coefficients of the components of SHOC that have a dimension: the eddy turnover
time scale `ℂˡ₅` (s) of [`BlendedMixingLength`](@ref).
"""
const DIMENSIONAL_COEFFICIENTS = (:ℂˡ₅,)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the names of the coefficients of `closure` that a calibration may train: those of
[`closure_parameters`](@ref) except the dimensional ones (`DIMENSIONAL_COEFFICIENTS`).

Not every name returned can be determined independently of the others: the five `ℂᴰ` of
[`ColumnStabilityDamping`](@ref) enter through one product, and `ℂˡ₂` of
[`BlendedMixingLength`](@ref) only divides by `ℂˡ₁`.
"""
trainable_parameter_names(closure) =
    filter(name -> name ∉ DIMENSIONAL_COEFFICIENTS, keys(closure_parameters(closure)))

# Extended by each closure (`TKEBasedTurbulenceClosure`: subgrid_distribution.jl)
function with_parameters end
