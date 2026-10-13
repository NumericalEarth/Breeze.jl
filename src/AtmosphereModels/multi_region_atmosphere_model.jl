#####
##### Multi-region grids: see the Multi-Region Grids section of .claude/rules/kernel-rules.md
#####

using Breeze.Utils: @regional
using Oceananigans: prognostic_fields
using Oceananigans.MultiRegion: ImmersedMultiRegionGrid, MultiRegionGrids
using Oceananigans.Utils: Utils, @apply_regionally, regions

const MultiRegionAtmosphereModel = AtmosphereModel{<:Any, <:Any, <:Any, <:Any, <:MultiRegionGrids}

@inline Utils.isregional(::MultiRegionAtmosphereModel) = true
@inline Utils.regions(model::MultiRegionAtmosphereModel) = regions(model.grid)

validate_grid(::ImmersedMultiRegionGrid) =
    throw(ArgumentError("AtmosphereModel does not support an ImmersedBoundaryGrid built on a MultiRegionGrid."))

@regional AtmosphereModel
@regional AtmosphereModelBuoyancy
@regional SlowDynamics
@regional SedimentingCondensate
@regional OutflowEnabledVelocity
@regional DensityWeightedImplicitOperator

# A connected multi-region grid rotates the horizontal pair `(u, v)` across regions, so the pair is filled
# together. `Vararg{Any, N}` makes the compiler specialize on the forwarded boundary-condition arguments.
function fill_vector_halo_regions!(𝐯::NamedTuple, args::Vararg{Any, N}; kwargs...) where N
    u, v, w = 𝐯
    fill_halo_regions!((u, v), args...; kwargs...)
    fill_halo_regions!(w, args...; kwargs...)
    return nothing
end

# Dynamics without prognostic momentum carry an empty momentum tuple
fill_vector_halo_regions!(::NamedTuple{()}, args...; kwargs...) = nothing

function fill_prognostic_halo_regions!(model::AtmosphereModel, args::Vararg{Any, N}; kwargs...) where N
    momentum = model.momentum
    fill_vector_halo_regions!(momentum, args...; kwargs...)
    scalars = Base.structdiff(prognostic_fields(model), momentum)
    fill_halo_regions!(scalars, args...; kwargs...)
    return nothing
end

# TODO: a multi-region field has no single parent array. Oceananigans should support `copyto!` between
# multi-region fields, halos included.
copy_parent!(dst, src) = @apply_regionally copyto!(parent(dst), parent(src))
