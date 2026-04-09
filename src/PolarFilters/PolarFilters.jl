module PolarFilters

export PolarFilter,
       Shapiro121,
       AbstractRolloff,
       materialize_polar_filter

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: Center

include("polar_filter.jl")
include("apply_polar_filter.jl")

end # module PolarFilters
