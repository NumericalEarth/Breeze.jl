module PolarFilters

export PolarFilter,
       add_polar_filter!,
       apply_polar_filter!,
       SharpTruncation,
       ExponentialRolloff

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

using Oceananigans: Oceananigans, interior
using Oceananigans.Grids: Center

include("polar_filter.jl")
include("apply_polar_filter.jl")

end # module PolarFilters
