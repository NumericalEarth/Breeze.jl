module AbstractOperations

# TODO: simplify these imports once `Oceananigans.AbstractOperations.@binary` no
# longer requires them. The macro expansion references non-exported helpers
# (`apply_op`, `grid_metric_operation`, `choose_location`, `BinaryOperation`,
# `GridMetric`) as bare symbols, so downstream callers must bring them into
# scope manually. Fully qualifying those references in `define_binary_operator`
# upstream would let us drop everything except `@binary` and the `Base`
# functions being registered.

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: @binary, BinaryOperation, GridMetric, choose_location
using Oceananigans.AbstractOperations: apply_op, grid_metric_operation

import Base: atan, atand, mod

# Register two-argument elementary functions useful for atmospheric diagnostics
# (e.g. wind direction from velocity components, periodic wrapping) so they can
# be composed with Fields in `BinaryOperation` trees without dropping to
# `interior(...)` / `@.` escape hatches.
@binary atan
@binary atand
@binary mod

end # module
