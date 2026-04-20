module AbstractOperations

using Oceananigans.AbstractOperations: @binary

# Register two-argument elementary functions useful for atmospheric diagnostics
# (e.g. wind direction from velocity components, periodic wrapping) so they can
# be composed with Fields in `BinaryOperation` trees without dropping to
# `interior(...)` / `@.` escape hatches.
@binary Base.atan
@binary Base.atand
@binary Base.mod

end # module
