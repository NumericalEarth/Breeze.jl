using Oceananigans: ReactantState
using Breeze: Utils

# On ReactantState a model is initialized by compiling and running `initialize!` itself, so construction
# leaves it uninitialized rather than running the initialization eagerly.
Utils.initialize_on_construction!(::ReactantState, x, args...) = nothing
