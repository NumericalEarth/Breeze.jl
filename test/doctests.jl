include(joinpath(@__DIR__, "setup.jl"))

# Loading `Breeze` into `Main` is necessary to work around
# <https://github.com/JuliaTesting/ParallelTestRunner.jl/issues/68>.
@eval Main using Breeze
using Documenter: DocMeta, doctest
using Logging: global_logger, ConsoleLogger

# Force extended logging messages
global_logger(ConsoleLogger(; show_limited=false))

# Another test on this worker may have left a Float32 default behind
Oceananigans.defaults.FloatType = Float64

DocMeta.setdocmeta!(Main.Breeze, :DocTestSetup, :(using Breeze); recursive = true)

doctest(Main.Breeze; manual = false)
