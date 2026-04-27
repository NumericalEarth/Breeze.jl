# Loading `Breeze` into `Main` is necessary to work around
# <https://github.com/JuliaTesting/ParallelTestRunner.jl/issues/68>.
@eval Main using Breeze
using Documenter: DocMeta, doctest
using Logging: global_logger, ConsoleLogger

# Force extended logging messages
global_logger(ConsoleLogger(; show_limited=false))

DocMeta.setdocmeta!(Main.Breeze, :DocTestSetup, :(using Breeze); recursive = true)

doctest(Main.Breeze; manual = false)
