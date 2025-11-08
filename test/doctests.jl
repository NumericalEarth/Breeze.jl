# Loading `Breeze` into `Main` is necessary to work around
# <https://github.com/JuliaTesting/ParallelTestRunner.jl/issues/68>.
@eval Main using Breeze
using Documenter: DocMeta, doctest
DocMeta.setdocmeta!(Main.Breeze, :DocTestSetup, :(using Breeze); recursive = true)

doctest(Main.Breeze; manual = false)
