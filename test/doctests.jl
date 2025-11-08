using Breeze
using Documenter: DocMeta, doctest

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive = true)

doctest(Breeze; manual = false)
