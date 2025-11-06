using Breeze
using Documenter
using DocumenterCitations

using CairoMakie
CairoMakie.activate!(type = "svg")
set_theme!(Theme(linewidth = 3))

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(dirname(@__FILE__), "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

makedocs(sitename="Breeze",
    plugins = [bib],
    pages=[
        "Home" => "index.md",
        "Thermodynamics" => "thermodynamics.md",
        "Dynamics" => "dynamics.md",
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm phase saturation adjustment" => "microphysics/saturation_adjustment.md",
        ],
        "Appendix" => Any[
            "Notation" => "appendix/notation.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
    ]
)

deploydocs(;
    repo = "github.com/NumericalEarth/Breeze.jl",
    devbranch = "main",
    push_preview = true,
    forcepush = true
)
