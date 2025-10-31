using Breeze
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(dirname(@__FILE__), "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

makedocs(sitename="Breeze",
    plugins = [bib],
    pages=[
        "Home" => "index.md",
        "Thermodynamics" => "thermodynamics.md",
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm phase saturation adjustment" => "microphysics/saturation_adjustment.md",
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
