using Breeze
using Documenter
using DocumenterCitations

using CairoMakie
CairoMakie.activate!(type = "svg")
set_theme!(Theme(linewidth = 3))

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(dirname(@__FILE__), "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

makedocs(
    ;
    sitename = "Breeze",
    plugins = [bib],
    pages=[
        "Home" => "index.md",
        "Thermodynamics" => "thermodynamics.md",
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm phase saturation adjustment" => "microphysics/saturation_adjustment.md",
        ],
        "Dycore equations and algorithms" => "dynamics.md",
        "Appendix" => Any[
            "Notation" => "appendix/notation.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
        "Contributors guide" => "contributing.md",
    ],
    draft = false,
)

deploydocs(
    ;
    repo = "github.com/NumericalEarth/Breeze.jl",
    deploy_repo = "github.com/NumericalEarth/BreezeDocumentation",
    devbranch = "main",
    # Only push previews if all the relevant environment variables are non-empty. This is an
    # attempt to work around https://github.com/JuliaDocs/Documenter.jl/issues/2048.
    push_preview = all(!isempty, (get(ENV, "GITHUB_TOKEN", ""), get(ENV, "DOCUMENTER_KEY", ""))),
    forcepush = true,
)
