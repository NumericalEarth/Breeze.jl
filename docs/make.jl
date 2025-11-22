using Breeze
using Documenter
using DocumenterCitations
using Literate

using CairoMakie
CairoMakie.activate!(type = "svg")
set_theme!(Theme(linewidth = 3))

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(@__DIR__, "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

examples_src_dir = joinpath(@__DIR__, "..", "examples")
literated_dir = joinpath(@__DIR__, "src", "literated")
mkpath(literated_dir)

example_scripts = [
    "thermal_bubble.jl",
    "wave_clouds.jl",
    # "prescribed_sst.jl", # this is a WIP
]

for script_file in example_scripts
    script_path = joinpath(examples_src_dir, script_file)
    Literate.markdown(script_path, literated_dir;
                      flavor = Literate.DocumenterFlavor(),
                      execute = true)
end

example_pages = Any[
    "Thermal bubble" => "literated/thermal_bubble.md",
    "Moist Kelvin-Helmholtz billows" => "literated/wave_clouds.md",
    # "Prescribed SST" => "literated/prescribed_sst.md",
]

makedocs(
    ;
    modules = [Breeze],
    sitename = "Breeze",
    plugins = [bib],
    pages=[
        "Home" => "index.md",
        "Examples" => example_pages,
        "Thermodynamics" => "thermodynamics.md",
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm phase saturation adjustment" => "microphysics/warm_phase_saturation_adjustment.md",
            "Mixed phase saturation adjustment" => "microphysics/mixed_phase_saturation_adjustment.md",
        ],
        "Developers" => Any[
            "Microphysics" => Any[
                "Microphysics Interface" => "developer/microphysics_interface.md",
            ],
        ],
        "Dycore equations and algorithms" => "dycore_equations_algorithms.md",
        "Appendix" => Any[
            "Notation" => "appendix/notation.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
        "Contributors guide" => "contributing.md",
    ],
    draft = false,
)
