using Breeze
using Documenter
using DocumenterCitations
using Literate

using CairoMakie
CairoMakie.activate!(type = "png")
set_theme!(Theme(linewidth = 3))

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(@__DIR__, "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

examples_src_dir = joinpath(@__DIR__, "..", "examples")
literated_dir = joinpath(@__DIR__, "src", "literated")
mkpath(literated_dir)
# We'll append the following postamble to the literate examples, to include
# information about the computing environment used to run them.
example_postamble = """

# ---

# ## Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

example_scripts = [
    "dry_thermal_bubble.jl",
    "cloudy_thermal_bubble.jl",
    "cloudy_kelvin_helmholtz.jl",
    "bomex.jl",
    "prescribed_sst.jl",
]

for script_file in example_scripts
    script_path = joinpath(examples_src_dir, script_file)
    Literate.markdown(script_path, literated_dir;
                      flavor = Literate.DocumenterFlavor(),
                      preprocess = content -> content * example_postamble,
                      execute = true)
end

example_pages = Any[
    "Stratified dry thermal bubble" => "literated/dry_thermal_bubble.md",
    "Cloudy thermal bubble" => "literated/cloudy_thermal_bubble.md",
    "Cloudy Kelvin-Helmholtz instability" => "literated/cloudy_kelvin_helmholtz.md",
    "Shallow cumulus convection (BOMEX)" => "literated/bomex.md",
    "Prescribed SST convection" => "literated/prescribed_sst.md",
]

makedocs(
    ;
    modules = [Breeze],
    sitename = "Breeze",
    plugins = [bib],
    format = Documenter.HTML(
        ;
        size_threshold_warn = 2 ^ 19, # 512 KiB
        size_threshold = 2 ^ 20, # 1 MiB
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => example_pages,
        "Thermodynamics" => "thermodynamics.md",
        "AtmosphereModel" => Any[
            "Diagnostics" => "atmosphere_model/diagnostics.md",
        ],
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm-phase saturation adjustment" => "microphysics/warm_phase_saturation_adjustment.md",
            "Mixed-phase saturation adjustment" => "microphysics/mixed_phase_saturation_adjustment.md",
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
    linkcheck = true,
    draft = false,
)

"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
function recursive_find(directory, pattern)
    mapreduce(vcat, walkdir(directory)) do (root, dirs, filenames)
        matched_filenames = filter(contains(pattern), filenames)
        map(filename -> joinpath(root, filename), matched_filenames)
    end
end

@info "Cleaning up temporary .jld2 and .nc output created by doctests or literated examples..."

for pattern in [r"\.jld2", r"\.nc"]
    filenames = recursive_find(@__DIR__, pattern)

    for filename in filenames
        rm(filename)
    end
end
