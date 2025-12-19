using Breeze
using RRTMGP, CloudMicrophysics # to load Breeze extensions
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

# ### Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

struct Example
    # Title of the example page in `Documenter` ToC
    title::String
    # Basename of the example file, without extension (`.jl` will be appended for the input
    # to `Literate.markdown`, `.md` will be appended for the generated file)
    basename::String
    # Whether to always build this example: set it to `false` for long-running examples to
    # be built only on `main` or on-demand in PRS.
    build_always::Bool
end

examples = [
    Example("Stratified dry thermal bubble", "dry_thermal_bubble", true),
    Example("Cloudy thermal bubble", "cloudy_thermal_bubble", true),
    Example("Cloudy Kelvin-Helmholtz instability", "cloudy_kelvin_helmholtz", true),
    Example("Shallow cumulus convection (BOMEX)", "bomex", true),
    Example("Precipitating shallow cumulus (RICO)", "rico", true),
    Example("Convection over prescribed sea surface temperature (SST)", "prescribed_sea_surface_temperature", true),
    Example("Inertia gravity wave", "inertia_gravity_wave", true),
    Example("Single column gray radiation", "single_column_radiation", true),
]

# Filter out long-running example if necessary
filter!(x -> x.build_always || get(ENV, "BREEZE_BUILD_ALL_EXAMPLES", "false") == "true", examples)

example_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in examples]

literate_code(script_path, literated_dir) = """
using Literate
using CairoMakie

CairoMakie.activate!(type = "png")
set_theme!(Theme(linewidth = 3))

@time $(repr(basename(script_path))) Literate.markdown($(repr(script_path)), $(repr(literated_dir));
                                                        flavor = Literate.DocumenterFlavor(),
                                                        preprocess = content -> content * $(repr(example_postamble)),
                                                        execute = true,
                                                       )
"""

semaphore = Base.Semaphore(Threads.nthreads(:interactive))
@time "literate" @sync for example in examples
    script_file = example.basename * ".jl"
    script_path = joinpath(examples_src_dir, script_file)
    Threads.@spawn :interactive Base.acquire(semaphore) do
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) -e $(literate_code(script_path, literated_dir))`)
    end
end

modules = Module[]
BreezeRRTMGPExt = isdefined(Base, :get_extension) ? Base.get_extension(Breeze, :BreezeRRTMGPExt) : Breeze.BreezeRRTMGPExt
BreezeCloudMicrophysicsExt = isdefined(Base, :get_extension) ? Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt) : Breeze.BreezeCloudMicrophysicsExt

for m in [Breeze, BreezeRRTMGPExt, BreezeCloudMicrophysicsExt]
    if !isnothing(m)
        push!(modules, m)
    end
end

makedocs(
    ;
    modules,
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
        "Radiative Transfer" => "radiative_transfer.md",
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
