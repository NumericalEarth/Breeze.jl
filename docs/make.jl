using Breeze
using RRTMGP, CloudMicrophysics # to load Breeze extensions
using Documenter
using DocumenterCitations

using CairoMakie
CairoMakie.activate!(type = "png")
set_theme!(Theme(linewidth = 3))

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

bib_filepath = joinpath(@__DIR__, "src", "breeze.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

examples_src_dir = joinpath(@__DIR__, "..", "examples")
literated_dir = joinpath(@__DIR__, "src", "literated")
mkpath(literated_dir)

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
    Example("Precipitating shallow cumulus (RICO)", "rico", false),
    Example("Convection over prescribed sea surface temperature (SST)", "prescribed_sea_surface_temperature", true),
    Example("Inertia gravity wave", "inertia_gravity_wave", true),
    Example("Single column radiation", "single_column_radiation", true),
    Example("Stationary parcel model", "stationary_parcel_model", true),
    Example("Rising parcel: adiabatic ascent", "rising_parcels", true),
    Example("Acoustic wave in shear layer", "acoustic_wave", true),
    Example("Cloud formation in prescribed updraft", "kinematic_driver", true),
    Example("Splitting supercell", "splitting_supercell", false),
    Example("Tropical cyclone world", "tropical_cyclone_world", false),
]

# Filter out long-running example if necessary
filter!(x -> x.build_always || get(ENV, "BREEZE_BUILD_ALL_EXAMPLES", "false") == "true", examples)
example_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in examples]
semaphore = Base.Semaphore(Threads.nthreads(:interactive))
@time "literate" @sync for example in examples
    script_file = example.basename * ".jl"
    script_path = joinpath(examples_src_dir, script_file)
    Threads.@spawn :interactive Base.acquire(semaphore) do
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(literated_dir)`)
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

# Automatically generate file with docstrings for all modules

function walk_submodules!(result, visited, mod::Module)
    for name in sort(names(mod; all=true, imported=false))
        isdefined(mod, name) || continue
        value = getproperty(mod, name)
        if value isa Module &&
            parentmodule(value) === mod &&
            !(value in visited) &&
            value !== mod

            push!(visited, value)
            push!(result, value)
            walk_submodules!(result, visited, value)
        end
    end
end

function get_submodules(mod::Module)
    result = Module[]
    visited = Set{Module}()

    walk_submodules!(result, visited, mod)
    return result
end

function write_api_md()
    modules = get_submodules(Breeze)
    append!(modules, [BreezeRRTMGPExt, BreezeCloudMicrophysicsExt])
    io = IOBuffer()

    println(io, """
            # API Documentation

            ## Public API

            ```@autodocs
            Modules = [Breeze]
            Private = false
            ```
            """)
    for mod in modules
        println(io, """
                ### $(chopprefix(string(mod), "Breeze."))

                ```@autodocs
                Modules = [$(mod)]
                Private = false
                ```
                """)
    end
    println(io, """
            ## Private API

            ```@autodocs
            Modules = [Breeze]
            Public = false
            ```
            """)
    for mod in modules
        println(io, """
                ### $(chopprefix(string(mod), "Breeze."))

                ```@autodocs
                Modules = [$(mod)]
                Public = false
                ```
                """)
    end

    # Remove multiple trailing whitespaces, but keep the final one.
    write(joinpath(@__DIR__, "src", "api.md"), strip(String(take!(io))) * "\n")
end

write_api_md()

# Let's build the docs!

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
                "Overview" => "developer/microphysics/overview.md",
                "Example implementation" => "developer/microphysics/example.md",
                "Future improvements" => "developer/microphysics/future_improvements.md",
            ],
        ],
        "Radiative Transfer" => "radiative_transfer.md",
        "Dynamics" => Any[
            "Governing equations" => "dycore_equations_algorithms.md",
            "Anelastic dynamics" => "anelastic_dynamics.md",
            "Compressible dynamics" => "compressible_dynamics.md",
        ],
        "Appendix" => Any[
            "Notation" => "appendix/notation.md",
            "Reproducibility of Breeze.jl models" => "reproducibility.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
        "Contributors guide" => "contributing.md",
    ],
    linkcheck = true,
    draft = false,
    doctest = true
)
