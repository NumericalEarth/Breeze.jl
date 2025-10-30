using Breeze
using Documenter

DocMeta.setdocmeta!(Breeze, :DocTestSetup, :(using Breeze); recursive=true)

makedocs(sitename="Breeze",
    pages=[
        "Home" => "index.md",
        "Thermodynamics" => "thermodynamics.md",
        "Microphysics" => Any[
            "Overview" => "microphysics/microphysics_overview.md",
            "Warm phase saturation adjustment" => "microphysics/saturation_adjustment.md",
        ],
        "API" => "api.md",
    ]
)

deploydocs(;
    repo = "github.com/NumericalEarth/Breeze.jl",
    devbranch = "main",
    push_preview = true,
    forcepush = true
)
