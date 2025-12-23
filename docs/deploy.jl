using Documenter

@info "Cleaning up temporary .jld2 and .nc output created by doctests or literated examples..."

for (root, _, filenames) in walkdir(@__DIR__)
    for file in filenames
        if any(occursin(file), (r"\.jld2$", r"\.nc$"))
            rm(joinpath(root, file))
        end
    end
end

deploydocs(;
    repo = "github.com/NumericalEarth/Breeze.jl",
    deploy_repo = "github.com/NumericalEarth/BreezeDocumentation",
    devbranch = "main",
    # Only push previews if all the relevant environment variables are non-empty. This is an
    # attempt to work around https://github.com/JuliaDocs/Documenter.jl/issues/2048.
    push_preview = all(!isempty, (get(ENV, "GITHUB_TOKEN", ""), get(ENV, "DOCUMENTER_KEY", ""))),
    forcepush = true
)
