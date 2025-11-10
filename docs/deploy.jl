using Documenter

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
