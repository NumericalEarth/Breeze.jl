# Manual literate-and-publish driver for the Breeze.jl validation studies.
#
# Validation studies live in `validation/<Study>/` as Literate-formatted scripts. Unlike the
# examples (which `docs/make.jl` builds and executes as part of the documentation), validation
# studies are in-depth and computationally expensive — they are run MANUALLY on appropriate
# hardware. This driver renders a study's committed narrative + figures WITHOUT re-executing it
# (`execute = false`): the figures referenced by the document must already exist (produced by
# running the study), so rendering is cheap and needs no GPU or simulation dependencies.
#
# Two output modes:
#   * default — render each study to GitHub-viewable markdown in place (needs only Literate).
#   * `--html` — build a browsable Documenter site under `validation/build/`: the top-level
#     `README.md` is the home page and each study is an entry in the sidebar table of contents
#     (needs Documenter, loaded only in this mode).
#
# Usage (from the repository root):
#   julia --project=validation validation/make.jl                      # render all studies (markdown)
#   julia --project=validation validation/make.jl DCMIP2016_TC         # render selected study/studies
#   julia --project=validation validation/make.jl --html              # build the HTML site (all studies)
#   julia --project=validation validation/make.jl --html DCMIP2016_TC # HTML site with selected studies
#
# To add a study, drop it in `validation/<Study>/` and register it in `STUDIES` below.

using Literate

const VALIDATION_DIR = @__DIR__

# Each study is registered as (subdirectory, Literate document script, human-readable title).
# The document script is the single file rendered for that study (e.g. the intercomparison,
# not the simulation generator it `include`s). The title is its table-of-contents entry in the
# HTML site. Command-line arguments (other than `--html`) select studies by their subdirectory —
# the first tuple element — e.g. `make.jl DCMIP2016_TC`.
const STUDIES = [
    ("DCMIP2016_TC", "dcmip2016_tc_intercomparison.jl", "DCMIP2016 tropical cyclone"),
]

# Render one study's Literate document to GitHub-viewable markdown, in place in its directory.
function render_study(dir, doc, title)
    studydir = joinpath(VALIDATION_DIR, dir)
    docpath  = joinpath(studydir, doc)
    isfile(docpath) || error("validation document not found: $docpath")
    out = splitext(doc)[1] * ".md"
    @info "literating \"$title\": $(joinpath(dir, doc)) → $(joinpath(dir, out))"
    # execute = false: do not re-run the (expensive) study; embed the committed figures it
    # references. CommonMarkFlavor yields plain markdown that renders directly on GitHub.
    Literate.markdown(docpath, studydir;
                      flavor = Literate.CommonMarkFlavor(),
                      execute = false)
    # Literate leaves a trailing blank line; normalize to a single terminating newline so the
    # committed markdown passes the repository whitespace checks and re-renders stay clean.
    mdpath = joinpath(studydir, out)
    write(mdpath, rstrip(read(mdpath, String)) * "\n")
    return nothing
end

# Build a browsable Documenter HTML site: the top-level README is the home page and each study
# is a sidebar table-of-contents entry. Documenter is referenced qualified, so it only needs to
# be loaded (via `@eval import Documenter`) when this mode is selected.
function build_html(studies)
    src = joinpath(VALIDATION_DIR, "build_src")          # staged Documenter source tree
    rm(src; recursive = true, force = true)
    mkpath(src)
    cp(joinpath(VALIDATION_DIR, "README.md"), joinpath(src, "index.md"); force = true)  # home page
    pages = Any["Home" => "index.md"]
    for (dir, doc, title) in studies
        studydir = joinpath(VALIDATION_DIR, dir)
        outdir   = joinpath(src, dir)
        mkpath(outdir)
        # CommonMarkFlavor (not DocumenterFlavor): the study chunks become static ```julia
        # blocks that Documenter renders but never runs. DocumenterFlavor would emit `@example`
        # blocks that Documenter *executes* at build time — which would try to launch the
        # simulations. Rendered into the staging tree; the committed GitHub markdown is untouched.
        Literate.markdown(joinpath(studydir, doc), outdir;
                          flavor = Literate.CommonMarkFlavor(), execute = false)
        # Copy the committed figures so the document's relative image links resolve, including
        # any in subdirectories (e.g. the Willson comparison PNGs under postproc/).
        for (root, _, files) in walkdir(studydir)
            rel = relpath(root, studydir)
            for f in files
                endswith(f, ".png") || continue
                dest = rel == "." ? outdir : joinpath(outdir, rel)
                mkpath(dest)
                cp(joinpath(root, f), joinpath(dest, f); force = true)
            end
        end
        push!(pages, title => joinpath(dir, splitext(doc)[1] * ".md"))
    end
    Documenter.makedocs(;
        sitename = "Breeze.jl validation",
        root     = VALIDATION_DIR,
        source   = "build_src",
        build    = "build",
        pages    = pages,
        format   = Documenter.HTML(prettyurls = false, edit_link = nothing, repolink = nothing),
        remotes  = nothing,
        warnonly = true,   # local site: don't fail on cross-render relative-link warnings
    )
    rm(src; recursive = true, force = true)   # drop the staging tree; keep build/
    @info "HTML site: $(joinpath(VALIDATION_DIR, "build", "index.html"))"
    return nothing
end

build_html_mode = "--html" in ARGS
study_args = filter(a -> a != "--html", ARGS)

selected = isempty(study_args) ? STUDIES : filter(s -> first(s) in study_args, STUDIES)
isempty(selected) && error("no matching study in $(first.(STUDIES)); requested $(study_args)")

if build_html_mode
    @eval import Documenter   # lazy: only loaded when building the HTML site
    Base.invokelatest(build_html, selected)
else
    for (dir, doc, title) in selected
        render_study(dir, doc, title)
    end
    @info "rendered $(length(selected)) validation study/studies."
end
