# # Package the reduced Shen et al. (2022) LES library as a lazy artifact
#
# After `reduce_shen_les_library.jl` has filled a directory with reduced members, this script
# turns it into the `shen_et_al_2022_les_profiles` artifact of `Artifacts.toml`: it hashes the directory,
# writes a gzipped tarball, and prints the `Artifacts.toml` entry. The tarball is uploaded to a
# GitHub release of Breeze.jl by hand (as `P3_lookup_tables` was) at the URL the entry names.
#
#     julia --project build_artifact.jl reduced shen_et_al_2022_les_profiles.tar.gz [v1.0]
#
# The optional third argument is the version in the release tag `shen_et_al_2022_les_profiles_<version>`.
# Every update is a new tag: the artifact is pinned by hash, so an asset is never replaced in place.
# `examples/single_column_tke_boundary_layer.jl` reads the members through `Artifacts.toml`.

using Pkg.Artifacts: create_artifact, archive_artifact
using SHA

const ARTIFACT_NAME = "shen_et_al_2022_les_profiles"

function main(args)
    reduced = length(args) ≥ 1 ? args[1] : "reduced"
    tarball = length(args) ≥ 2 ? args[2] : "$ARTIFACT_NAME.tar.gz"
    version = length(args) ≥ 3 ? args[3] : "v1.0"
    release_url = "https://github.com/NumericalEarth/Breeze.jl/releases/download/$(ARTIFACT_NAME)_$(version)/$(ARTIFACT_NAME).tar.gz"
    isdir(reduced) || error("$reduced is not a directory")

    members = filter(endswith(".nc"), readdir(reduced))
    @info "Packaging $(length(members)) reduced members from $reduced"

    tree_hash = create_artifact() do directory
        for member in members
            cp(joinpath(reduced, member), joinpath(directory, member))
        end
    end

    archive_artifact(tree_hash, tarball)
    tarball_sha256 = bytes2hex(open(sha256, tarball))

    println("\nAdd to Artifacts.toml (and upload $tarball to $release_url):\n")
    println("""
    [$ARTIFACT_NAME]
    git-tree-sha1 = "$tree_hash"
    lazy = true

        [[$ARTIFACT_NAME.download]]
        sha256 = "$tarball_sha256"
        url = "$release_url"
    """)
    @info "Tarball" tarball size_MB = round(filesize(tarball) / 1e6, digits = 1)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
