using Downloads: Downloads
using SHA: sha256
using p7zip_jll: p7zip
using ArtifactUtils: add_artifact!, artifact_from_directory
using gh_cli_jll: gh_cli_jll
using Pkg.Artifacts: archive_artifact

const P3_TABLE_BASE_URL = "https://github.com/P3-microphysics/P3-microphysics/raw/f7b3216d8f5d006621425fee272e17f355b50f09/lookup_tables/"
const P3_TABLE_FILES = Dict(
    "p3_lookupTable_1.dat-v6.9-2momI.gz" => "3a89d320f755daa66f8b93956dbe41c7b3d9e79b160c067210e24367e61ee1e5",
    "p3_lookupTable_1.dat-v6.9-3momI.gz" => "26a44b623d5de4355e42e843cf19b3245709b359261c4234fee301a3bbf186e7",
    "p3_lookupTable_3.dat-v1.4.gz" => "30fa735ea5f0bc8009c72d034ce03d957831af11cd4b49414ac5ab57cbda8ddc",
)

const RELEASE_NAME = get(ARGS, 1, "P3_lookup_tables_v1.0")

mktempdir() do dir
    repo = "NumericalEarth/Breeze.jl"
    tarball_name = "P3_lookup_tables.tar.gz"
    tarball_path = joinpath(dir, tarball_name)
    gh = addenv(gh_cli_jll.gh(), "DO_NOT_TRACK" => "true")

    for (file, hash) in P3_TABLE_FILES
        path = joinpath(dir, file)
        # Download the file
        Downloads.download(P3_TABLE_BASE_URL * file, path)
        open(path, "r") do io
            computed_hash = bytes2hex(sha256(io))
            if computed_hash != hash
                error("Download of $(file) failed: computed SHA256 hash ($(computed_hash)) is different from expected one ($(hash))")
            end
        end
        # Uncompress it
        run(`$(p7zip()) e -o$(dir) $(path)`)
        # Delete the compressed one
        rm(path)
    end

    @info "Creating artifact from directory" dir
    artifact_id = artifact_from_directory(dir; follow_symlinks = false)
    @info "Creating archive from artifact (could take some time)" artifact_id tarball_path
    archive_artifact(artifact_id, tarball_path)

    @info "Creating release with tag $(RELEASE_NAME)"
    run(`$(gh) release create $(RELEASE_NAME) --repo $(repo) --title $(RELEASE_NAME) --notes ""`)

    @info("Uploading files to $(repo) with tag $(RELEASE_NAME)", tarball_path)
    run(`$(gh) release upload $(RELEASE_NAME) $(tarball_path) --repo $(repo) --clobber`)

    add_artifact!(
        joinpath(dirname(dirname(@__DIR__)), "Artifacts.toml"),
        "P3_lookup_tables",
        "https://github.com/$(repo)/releases/download/$(RELEASE_NAME)/$(tarball_name)";
        platform=nothing,
        lazy=true,
        force=true,
    )
end
