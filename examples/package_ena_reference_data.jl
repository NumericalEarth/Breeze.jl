# Prepare a local Julia artifact containing the ARM inputs, SAM comparison
# subsets, and observations needed by the ENA examples. This does not publish
# a release. Run with `julia --project=examples examples/package_ena_reference_data.jl`.
# The resulting archive can be attached to a GitHub release, then registered
# in Artifacts.toml using its printed tree hash, SHA-256, and public download URL.
using Pkg.Artifacts: create_artifact, archive_artifact
using Downloads
using SHA

function package_ena_reference_data(; directory=joinpath(@__DIR__, "data", "ena"),
                                    output=joinpath(@__DIR__, "ena_output", "ena_reference_data_20170718_v1.tar.gz"))
    forcings = ("era5", "era5s1n0", "merra2", "merra2s1n0")
    names = ("snd", "lsf", "sfc", "prm", "grd", "README_LASSO-ENA.txt", "inputs.sha256",
             "sam_comparison.nc", "comparison_source.txt", "comparison.sha256")
    files = [joinpath("lasso", forcing, name) for forcing in forcings for name in names]
    append!(files, [joinpath("observations", name) for name in
                   ("enamwrret2turnC1.c1.20170718.031053.nc", "enasondewnpnC1.b1.20170718.113800.cdf")])
    all(path -> isfile(joinpath(directory, path)), files) || error("Download the ENA reference data before packaging.")
    hash = create_artifact() do artifact
        for file in files
            destination = joinpath(artifact, file)
            mkpath(dirname(destination))
            cp(joinpath(directory, file), destination)
        end
        cp(joinpath(@__DIR__, "setups", "ena_data_attribution.txt"), joinpath(artifact, "ATTRIBUTION.txt"))
        Downloads.download("https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                           joinpath(artifact, "LICENSE-CC-BY-4.0.txt"); timeout=30)
        open(joinpath(artifact, "SHA256SUMS"), "w") do io
            for file in sort(vcat(files, ["ATTRIBUTION.txt", "LICENSE-CC-BY-4.0.txt"]))
                println(io, bytes2hex(open(sha256, joinpath(artifact, file))), "  ", file)
            end
        end
    end
    mkpath(dirname(output))
    archive_artifact(hash, output)
    checksum = bytes2hex(open(sha256, output))
    write(output * ".sha256", checksum * "  " * basename(output) * "\n")
    @info "Prepared ENA reference artifact" output git_tree_sha1=string(hash) sha256=checksum
    return (; output, hash, checksum)
end

if abspath(PROGRAM_FILE) == @__FILE__
    package_ena_reference_data()
end
