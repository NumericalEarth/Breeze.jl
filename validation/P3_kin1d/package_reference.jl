# Package existing official-P3 trajectories for download without a Fortran build.
# Diagnostic interventions are deliberately excluded from the reference artifact.
using Pkg.Artifacts
using DelimitedFiles
using SHA

repository = normpath(joinpath(@__DIR__, "../.."))
output = joinpath(repository, "validation_output")
processes = joinpath(output, "p3_processes")
matrix = joinpath(output, "p3_matrix")
release = joinpath(output, "p3_reference_release")
mkpath(release)

function copy_reference_files(source, destination, names)
    mkpath(destination)
    for name in names
        cp(joinpath(source, name), joinpath(destination, name))
    end
end

hash = create_artifact() do directory
    cp(joinpath(@__DIR__, "reference_data.txt"), joinpath(directory, "README.txt"))
    write(joinpath(directory, "packaging_revision.txt"), read(`git -C $repository rev-parse HEAD`, String))
    copy_reference_files(processes, joinpath(directory, "provenance"),
                         ("provenance.txt", "reference.patch"))
    copy_reference_files(matrix, joinpath(directory, "matrix"),
                         ("cases.dat", "case_columns.txt"))
    for dt in ("0.003125", "0.0015625", "0.00078125")
        name = "reference_dt$dt"
        source = joinpath(matrix, name)
        data = readdlm(joinpath(source, "reference_final.dat"))
        @assert size(data, 2) == 15 && all(isfinite, data)
        @assert all(count(==(id), data[:, 1]) == 540 for id in unique(data[:, 1]))
        copy_reference_files(source, joinpath(directory, "matrix", name),
                             ("cases.dat", "reference_final.dat", "trajectory_probe.f90"))
    end
    for dt in ("0.00625", "0.003125", "0.0015625")
        source = joinpath(processes, "trajectories_dt$(dt)_liquid_ice")
        destination = joinpath(directory, "two_ascents", "reference_dt$dt")
        copy_reference_files(source, destination, ("cases.dat", "trajectory_probe.f90"))
        data = readdlm(joinpath(source, "reference_final.dat"))
        data = data[data[:, 1] .>= 13, :]
        @assert size(data) == (1080, 15) && all(isfinite, data)
        writedlm(joinpath(destination, "reference_final.dat"), data)
    end
    copy_reference_files(joinpath(matrix, "outliers", "baseline"), joinpath(directory, "event_diagnostics"),
                         ("reference_rates.dat", "rate_columns.txt", "intervention.patch", "trajectory_probe.f90"))
    @assert readdlm(joinpath(matrix, "outliers", "baseline", "reference_final.dat")) ==
            readdlm(joinpath(matrix, "reference_dt0.00078125", "reference_final.dat"))
    open(joinpath(directory, "SHA256SUMS"), "w") do io
        for (root, _, files) in walkdir(directory), file in sort(files)
            file == "SHA256SUMS" && continue
            path = joinpath(root, file)
            println(io, bytes2hex(open(sha256, path)), "  ", relpath(path, directory))
        end
    end
end

archive = joinpath(release, "P3_parcel_reference.tar.gz")
archive_artifact(hash, archive)
checksum = bytes2hex(open(sha256, archive))
url = "https://github.com/NumericalEarth/Breeze.jl/releases/download/P3_parcel_reference_v1.0/P3_parcel_reference.tar.gz"
binding = """
[P3_parcel_reference]
git-tree-sha1 = "$hash"
lazy = true

    [[P3_parcel_reference.download]]
    sha256 = "$checksum"
    url = "$url"
"""
write(joinpath(release, "Artifacts.toml"), binding)
println(binding)
println("Archive: ", archive)
