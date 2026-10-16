# Request the four 18 July 2017 LASSO SAM input bundles (morr-aer2-flxsst)
# at https://adc.arm.gov/lasso/#/ena, then pass the FTP directory from ARM's
# staging email: julia --project=examples examples/download_lasso_ena.jl <ftp-directory>
# Staging directories expire. Input provenance is retained with SHA-256 hashes.
# Add `--statistics` to download the four matching SAM domain statistics instead.
# Request `samstat` alongside `samin` when placing the ARM order.
using Downloads
using Tar
using SHA

function extract_lasso_ena(archive, directory)
    names = ("snd", "lsf", "sfc", "prm", "grd")
    mkpath(directory)
    mktempdir() do temporary
        Tar.extract(header -> header.type == :file &&
                    (header.path in ("LASSO_ENA/" .* names) || header.path == "README_LASSO-ENA.txt"),
                    archive, temporary)
        for name in names
            cp(joinpath(temporary, "LASSO_ENA", name), joinpath(directory, name); force=true)
        end
        cp(joinpath(temporary, "README_LASSO-ENA.txt"), joinpath(directory, "README_LASSO-ENA.txt"); force=true)
    end
    open(joinpath(directory, "inputs.sha256"), "w") do io
        for name in names
            println(io, bytes2hex(open(sha256, joinpath(directory, name))), "  ", name)
        end
    end
    return directory
end

function download_lasso_ena(staging_url; directory=joinpath(@__DIR__, "data", "ena", "lasso"))
    for forcing in ("era5", "era5s1n0", "merra2", "merra2s1n0")
        output = joinpath(directory, forcing)
        all(name -> isfile(joinpath(output, name)), ("snd", "lsf", "sfc", "prm", "grd")) && continue
        stream = "enalasso_samin_20170718$(forcing)d25x100_morr-aer2-flxsstC1.m0"
        filename = stream * ".20170718.000000.tar.v0"
        mktempdir() do temporary
            archive = joinpath(temporary, filename)
            Downloads.download(rstrip(staging_url, '/') * "/$stream/$filename", archive; timeout=600)
            extract_lasso_ena(archive, output)
        end
    end
    return directory
end

function download_lasso_ena_statistics(staging_url; directory=joinpath(@__DIR__, "data", "ena", "lasso"))
    for forcing in ("era5", "era5s1n0", "merra2", "merra2s1n0")
        output = joinpath(directory, forcing)
        mkpath(output)
        path = joinpath(output, "sam_statistics.nc")
        stream = "enalasso_samstat_20170718$(forcing)d25x100_morr-aer2-flxsstC1.m1"
        filename = stream * ".20170718.000000.nc.v0"
        if !isfile(path)
            temporary = path * ".download"
            @info "Downloading archived SAM statistics" forcing
            try
                Downloads.download(rstrip(staging_url, '/') * "/$stream/$filename", temporary; timeout=600)
                mv(temporary, path)
            finally
                isfile(temporary) && rm(temporary)
            end
        end
        write(joinpath(output, "statistics.sha256"), bytes2hex(open(sha256, path)) * "  sam_statistics.nc\n")
        write(joinpath(output, "statistics_source.txt"),
              "ARM LASSO-ENA SAM statistics: https://doi.org/10.5439/2572635\n" *
              "archive_file = $filename\n")
    end
    return directory
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1
        download_lasso_ena(only(ARGS))
    elseif length(ARGS) == 2 && ARGS[2] == "--statistics"
        download_lasso_ena_statistics(ARGS[1])
    else
        error("Pass the FTP staging directory from your ARM email, optionally followed by --statistics.")
    end
end
