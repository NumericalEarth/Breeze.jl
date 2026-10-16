# Download only the July 18 observations used by compare_ena_results.jl.
# Export ARM_USERNAME and ARM_TOKEN before running. Credentials stay in memory.
using Downloads
using Printf

function ena_url_component(value)
    return join(map(codeunits(value)) do byte
        safe = UInt8('a') ≤ byte ≤ UInt8('z') || UInt8('A') ≤ byte ≤ UInt8('Z') ||
               UInt8('0') ≤ byte ≤ UInt8('9') || byte in codeunits("-._~")
        safe ? string(Char(byte)) : @sprintf("%%%02X", byte)
    end)
end

function download_ena_observations(directory=joinpath(@__DIR__, "data", "ena", "observations");
                                   turbulence=false)
    username = ENV["ARM_USERNAME"]
    token = ENV["ARM_TOKEN"]
    user = ena_url_component(username * ":" * token)
    files = ("enamwrret2turnC1.c1.20170718.031053.nc",
             "enasondewnpnC1.b1.20170718.113800.cdf",
             "enaceilC1.b1.20170718.000012.nc")
    if turbulence
        files = (files..., "enadlprofwstats4newsC1.c1.20170718.000000.nc",
                 "enasondewnpnC1.b1.20170718.053100.cdf")
    end
    mkpath(directory)
    for filename in files
        path = joinpath(directory, filename)
        isfile(path) && continue
        url = "https://adc.arm.gov/armlive/saveData?user=$user&file=$filename"
        temporary = path * ".download"
        try
            Downloads.download(url, temporary; timeout=60)
            mv(temporary, path)
        catch
            isfile(temporary) && rm(temporary)
            # Downloads errors can include the authenticated URL. Do not rethrow it.
            error("ARM download failed for $filename. Check credentials and archive availability.")
        end
    end
    return directory
end

if abspath(PROGRAM_FILE) == @__FILE__
    download_ena_observations()
end
