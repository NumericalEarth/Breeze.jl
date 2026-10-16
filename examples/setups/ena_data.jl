# Input adapters for the 18 July 2017 Eastern North Atlantic case.
# Source: https://github.com/dmechem/ENA_variability_LES_bulk_paper
# Covert et al. (2022): https://doi.org/10.5194/acp-22-1159-2022

using Breeze
using Dates
using Downloads
using DelimitedFiles

struct ENACase{P, S}
    name :: String
    date :: Date
    source :: String
    profiles :: P
    surface :: S
end

function download_covert_ena(directory=joinpath(@__DIR__, "..", "data", "ena", "covert_20170718"))
    revision = "a403498ff44045b531af5dbfe2ee461a9c878dd9"
    source = "https://raw.githubusercontent.com/dmechem/ENA_variability_LES_bulk_paper/$revision"
    mkpath(directory)
    for name in ("snd", "lsf_time_varying", "sfc_time_varying")
        path = joinpath(directory, name)
        if !isfile(path)
            temporary = path * ".download"
            Downloads.download("$source/$name", temporary; timeout=30)
            mv(temporary, path)
        end
    end
    return directory
end

function read_sam_profiles(path, columns)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    times = Float64[]
    pressures = Float64[]
    profiles = Matrix{Float64}[]
    line = 2
    while line ≤ length(lines)
        header = split(replace(lines[line], ',' => ' '))
        push!(times, parse(Float64, header[1]))
        count = parse(Int, header[2])
        push!(pressures, 100parse(Float64, header[3]))
        profile = reduce(vcat, [permutedims(parse.(Float64, split(row)[1:columns]))
                               for row in lines[line+1:line+count]])
        all(isfinite, profile) || error("Nonfinite data in $path")
        push!(profiles, profile)
        line += count + 1
    end
    issorted(times) && all(diff(times) .> 0) || error("Invalid SAM times in $path")
    return (; times, pressures, profiles)
end

# LASSO supplies height-coordinate profiles with separate geostrophic winds.
# Its original SST-driven surface scheme is replaced here by the same prescribed
# flux scheme as Covert. H and LE come from the archived sfc tables; missing
# stress (-999.9) uses the shared Covert stress, never the missing-value sentinel.
function lasso_ena_case(name; directory=joinpath(@__DIR__, "..", "data", "ena", "lasso", name),
                        common=covert_ena_case())
    name in ("era5", "era5s1n0", "merra2", "merra2s1n0") ||
        throw(ArgumentError("Unknown LASSO forcing $name."))
    data = read_sam_profiles(joinpath(directory, "lsf"), 9)
    times = (data.times .- 199.25) .* 86400
    height = first(data.profiles)[:, 1]
    all(raw -> raw[:, 1] == height, data.profiles) || error("Changing LASSO height coordinates.")
    names = (:temperature_tendency, :moisture_tendency, :u, :v, :subsidence)
    columns = (3, 4, 8, 9, 7)
    profiles = NamedTuple{names}(Tuple(TimeHeightProfile(height, times,
                   hcat((raw[:, column] for raw in data.profiles)...)) for column in columns))
    raw = readdlm(joinpath(directory, "sfc"), Float64; skipstart=1)
    surface_times = (raw[:, 1] .- 199.25) .* 86400
    surface_names = (:temperature, :sensible_heat_flux, :latent_heat_flux)
    surface = NamedTuple{surface_names}(Tuple(TimeHeightProfile([0.0, 1.0], surface_times,
                       repeat(permutedims(raw[:, column]), 2)) for column in 2:4))
    all(raw[:, 5] .== -999.9) || error("Review LASSO stress convention before importing.")
    surface = merge(surface, (; kinematic_stress=common.surface.kinematic_stress))
    source = "https://doi.org/10.5439/2572661\n" *
             "enalasso_samin_20170718$(name)d25x100_morr-aer2-flxsstC1.m0.20170718.000000.tar.v0"
    return ENACase("lasso_" * name, Date(2017, 7, 18), source, profiles, surface)
end

# Integrate the hypsometric equation on the input sounding, not on the LES grid.
# SAM's pressure-coordinate files contain a duplicate level at the inversion;
# retain the first value there, rather than introduce a zero interpolation interval.
function sam_sounding(path)
    data = read_sam_profiles(path, 6)
    raw = first(data.profiles)
    indices = unique(i -> raw[i, 2], axes(raw, 1))
    raw = raw[indices, :]
    pressure = 100raw[:, 2]
    θ = raw[:, 3]
    q = raw[:, 4] / 1000
    # The Covert idealized sounding specifies total water, interpreted here as
    # specific humidity (the Boussinesq SAM convention), as in the published case.
    temperature = θ .* (pressure ./ 1e5) .^ (287.04 / 1004.0)
    virtual_temperature = temperature .* (1 .+ 0.608q)
    height = zeros(length(pressure))
    for n in 2:length(height)
        height[n] = height[n-1] + 287.04 / 9.81 *
                    (virtual_temperature[n-1] + virtual_temperature[n]) / 2 * log(pressure[n-1] / pressure[n])
    end
    height .-= linear_profile(reverse(pressure), reverse(height), first(data.pressures))
    return (; height, pressure, θ, q, u=raw[:, 5], v=raw[:, 6], surface_pressure=first(data.pressures))
end

function linear_profile(coordinates, values, x)
    n = clamp(searchsortedlast(coordinates, x), 1, length(coordinates)-1)
    fraction = clamp((x - coordinates[n]) / (coordinates[n+1] - coordinates[n]), 0, 1)
    return (1 - fraction) * values[n] + fraction * values[n+1]
end

function covert_ena_case(directory=download_covert_ena())
    sounding = sam_sounding(joinpath(directory, "snd"))
    data = read_sam_profiles(joinpath(directory, "lsf_time_varying"), 7)
    # The clock starts at 06 UTC on day 199 of 2017.
    times = (data.times .- 199.25) .* 86400
    height = sounding.height
    names = (:temperature_tendency, :moisture_tendency, :u, :v, :subsidence)
    arrays = NamedTuple{names}(Tuple(zeros(length(height), length(times)) for name in names))
    for (n, raw) in enumerate(data.profiles)
        pressure = 100raw[:, 2]
        for (column, name) in enumerate(names)
            arrays[name][:, n] .= linear_profile.(Ref(reverse(pressure)), Ref(reverse(raw[:, column+2])), sounding.pressure)
        end
    end
    profiles = map(values -> TimeHeightProfile(height, times, values), arrays)
    raw_surface = readdlm(joinpath(directory, "sfc_time_varying"), Float64; skipstart=1)
    surface_times = (raw_surface[:, 1] .- 199.25) .* 86400
    surface_names = (:temperature, :sensible_heat_flux, :latent_heat_flux, :kinematic_stress)
    surface = NamedTuple{surface_names}(Tuple(TimeHeightProfile([0.0, 1.0], surface_times,
                    repeat(permutedims(raw_surface[:, n+1]), 2)) for n in 1:4))
    return ENACase("covert_20170718", Date(2017, 7, 18),
                   "https://github.com/dmechem/ENA_variability_LES_bulk_paper/tree/a403498ff44045b531af5dbfe2ee461a9c878dd9",
                   profiles, surface)
end
