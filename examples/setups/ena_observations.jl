# Observation operators shared by the ENA inverse experiment and its figures.
using NCDatasets
using Dates
using Statistics
using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, PlanarLiquidSurface,
                              dry_air_gas_constant, vapor_gas_constant

ena_observations_directory() = joinpath(@__DIR__, "..", "data", "ena", "observations")

function ena_sounding(; filename="enasondewnpnC1.b1.20170718.113800.cdf",
                       directory=ena_observations_directory())
    NCDataset(joinpath(directory, filename)) do dataset
        read(name) = Float64.(coalesce.(Array(dataset[name]), NaN))
        good(name) = coalesce.(dataset["qc_" * name][:], 1) .== 0
        temperature = read("tdry") .+ 273.15
        dewpoint = read("dp") .+ 273.15
        pressure = 100 .* read("pres")
        constants = ThermodynamicConstants(Float64)
        ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
        vapor_pressure = saturation_vapor_pressure.(dewpoint, Ref(constants), Ref(PlanarLiquidSurface()))
        # Specific humidity (vapor / moist air), not dry-air mixing ratio.
        vapor = ε .* vapor_pressure ./ (pressure .- (1 - ε) .* vapor_pressure)
        values = hcat(temperature, vapor, read("u_wind"), read("v_wind"))
        valid = hcat(good("tdry"), good("dp") .& good("pres"), good("u_wind"), good("v_wind"))
        valid .&= isfinite.(values)
        valid[:, 2] .&= (0 .< vapor_pressure .< pressure) .& (0 .≤ vapor .< 1)
        time = Dates.value.(DateTime.(dataset["time"][:]) .- DateTime(2017, 7, 18, 6)) ./ 1000
        return (; height=read("alt"), time, values, valid)
    end
end

# Bin the ascent onto LES cells, retaining each bin's actual sampling time.
# We use separate masks for T, qᵛ, u and v; a missing variable never becomes zero data.
function ena_sounding_target(faces; sounding=ena_sounding(), scales=(0.5, 0.0005, 1.0, 1.0))
    all(diff(faces) .> 0) || throw(ArgumentError("Cell faces must be strictly increasing."))
    length(scales) == 4 && all(isfinite, scales) && all(>(0), scales) ||
        throw(ArgumentError("Provide four finite positive mismatch scales."))
    Nz = length(faces) - 1
    values = zeros(Nz, 4)
    times = zeros(Nz, 4)
    weights = zeros(Nz, 4)
    for k in 1:Nz, variable in 1:4
        selected = (faces[k] .≤ sounding.height .< faces[k+1]) .& sounding.valid[:, variable]
        if any(selected)
            values[k, variable] = mean(sounding.values[selected, variable])
            times[k, variable] = mean(sounding.time[selected])
            weights[k, variable] = 1 / scales[variable]^2
        end
    end
    # Equal weight per observed variable, independently of its number of valid bins.
    for variable in 1:4
        count = sum(weights[:, variable] .> 0)
        count > 0 || throw(ArgumentError("Sounding has no valid values for variable $variable."))
        weights[:, variable] ./= 4count
    end
    return (; values, times, weights, height=(faces[1:end-1] .+ faces[2:end]) ./ 2)
end

function ena_lidar_statistics(; directory=ena_observations_directory())
    filename = "enadlprofwstats4newsC1.c1.20170718.000000.nc"
    NCDataset(joinpath(directory, filename)) do dataset
        read(name) = Float64.(coalesce.(Array(dataset[name]), NaN))
        time = Dates.value.(DateTime.(dataset["time"][:]) .- DateTime(2017, 7, 18)) ./ 3.6e6
        height = read("height")
        variance = read("w_variance") # NCDatasets orders these dimensions (height, time).
        cloud_base = read("dl_cbh")
        # This 2017 product has no qc_* fields. This is a conservative screening
        # mask for plotting, NOT a claim that drizzle/island effects have been removed.
        screened = isfinite.(variance) .& (variance .≥ 0) .& (height .≥ 105)
        screened .&= height .< permutedims(cloud_base .- 60)
        screened .&= read("snr") .≥ dataset["snr_threshold"][]
        return (; time, height, height_msl=height .+ dataset["alt"][], variance, screened,
                skewness=read("w_skewness"), kurtosis=read("w_kurtosis"), cloud_base,
                surface_variance=read("ecor_w_var"), surface_tke=read("ecor_tke"),
                averaging_seconds=dataset["averaging_time"][], filename)
    end
end
