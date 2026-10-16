# Compare July 18 LES output with quality-controlled ARM observations.
# Run after ena_forcing_comparison.jl. All figures are generated with Julia.
using CairoMakie
using NCDatasets
using Dates
using DelimitedFiles
using Statistics
using Printf

function ena_observed_liquid_water_path(directory)
    path = joinpath(directory, "enamwrret2turnC1.c1.20170718.031053.nc")
    return NCDataset(path) do dataset
        time = Dates.value.(DateTime.(dataset["time"][:]) .- DateTime(2017, 7, 18)) ./ 3.6e6
        water = coalesce.(dataset["phys_lwp"][:], NaN) ./ 1000 # g/m² → kg/m²
        quality = coalesce.(dataset["qc_phys_lwp"][:], 1) .== 0
        quality .&= coalesce.(dataset["phys_qc_flag"][:], 1) .== 0
        valid = quality .& isfinite.(water) .& (6 .≤ time .≤ 15)
        return (; time=time[valid], water=water[valid])
    end
end

function ena_interval_means(time, values, edges)
    return map(zip(edges[1:end-1], edges[2:end])) do (left, right)
        selected = (left .≤ time .< right) .& isfinite.(values)
        return any(selected) ? mean(values[selected]) : NaN
    end
end

function compare_ena_results(; output_directory=joinpath(@__DIR__, "ena_output"),
                             observations_directory=joinpath(@__DIR__, "data", "ena", "observations"))
    observations = ena_observed_liquid_water_path(observations_directory)
    edges = collect(6:0.5:15)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    observed = ena_interval_means(observations.time, observations.water, edges)
    figure = Figure(size=(1100, 850))
    axis = Axis(figure[1, 1:3]; xlabel="18 July 2017, UTC hour", ylabel="Liquid water path (kg m⁻²)")
    scatterlines!(axis, centers, observed; color=:black, label="ARM MWRRET (30 min means)", linewidth=2)
    xlims!(axis, 6, 15)
    temperature_axis = Axis(figure[3, 1]; xlabel="Temperature (K)", ylabel="Height (m)", title="11:38 UTC sounding")
    zonal_axis = Axis(figure[3, 2]; xlabel="Zonal velocity (m s⁻¹)", ylabel="Height (m)")
    meridional_axis = Axis(figure[3, 3]; xlabel="Meridional velocity (m s⁻¹)", ylabel="Height (m)")
    axes = (temperature_axis, zonal_axis, meridional_axis)
    NCDataset(joinpath(observations_directory, "enasondewnpnC1.b1.20170718.113800.cdf")) do dataset
        height = coalesce.(dataset["alt"][:], NaN)
        for (ax, name, offset) in zip(axes, ("tdry", "u_wind", "v_wind"), (273.15, 0, 0))
            values = coalesce.(dataset[name][:], NaN) .+ offset
            quality = coalesce.(dataset["qc_"*name][:], 1) .== 0
            valid = quality .& isfinite.(height) .& isfinite.(values) .& (0 .≤ height .≤ 3000)
            lines!(ax, values[valid], height[valid]; color=:black, linewidth=2)
            ylims!(ax, 0, 3000)
        end
    end
    directories = filter(path -> isfile(joinpath(path, "timeseries.csv")), readdir(output_directory; join=true))
    colors = Makie.wong_colors()
    labels = Dict("covert_20170718" => "Covert / VARANAL", "lasso_era5" => "LASSO ERA5 · 5°",
                  "lasso_era5s1n0" => "LASSO ERA5 · 1°", "lasso_merra2" => "LASSO MERRA-2 · 5°",
                  "lasso_merra2s1n0" => "LASSO MERRA-2 · 1°")
    metrics = joinpath(output_directory, "comparison_metrics.csv")
    open(metrics, "w") do io
        println(io, "case,analysis_start_UTC,analysis_end_UTC,matched_30min_bins,lwp_bias_kg_m2,lwp_rmse_kg_m2")
        for (index, directory) in enumerate(directories)
            name = basename(directory)
            color = colors[mod1(index, length(colors))]
            data = readdlm(joinpath(directory, "timeseries.csv"), ',', Float64; skipstart=1)
            isempty(data) && continue
            time = 6 .+ data[:, 1] ./ 3600
            modeled = ena_interval_means(time, data[:, 2], edges)
            complete = edges[2:end] .≤ maximum(time)
            selected = complete .& (9 .≤ centers .< 12) .& isfinite.(observed) .& isfinite.(modeled)
            difference = modeled[selected] .- observed[selected]
            bias = isempty(difference) ? NaN : mean(difference)
            rmse = isempty(difference) ? NaN : sqrt(mean(abs2, difference))
            println(io, join((name, 9, 12, count(selected), bias, rmse), ','))
            lines!(axis, time, data[:, 2]; color, label=get(labels, name, name))
            profiles = readdlm(joinpath(directory, "profiles.csv"), ',', Float64; skipstart=1)
            target = (11 + 38/60 - 6) * 3600
            if maximum(profiles[:, 1]) ≥ target
                nearest = profiles[argmin(abs.(profiles[:, 1] .- target)), 1]
                selected_profile = profiles[:, 1] .== nearest
                for (ax, column) in zip(axes, (3, 5, 6))
                    lines!(ax, profiles[selected_profile, column], profiles[selected_profile, 2]; color)
                end
            end
        end
    end
    Legend(figure[2, 1:3], axis; orientation=:horizontal, nbanks=2, framevisible=false)
    Label(figure[4, 1:3], "P3 LES · shared initial state · mean-column RRTMGP radiation\n" *
          "Point observations versus domain means; coarse-grid experiment, not a converged reproduction.", fontsize=13)
    save(joinpath(output_directory, "ena_observation_comparison.png"), figure)
    return figure
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_ena_results()
end
