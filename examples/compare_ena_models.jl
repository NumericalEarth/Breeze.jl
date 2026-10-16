# # Breeze and archived SAM simulations of 18 July 2017
#
# Run with `julia --project=examples examples/compare_ena_models.jl` after the
# Breeze experiments and ARM downloads. The four SAM `morr-aer2-flxsst` statistics
# files come from https://doi.org/10.5439/2572635. Download them with
# `examples/download_lasso_ena.jl <ftp-directory> --statistics`.
# A variable-only `sam_comparison.nc` subset is also accepted; it must preserve
# the original coordinates, variables, units, and simulation metadata.
#
# Both models use the same named forcing sources, but their configurations differ:
# Breeze starts at 06 UTC with the shared Covert sounding, P3, a 6.4 km domain,
# prescribed surface fluxes, no wind nudging, and mean-column RRTMGP radiation.
# Archived SAM starts at midnight with its own sounding, Morrison microphysics,
# a larger domain, interactive surface fluxes, and wind nudging. Consequently,
# these comparisons assess complete configurations, not isolated model components.

include("compare_ena_results.jl")

function ena_sam_results(directory, forcing)
    full_path = joinpath(directory, forcing, "sam_statistics.nc")
    subset_path = joinpath(directory, forcing, "sam_comparison.nc")
    path = isfile(full_path) ? full_path : subset_path
    isfile(path) || error("Missing SAM statistics for $forcing; download the samstat file first.")
    return NCDataset(path) do dataset
        expected = "20170718$(forcing)d25x100_morr-aer2-flxsst"
        dataset.attrib["sim_name"] == expected || error("Unexpected SAM simulation in $path.")
        for (name, units) in (("time", "day"), ("z", "m"), ("CWP", "g/m2"),
                              ("RWP", "g/m2"), ("TABS", "K"), ("QV", "g/kg"),
                              ("U", "m/s"), ("V", "m/s"))
            strip(dataset[name].attrib["units"]) == units || error("Unexpected units for SAM $name.")
        end
        ## SAM stores Julian day (1 January = day 1). The times are already
        ## centers of the two-minute averaging intervals, beginning at 00:01 UTC.
        day = dayofyear(Date(2017, 7, 18))
        time = (Float64.(dataset["time"][:]) .- day) .* 24
        all(0 .< time .< 24) || error("SAM times do not fall on 18 July 2017.")
        all(diff(time) .> 0) || error("SAM times must increase strictly.")
        height = Float64.(dataset["z"][:])
        ## Breeze's saved LWP includes both cloud and rain; use CWP + RWP here.
        cloud = Float64.(coalesce.(dataset["CWP"][:], NaN)) ./ 1000
        rain = Float64.(coalesce.(dataset["RWP"][:], NaN)) ./ 1000
        profiles = map(("TABS", "QV", "U", "V")) do name
            dimnames(dataset[name]) == ("z", "time") || error("Unexpected dimensions for SAM $name.")
            values = Float64.(coalesce.(dataset[name][:, :], NaN))
            return name == "QV" ? values ./ 1000 : values
        end
        return (; time, height, water=cloud+rain, profiles, path)
    end
end

function ena_observed_sounding(directory)
    path = joinpath(directory, "enasondewnpnC1.b1.20170718.113800.cdf")
    return NCDataset(path) do dataset
        height = Float64.(coalesce.(dataset["alt"][:], NaN))
        ## The radiosonde reports vapor mixing ratio per dry-air mass (g/kg).
        ## Compare temperature and winds directly; do not mix humidity conventions.
        variables = ("tdry", "u_wind", "v_wind")
        profiles = map(zip(variables, (273.15, 0, 0))) do (name, offset)
            values = Float64.(coalesce.(dataset[name][:], NaN)) .+ offset
            good = coalesce.(dataset["qc_" * name][:], 1) .== 0
            values[.!good] .= NaN
            values
        end
        return (; height, profiles)
    end
end

function compare_ena_models(; output_directory=joinpath(@__DIR__, "ena_output"),
                            sam_directory=joinpath(@__DIR__, "data", "ena", "lasso"),
                            observations_directory=joinpath(@__DIR__, "data", "ena", "observations"))
    forcings = ("era5", "era5s1n0", "merra2", "merra2s1n0")
    labels = ("ERA5 · 5°", "ERA5 · 1°", "MERRA-2 · 5°", "MERRA-2 · 1°")
    observed = ena_observed_liquid_water_path(observations_directory)
    sounding = ena_observed_sounding(observations_directory)
    edges = collect(6:0.5:15)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    observed_means = ena_interval_means(observed.time, observed.water, edges)
    analysis = 9 .≤ centers .< 12
    water_figure = Figure(size=(1100, 850))
    profile_figure = Figure(size=(1400, 1100))
    breeze_color, sam_color = Makie.wong_colors()[[1, 2]]
    water_axes = Axis[]
    profile_axes = [Axis(profile_figure[row+1, column];
                         xlabel=("Temperature (K)", "Water vapor (kg kg⁻¹)",
                                 "Zonal velocity (m s⁻¹)", "Meridional velocity (m s⁻¹)")[column],
                         ylabel=column == 1 ? "Height (m)" : "",
                         title=column == 1 ? labels[row] : "") for row in 1:4, column in 1:4]
    Label(water_figure[0, 1:2], "18 July 2017 · Breeze P3 and archived LASSO SAM", fontsize=22)
    Label(profile_figure[0, 1:4], "11:30–12:00 UTC domain means and the 11:38 UTC radiosonde", fontsize=21)
    metric_path = joinpath(output_directory, "model_comparison_metrics.csv")
    paired_path = joinpath(output_directory, "breeze_sam_differences.csv")
    open(metric_path, "w") do metrics
        open(paired_path, "w") do paired
            println(metrics, "forcing,model,analysis_start_UTC,analysis_end_UTC,matched_30min_bins,lwp_mean_kg_m2,observed_mean_kg_m2,lwp_bias_kg_m2,lwp_rmse_kg_m2")
            println(paired, "forcing,analysis_start_UTC,analysis_end_UTC,matched_30min_bins,breeze_minus_sam_bias_kg_m2,breeze_minus_sam_rmse_kg_m2")
            for (index, forcing) in enumerate(forcings)
                sam = ena_sam_results(sam_directory, forcing)
                directory = joinpath(output_directory, "lasso_" * forcing)
                data = readdlm(joinpath(directory, "timeseries.csv"), ',', Float64; skipstart=1)
                breeze_time = 6 .+ data[:, 1] ./ 3600
                breeze_water = data[:, 2]
                maximum(breeze_time) ≥ 15 || error("Breeze run $forcing is incomplete.")
                sam_means = ena_interval_means(sam.time, sam.water, edges)
                breeze_means = ena_interval_means(breeze_time, breeze_water, edges)
                selected = analysis .& isfinite.(observed_means) .& isfinite.(sam_means) .& isfinite.(breeze_means)
                count(selected) == 6 || error("Expected all six matched 09–12 UTC bins for $forcing.")
                for (model, means) in (("Breeze_P3", breeze_means), ("SAM_Morrison", sam_means))
                    difference = means[selected] .- observed_means[selected]
                    println(metrics, join((forcing, model, 9, 12, count(selected), mean(means[selected]),
                                          mean(observed_means[selected]), mean(difference),
                                          sqrt(mean(abs2, difference))), ','))
                end
                difference = breeze_means[selected] .- sam_means[selected]
                println(paired, join((forcing, 9, 12, count(selected), mean(difference),
                                     sqrt(mean(abs2, difference))), ','))
                row, column = divrem(index-1, 2) .+ 1
                axis = Axis(water_figure[row, column]; title=labels[index],
                            xlabel="UTC hour", ylabel="Liquid water path (kg m⁻²)")
                push!(water_axes, axis)
                vspan!(axis, 9, 12; color=(:gray, 0.09))
                lines!(axis, breeze_time, breeze_water; color=breeze_color, label="Breeze · P3", linewidth=2)
                visible = 6 .≤ sam.time .≤ 15
                lines!(axis, sam.time[visible], sam.water[visible]; color=sam_color,
                       label="SAM · Morrison", linewidth=2)
                scatterlines!(axis, centers, observed_means; color=:black, linewidth=2,
                              markersize=6, label="ARM MWRRET · 30 min")
                xlims!(axis, 6, 15)
                ylims!(axis, 0, 0.6)
                profiles = readdlm(joinpath(directory, "profiles.csv"), ',', Float64; skipstart=1)
                profile_time = 6 .+ profiles[:, 1] ./ 3600
                breeze_selection = 11.5 .≤ profile_time .< 12
                breeze_height = sort(unique(profiles[breeze_selection, 2]))
                sam_selection = 11.5 .≤ sam.time .< 12
                for column in 1:4
                    ax = profile_axes[index, column]
                    breeze_values = [mean(profiles[breeze_selection .& (profiles[:, 2] .== z), column+2])
                                     for z in breeze_height]
                    sam_values = vec(mean(sam.profiles[column][:, sam_selection]; dims=2))
                    lines!(ax, breeze_values, breeze_height; color=breeze_color, linewidth=2)
                    visible_height = 0 .≤ sam.height .≤ 3000
                    lines!(ax, sam_values[visible_height], sam.height[visible_height]; color=sam_color, linewidth=2)
                    if column != 2
                        observed_column = column == 1 ? 1 : column-1
                        values = sounding.profiles[observed_column]
                        good = isfinite.(values) .& isfinite.(sounding.height) .& (0 .≤ sounding.height .≤ 3000)
                        lines!(ax, values[good], sounding.height[good]; color=:black, linewidth=2)
                    end
                    ylims!(ax, 0, 3000)
                end
            end
        end
    end
    Legend(water_figure[3, 1:2], first(water_axes); orientation=:horizontal, framevisible=false)
    Label(water_figure[4, 1:2], "Shading: 09–12 UTC scoring window. LWP includes cloud + rain.\n" *
          "Same forcing sources; initial state, domain, microphysics, radiation, nudging, and surface treatment differ.", fontsize=13)
    legend_elements = [LineElement(color=color, linewidth=2) for color in (breeze_color, sam_color, :black)]
    Legend(profile_figure[1, 1:4], legend_elements, ["Breeze · P3", "SAM · Morrison", "ARM sonde · T, u, v"];
           orientation=:horizontal, framevisible=false)
    Label(profile_figure[6, 1:4], "Same forcing sources, different model configurations. Point sounding versus domain means.\n" *
          "The sounding also underlies Breeze's idealized initialization; it is not independent validation of that initial state.", fontsize=13)
    save(joinpath(output_directory, "ena_breeze_sam_lwp.png"), water_figure)
    save(joinpath(output_directory, "ena_breeze_sam_profiles.png"), profile_figure)
    return (; water_figure, profile_figure, metric_path, paired_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_ena_models()
end
