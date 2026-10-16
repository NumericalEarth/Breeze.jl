# Inspect the sounding and turbulence targets before assigning them loss weights.
# The 2017 lidar file has no QC bit fields: the displayed subcloud/SNR mask is
# provisional. Drizzle and wind-sector screening are still needed for assimilation.
using CairoMakie
include("setups/ena_observations.jl")

function plot_ena_observation_targets(; directory=joinpath(@__DIR__, "ena_output", "optimization"))
    mkpath(directory)
    morning = ena_sounding(filename="enasondewnpnC1.b1.20170718.053100.cdf")
    sounding = ena_sounding()
    lidar = ena_lidar_statistics()
    figure = Figure(size=(1250, 950))
    soundings_layout = figure[1, 1] = GridLayout()
    lidar_layout = figure[2, 1] = GridLayout()
    context_layout = figure[3, 1] = GridLayout()
    labels = ("Temperature (K)", "Specific humidity (kg kg⁻¹)", "Zonal wind (m s⁻¹)", "Meridional wind (m s⁻¹)")
    for variable in 1:4
        axis = Axis(soundings_layout[1, variable]; xlabel=labels[variable], ylabel=variable == 1 ? "Altitude MSL (m)" : "")
        for (data, label, color) in ((morning, "05:31 sounding", :gray55), (sounding, "11:38 sounding", :black))
            selected = data.valid[:, variable] .& (0 .≤ data.height .≤ 3000)
            lines!(axis, data.values[selected, variable], data.height[selected]; label, color)
        end
        ylims!(axis, 0, 3000)
        variable > 1 && hideydecorations!(axis; grid=false)
        variable == 2 && (axis.xticks = [0.006, 0.009, 0.012])
        variable == 1 && axislegend(axis; position=:rt)
    end
    selected = 6 .≤ lidar.time .≤ 15
    variance = ifelse.(lidar.screened, lidar.variance, NaN)
    axis = Axis(lidar_layout[1, 1]; xlabel="18 July 2017, UTC hour", ylabel="Height above site (m)",
                title="Doppler-lidar vertical-velocity variance · provisional subcloud/SNR screening")
    heatmap = heatmap!(axis, lidar.time[selected], lidar.height, permutedims(variance[:, selected]);
                      colormap=:viridis, colorrange=(0, 0.5))
    lines!(axis, lidar.time[selected], lidar.cloud_base[selected]; color=:white, label="Lidar cloud base")
    ylims!(axis, 0, 1200)
    Colorbar(lidar_layout[1, 2], heatmap; label="⟨w′²⟩ (m² s⁻²)")
    profile_axis = Axis(context_layout[1, 1]; xlabel="⟨w′²⟩ (m² s⁻²)", ylabel="Height above site (m)",
                        title="11:30 UTC: 11:15–11:45 averaging window")
    index = argmin(abs.(lidar.time .- 11.5))
    lines!(profile_axis, variance[:, index], lidar.height; color=:black)
    ylims!(profile_axis, 0, 1200)
    surface_axis = Axis(context_layout[1, 2]; xlabel="18 July 2017, UTC hour", ylabel="m² s⁻²",
                        title="Island surface measurements (context only)")
    lines!(surface_axis, lidar.time[selected], lidar.surface_variance[selected]; label="⟨w′²⟩")
    lines!(surface_axis, lidar.time[selected], lidar.surface_tke[selected]; label="TKE")
    axislegend(surface_axis; position=:rt)
    Label(figure[4, 1], "Lidar: 30-minute windows every 10 minutes; 30 m range gates.\n" *
          "Resolved LES variance needs matched sampling and averaging; radar Doppler velocity includes particle fall speed.", fontsize=14)
    save(joinpath(directory, "ena_observation_targets.png"), figure)
    return figure
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_ena_observation_targets()
end
