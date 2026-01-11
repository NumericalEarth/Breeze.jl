averages_filename = "rico.jld2"
θt = FieldTimeSeries(averages_filename, "θ")
qᵛt = FieldTimeSeries(averages_filename, "qᵛ")
qᶜˡt = FieldTimeSeries(averages_filename, "qᶜˡ")
qʳt = FieldTimeSeries(averages_filename, "qʳ")
#qᵗt = FieldTimeSeries(averages_filename, "qᵗ")
ut = FieldTimeSeries(averages_filename, "u")
vt = FieldTimeSeries(averages_filename, "v")

fig = Figure(size=(900, 800), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵗ (kg/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axqlr = Axis(fig[2, 2], xlabel="qᶜˡ, qʳ (kg/kg)", ylabel="z (m)")

times = θt.times
Nt = length(times)

# Plot initial condition (t=0)
n_init = 1
lines!(axθ, θt[n_init], color=:gray, linestyle=:dash, label="Initial")
lines!(axq, qᵛt[n_init], color=:gray, linestyle=:dash)
lines!(axuv, ut[n_init], color=:gray, linestyle=:solid) # u
lines!(axuv, vt[n_init], color=:gray, linestyle=:dash)  # v
lines!(axqlr, qᶜˡt[n_init], color=:gray, linestyle=:solid) # qcl
lines!(axqlr, qʳt[n_init], color=:gray, linestyle=:dash)  # qr

# Compute and plot average of last 4 hours
# Assuming hourly outputs, last 4 hours are the last 4 indices
indices = (Nt-3):Nt
if length(indices) > 0
    θ_avg = sum(θt[n] for n in indices) / length(indices)
    qᵛ_avg = sum(qᵛt[n] for n in indices) / length(indices)
    u_avg = sum(ut[n] for n in indices) / length(indices)
    v_avg = sum(vt[n] for n in indices) / length(indices)
    qᶜˡ_avg = sum(qᶜˡt[n] for n in indices) / length(indices)
    qʳ_avg = sum(qʳt[n] for n in indices) / length(indices)

    color = Makie.wong_colors()[1]
    t_start = round(Int, times[indices[1]] / hour)
    t_end = round(Int, times[indices[end]] / hour)
    label_avg = "Average last 4 hrs ($(t_start)-$(t_end) h)"
    
    lines!(axθ, θ_avg, color=color, label=label_avg)
    lines!(axq, qᵛ_avg, color=color)
    
    # u and v
    lines!(axuv, u_avg, color=color, linestyle=:solid, label="u")
    lines!(axuv, v_avg, color=color, linestyle=:dash, label="v")
    
    # qcl and qr
    lines!(axqlr, qᶜˡ_avg, color=color, linestyle=:solid, label="qᶜˡ")
    lines!(axqlr, qʳ_avg, color=color, linestyle=:dash, label="qʳ")
end

# Set axis limits to focus on the boundary layer
for ax in (axθ, axq, axuv, axqlr)
    ylims!(ax, 0, 4000)
end

xlims!(axθ, 296, 318)
xlims!(axq, 0, 18e-3)
xlims!(axuv, -12, 2)
xlims!(axqlr, 0, 2e-5)

# Add legends and annotations
axislegend(axθ, position=:rb)
axislegend(axuv, position=:rb)
axislegend(axqlr, position=:rb)

fig[0, :] = Label(fig, "RICO: Mean profiles", fontsize=18, tellwidth=false)

save("rico_profiles.png", fig)