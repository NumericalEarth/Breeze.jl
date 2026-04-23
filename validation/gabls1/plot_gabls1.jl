# # GABLS1: Comparison with intercomparison data
#
# This script loads Breeze output from gabls1.jl and compares horizontally-averaged
# profiles at hour 9 against intercomparison data from 8 LES groups
# (Beare et al. 2006, Table 1).

using CairoMakie
using Oceananigans
using Printf

# ## Load Breeze output

filename = joinpath(@__DIR__, "gabls1_averages.jld2")

if !isfile(filename)
    error("Output file not found: $filename\n" *
          "Run gabls1.jl first to generate the simulation output.")
end

uts  = FieldTimeSeries(filename, "u")
vts  = FieldTimeSeries(filename, "v")
θts  = FieldTimeSeries(filename, "θ")
w²ts = FieldTimeSeries(filename, "w²")
uwts = FieldTimeSeries(filename, "uw")
vwts = FieldTimeSeries(filename, "vw")
wθts = FieldTimeSeries(filename, "wθ")

times = uts.times
Nt = length(times)

# Extract the grid for height coordinates
grid = uts.grid
z = Oceananigans.Grids.znodes(grid, Center())

@info "Loaded $(Nt) time snapshots, times = $(times ./ 3600) hours"
@info "Last snapshot is the $(Int(times[end]/3600))-hour average"

# ## Load reference data
#
# Intercomparison data from the ERF validation repository.
# Data files contain 4 blocks of 128 values: heights, U, V, θ.

function read_gabls_data(filepath)
    lines = readlines(filepath)
    header = strip(lines[1])
    N = parse(Int, strip(lines[2]))  # N = 128 (number of values per variable)

    values = Float64[]
    for line in lines[3:end]
        for s in split(strip(line))
            push!(values, parse(Float64, s))
        end
    end

    # Block layout: first N values = z, next N = U, next N = V, next N = θ
    # Julia column-major reshape gives columns in order
    data = reshape(values, N, :)  # (128, 4)

    return (; z    = data[:, 1],
              U    = data[:, 2],
              V    = data[:, 3],
              θ    = data[:, 4],
              name = header)
end

# Load all reference groups
reference_data_dir = joinpath(@__DIR__, "reference_data")
groups = ["NCAR", "LLNL", "CORA", "CSU", "IMUK", "MO", "NERSC", "UIB"]

reference = Dict{String, NamedTuple}()
for g in groups
    filepath = joinpath(reference_data_dir, "$(g)_A9_128.dat")
    if isfile(filepath)
        reference[g] = read_gabls_data(filepath)
        @info "Loaded reference data: $(g) — $(reference[g].name)"
    else
        @warn "Reference file not found: $filepath"
    end
end

# ## Extract Breeze profiles at hour 9
#
# The last output snapshot is the average over hours 8-9.

u_breeze = interior(uts[Nt], 1, 1, :)
v_breeze = interior(vts[Nt], 1, 1, :)
θ_breeze = interior(θts[Nt], 1, 1, :)
w²_breeze = interior(w²ts[Nt], 1, 1, :)
uw_breeze = interior(uwts[Nt], 1, 1, :)
vw_breeze = interior(vwts[Nt], 1, 1, :)
wθ_breeze = interior(wθts[Nt], 1, 1, :)

wind_speed_breeze = @. sqrt(u_breeze^2 + v_breeze^2)

# ## Figure 1: Mean profiles comparison
#
# Compare wind speed, v-component, and potential temperature profiles.

fig1 = Figure(size=(1200, 500), fontsize=14)

ax_ws = Axis(fig1[1, 1], xlabel="Wind speed (m/s)", ylabel="z (m)",
             title="Horizontal wind speed")
ax_v  = Axis(fig1[1, 2], xlabel="V (m/s)", ylabel="z (m)",
             title="Meridional wind")
ax_θ  = Axis(fig1[1, 3], xlabel="θ (K)", ylabel="z (m)",
             title="Potential temperature")

# Plot reference data (thin gray lines)
for (i, g) in enumerate(groups)
    haskey(reference, g) || continue
    data = reference[g]
    ws_ref = @. sqrt(data.U^2 + data.V^2)
    label = i == 1 ? "Intercomparison" : nothing
    lines!(ax_ws, ws_ref, data.z, color=(:gray, 0.5), linewidth=1; label)
    lines!(ax_v,  data.V,  data.z, color=(:gray, 0.5), linewidth=1)
    lines!(ax_θ,  data.θ,  data.z, color=(:gray, 0.5), linewidth=1)
end

# Plot Breeze result (thick blue line)
lines!(ax_ws, wind_speed_breeze, z, color=:dodgerblue, linewidth=3, label="Breeze (ILES)")
lines!(ax_v,  v_breeze,          z, color=:dodgerblue, linewidth=3)
lines!(ax_θ,  θ_breeze,          z, color=:dodgerblue, linewidth=3)

# Set axis limits to focus on the boundary layer
for ax in (ax_ws, ax_v, ax_θ)
    ylims!(ax, 0, 350)
end

xlims!(ax_ws, 0, 12)
xlims!(ax_v, -3.5, 1)
xlims!(ax_θ, 262, 268)

axislegend(ax_ws, position=:rt)

fig1[0, :] = Label(fig1, "GABLS1: Mean profiles at hour 9 (Breeze vs intercomparison)",
                    fontsize=18, tellwidth=false)

save(joinpath(@__DIR__, "gabls1_profiles.png"), fig1)
@info "Saved gabls1_profiles.png"

fig1

# ## Figure 2: Log-law comparison
#
# Plot wind speed vs log(z) to demonstrate the near-surface log-law mismatch.
# In MOST, the wind profile should be linear in log(z). The ILES solution
# overshoots because the first grid point does not "see" the logarithmic layer.

fig2 = Figure(size=(600, 500), fontsize=14)

ax_log = Axis(fig2[1, 1],
              xlabel = "Wind speed (m/s)",
              ylabel = "z (m)",
              yscale = log10,
              title = "Wind speed vs height (log scale)")

# Reference data
for g in groups
    haskey(reference, g) || continue
    data = reference[g]
    ws_ref = @. sqrt(data.U^2 + data.V^2)
    kmax = findfirst(data.z .> 200)
    lines!(ax_log, ws_ref[1:kmax], data.z[1:kmax],
           color=(:gray, 0.5), linewidth=1)
end

# Breeze result
kmax_b = findfirst(z .> 200)
lines!(ax_log, wind_speed_breeze[1:kmax_b], z[1:kmax_b],
       color=:dodgerblue, linewidth=3, label="Breeze (ILES)")

# Reference log-law profile (Monin-Obukhov neutral)
z₀ = 0.1
u_star_approx = 0.3  # approximate friction velocity
z_log = range(z₀, 200, length=100)
u_log = @. u_star_approx / 0.4 * log(z_log / z₀)
lines!(ax_log, u_log, z_log, color=:black, linewidth=2, linestyle=:dash,
       label=@sprintf("Log law (u★ ≈ %.2f m/s)", u_star_approx))

ylims!(ax_log, 1, 200)
xlims!(ax_log, 0, 12)

axislegend(ax_log, position=:rb)

save(joinpath(@__DIR__, "gabls1_loglaw.png"), fig2)
@info "Saved gabls1_loglaw.png"

fig2

# ## Figure 3: Turbulent flux profiles

fig3 = Figure(size=(1000, 450), fontsize=14)

ax_uw = Axis(fig3[1, 1], xlabel="Momentum flux (m²/s²)", ylabel="z (m)",
             title="Momentum fluxes")
ax_wθ = Axis(fig3[1, 2], xlabel="Heat flux (K m/s)", ylabel="z (m)",
             title="Heat flux w'θ'")
ax_tke = Axis(fig3[1, 3], xlabel="w'² (m²/s²)", ylabel="z (m)",
              title="Vertical velocity variance")

lines!(ax_uw,  uw_breeze, z, color=:dodgerblue, linewidth=2, label="u'w'")
lines!(ax_uw,  vw_breeze, z, color=:firebrick,  linewidth=2, label="v'w'")
lines!(ax_wθ,  wθ_breeze, z, color=:dodgerblue, linewidth=2)
lines!(ax_tke, w²_breeze, z, color=:dodgerblue, linewidth=2)

axislegend(ax_uw, position=:rt)

for ax in (ax_uw, ax_wθ, ax_tke)
    ylims!(ax, 0, 300)
end

fig3[0, :] = Label(fig3, "GABLS1: Turbulent flux profiles at hour 9",
                    fontsize=18, tellwidth=false)

save(joinpath(@__DIR__, "gabls1_fluxes.png"), fig3)
@info "Saved gabls1_fluxes.png"

fig3

# ## Figure 4: Profile evolution
#
# Show the evolution of horizontally-averaged profiles every hour.

fig4 = Figure(size=(1100, 500), fontsize=14)

ax_θe = Axis(fig4[1, 1], xlabel="θ (K)", ylabel="z (m)", title="Potential temperature")
ax_ue = Axis(fig4[1, 2], xlabel="Speed (m/s)", ylabel="z (m)", title="Wind speed")
ax_ve = Axis(fig4[1, 3], xlabel="V (m/s)", ylabel="z (m)", title="Meridional wind")

default_colors = Makie.wong_colors()
colors = [default_colors[mod1(i, length(default_colors))] for i in 1:Nt]

for n in 1:Nt
    u_n = interior(uts[n], 1, 1, :)
    v_n = interior(vts[n], 1, 1, :)
    θ_n = interior(θts[n], 1, 1, :)
    ws_n = @. sqrt(u_n^2 + v_n^2)

    label = "$(Int(times[n]/3600)) hr"

    lines!(ax_θe, θ_n, z,  color=colors[n], linewidth=2, label=label)
    lines!(ax_ue, ws_n, z, color=colors[n], linewidth=2)
    lines!(ax_ve, v_n, z,  color=colors[n], linewidth=2)
end

for ax in (ax_θe, ax_ue, ax_ve)
    ylims!(ax, 0, 350)
end

axislegend(ax_θe, position=:rt)

fig4[0, :] = Label(fig4, "GABLS1: Profile evolution (hourly averages)",
                    fontsize=18, tellwidth=false)

save(joinpath(@__DIR__, "gabls1_evolution.png"), fig4)
@info "Saved gabls1_evolution.png"

fig4
