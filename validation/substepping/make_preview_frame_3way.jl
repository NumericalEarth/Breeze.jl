#####
##### Three-panel stills at key times: anelastic | explicit | substepper.
#####

using Breeze, Oceananigans, Oceananigans.Units, CairoMakie, Printf

OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")
wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "w")
we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),     "w")
ws_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")

grid  = wa_ts.grid
x_km  = collect(xnodes(grid, Center())) ./ 1e3
z_km  = collect(znodes(grid, Face()))   ./ 1e3

tmax = min(wa_ts.times[end], we_ts.times[end], ws_ts.times[end])

for t_target in (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0)
    t_target > tmax && break
    ia = argmin(abs.(wa_ts.times .- t_target))
    ie = argmin(abs.(we_ts.times .- t_target))
    is = argmin(abs.(ws_ts.times .- t_target))
    wa = Array(interior(wa_ts[ia]))[:, 1, :]
    we = Array(interior(we_ts[ie]))[:, 1, :]
    ws = Array(interior(ws_ts[is]))[:, 1, :]
    vmax = max(maximum(abs, wa), maximum(abs, we), maximum(abs, ws))
    vmax = isfinite(vmax) && vmax > 0 ? vmax : 1.0

    fig = Figure(size = (1800, 560), fontsize = 15)
    fig[0, 1:4] = Label(fig, @sprintf("Dry thermal bubble — t = %4.0f s", we_ts.times[ie]),
                        fontsize = 20, tellwidth = false)
    ax_a = Axis(fig[1, 1]; title = "Anelastic",              xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
    ax_e = Axis(fig[1, 2]; title = "Compressible explicit",  xlabel = "x (km)", aspect = DataAspect())
    ax_s = Axis(fig[1, 3]; title = "Compressible substepper",xlabel = "x (km)", aspect = DataAspect())
    hm = heatmap!(ax_a, x_km, z_km, wa; colormap = :balance, colorrange = (-vmax, vmax))
    heatmap!(     ax_e, x_km, z_km, we; colormap = :balance, colorrange = (-vmax, vmax))
    heatmap!(     ax_s, x_km, z_km, ws; colormap = :balance, colorrange = (-vmax, vmax))
    Colorbar(fig[1, 4], hm; label = "w (m/s)")

    path = joinpath(OUTDIR, @sprintf("frame3_t%04d.png", round(Int, we_ts.times[ie])))
    save(path, fig)
    @info "wrote" path
end
