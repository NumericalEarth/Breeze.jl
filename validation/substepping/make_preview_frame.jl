#####
##### One-frame snapshot from the middle of the bubble simulation for preview.
#####

using Breeze, Oceananigans, Oceananigans.Units, CairoMakie, Printf

OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")
wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "w")
wc_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")

grid  = wa_ts.grid
x_km  = collect(xnodes(grid, Center())) ./ 1e3
z_km  = collect(znodes(grid, Face()))   ./ 1e3

# Find three representative times: 100s (accelerating), 400s (peak rise),
# 900s (near-secondary peak), 1500s (end).
for t_target in (100.0, 400.0, 900.0, 1500.0)
    ia = argmin(abs.(wa_ts.times .- t_target))
    ic = argmin(abs.(wc_ts.times .- t_target))
    wa = Array(interior(wa_ts[ia]))[:, 1, :]
    wc = Array(interior(wc_ts[ic]))[:, 1, :]
    vmax = max(maximum(abs, wa), maximum(abs, wc))
    vmax = isfinite(vmax) && vmax > 0 ? vmax : 1.0

    fig = Figure(size = (1500, 620), fontsize = 15)
    fig[0, 1:3] = Label(fig, @sprintf("Dry thermal bubble — t = %4.0f s", wa_ts.times[ia]),
                        fontsize = 20, tellwidth = false)
    ax_a = Axis(fig[1, 1]; title = "Anelastic",        xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
    ax_c = Axis(fig[1, 2]; title = "Compressible (substepper)", xlabel = "x (km)",           aspect = DataAspect())
    hm_a = heatmap!(ax_a, x_km, z_km, wa; colormap = :balance, colorrange = (-vmax, vmax))
    heatmap!(           ax_c, x_km, z_km, wc; colormap = :balance, colorrange = (-vmax, vmax))
    Colorbar(fig[1, 3], hm_a; label = "w (m/s)")

    path = joinpath(OUTDIR, @sprintf("frame_t%04d.png", round(Int, wa_ts.times[ia])))
    save(path, fig)
    @info "wrote" path
end
