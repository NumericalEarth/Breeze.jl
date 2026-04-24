#####
##### Three-panel animation: anelastic | explicit compressible | substepped
##### compressible. Uses the JLD2 outputs saved by 07 + 08.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")

wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "w")
we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),     "w")
ws_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")

ta, te, ts = wa_ts.times, we_ts.times, ws_ts.times

# Use the common time range
tmax = min(ta[end], te[end], ts[end])
mask_a = findall(t -> t ≤ tmax + 1e-6, ta)
mask_e = findall(t -> t ≤ tmax + 1e-6, te)
mask_s = findall(t -> t ≤ tmax + 1e-6, ts)
Nt = min(length(mask_a), length(mask_e), length(mask_s))
@info "Building 3-panel animation" Nt tmax

grid = wa_ts.grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

vmax = let v = 0.0
    for i in 1:Nt
        v = max(v,
                maximum(abs, interior(wa_ts[mask_a[i]])),
                maximum(abs, interior(we_ts[mask_e[i]])),
                maximum(abs, interior(ws_ts[mask_s[i]])))
    end
    isfinite(v) && v > 0 ? v : 1.0
end
@info "shared color range" vmax

n = Observable(1)

wa_slice = @lift Array(interior(wa_ts[mask_a[$n]]))[:, 1, :]
we_slice = @lift Array(interior(we_ts[mask_e[$n]]))[:, 1, :]
ws_slice = @lift Array(interior(ws_ts[mask_s[$n]]))[:, 1, :]

fig = Figure(size = (1800, 560), fontsize = 15)
title_node = @lift @sprintf("Dry thermal bubble — t = %5.1f s", ta[mask_a[$n]])
fig[0, 1:4] = Label(fig, title_node, fontsize = 20, tellwidth = false)

ax_a = Axis(fig[1, 1]; title = "Anelastic",
            xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
ax_e = Axis(fig[1, 2]; title = "Compressible explicit (Δt = 0.1 s)",
            xlabel = "x (km)", aspect = DataAspect())
ax_s = Axis(fig[1, 3]; title = "Compressible substepper",
            xlabel = "x (km)", aspect = DataAspect())

hm = heatmap!(ax_a, x_km, z_km, wa_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_e, x_km, z_km, we_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_s, x_km, z_km, ws_slice; colormap = :balance, colorrange = (-vmax, vmax))

Colorbar(fig[1, 4], hm; label = "w (m/s)")

mp4 = joinpath(OUTDIR, "bubble_three_way.mp4")
record(fig, mp4, 1:Nt; framerate = 15) do i
    n[] = i
end
@info "wrote $mp4"

gif = joinpath(OUTDIR, "bubble_three_way.gif")
record(fig, gif, 1:Nt; framerate = 15) do i
    n[] = i
end
@info "wrote $gif"
