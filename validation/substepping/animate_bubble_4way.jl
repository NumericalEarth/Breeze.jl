#####
##### 4-panel animation: anelastic | explicit | substepper fw=0.8 (ring) |
##### substepper fw=0.9 (ring-free). Uses the outputs already on disk from
##### runs 07, 08, 11.
#####
##### The fw=0.9 run only covers 0–420 s, so the whole animation is trimmed
##### to that range.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")
const SWEEP  = joinpath(OUTDIR, "knob_sweep")

wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),       "w")
we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),        "w")
ws_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"),    "w")  # wizard fw=0.8
w9_ts = FieldTimeSeries(joinpath(SWEEP,  "Press0.5_fw0.9.jld2"),  "w")

tmax = min(wa_ts.times[end], we_ts.times[end], ws_ts.times[end], w9_ts.times[end])

mask_a = findall(<=(tmax + 1e-6), wa_ts.times)
mask_e = findall(<=(tmax + 1e-6), we_ts.times)
mask_s = findall(<=(tmax + 1e-6), ws_ts.times)
mask_9 = findall(<=(tmax + 1e-6), w9_ts.times)

Nt = min(length(mask_a), length(mask_e), length(mask_s), length(mask_9))
@info "Building 4-panel animation" Nt tmax

grid = wa_ts.grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

vmax = let v = 0.0
    for i in 1:Nt
        v = max(v,
                maximum(abs, interior(wa_ts[mask_a[i]])),
                maximum(abs, interior(we_ts[mask_e[i]])),
                maximum(abs, interior(ws_ts[mask_s[i]])),
                maximum(abs, interior(w9_ts[mask_9[i]])))
    end
    isfinite(v) && v > 0 ? v : 1.0
end
@info "shared color range" vmax

n = Observable(1)

wa_slice = @lift Array(interior(wa_ts[mask_a[$n]]))[:, 1, :]
we_slice = @lift Array(interior(we_ts[mask_e[$n]]))[:, 1, :]
ws_slice = @lift Array(interior(ws_ts[mask_s[$n]]))[:, 1, :]
w9_slice = @lift Array(interior(w9_ts[mask_9[$n]]))[:, 1, :]

fig = Figure(size = (2000, 560), fontsize = 15)
title_node = @lift @sprintf("Dry thermal bubble — t = %5.1f s", wa_ts.times[mask_a[$n]])
fig[0, 1:5] = Label(fig, title_node, fontsize = 22, tellwidth = false)

ax_a = Axis(fig[1, 1]; title = "Anelastic",                          xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
ax_e = Axis(fig[1, 2]; title = "Compressible explicit (Δt = 0.1 s)",  xlabel = "x (km)",                    aspect = DataAspect())
ax_s = Axis(fig[1, 3]; title = "Substepper fw = 0.8 (has ring)",      xlabel = "x (km)",                    aspect = DataAspect())
ax_9 = Axis(fig[1, 4]; title = "Substepper fw = 0.9 (ring-free)",     xlabel = "x (km)",                    aspect = DataAspect())

hm = heatmap!(ax_a, x_km, z_km, wa_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_e, x_km, z_km, we_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_s, x_km, z_km, ws_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_9, x_km, z_km, w9_slice; colormap = :balance, colorrange = (-vmax, vmax))

Colorbar(fig[1, 5], hm; label = "w (m/s)")

mp4 = joinpath(OUTDIR, "bubble_four_way.mp4")
record(fig, mp4, 1:Nt; framerate = 12) do i
    n[] = i
end
@info "wrote $mp4"

gif = joinpath(OUTDIR, "bubble_four_way.gif")
record(fig, gif, 1:Nt; framerate = 12) do i
    n[] = i
end
@info "wrote $gif"
