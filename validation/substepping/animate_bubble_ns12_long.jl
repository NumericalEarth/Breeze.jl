#####
##### Three-panel animation for the Ns=12 long bubble run.
##### anelastic | explicit-compressible (ground truth) | Ns=12 substepper
#####
##### Substepped run NaNs early; this script handles partial timeseries by
##### masking non-finite values. Movie spans the common time range.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_ns12_long")

wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),  "w")
we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),   "w")
ws_ts = FieldTimeSeries(joinpath(OUTDIR, "substepped.jld2"), "w")

ta, te, tsub = wa_ts.times, we_ts.times, ws_ts.times

tmax = min(ta[end], te[end], tsub[end])
mask_a = findall(t -> t ≤ tmax + 1e-6, ta)
mask_e = findall(t -> t ≤ tmax + 1e-6, te)
mask_s = findall(t -> t ≤ tmax + 1e-6, tsub)
Nt = min(length(mask_a), length(mask_e), length(mask_s))
@info "Building 3-panel animation" Nt tmax

grid = wa_ts.grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

function _safe_max_abs(field)
    a = Array(interior(field))
    m = 0.0
    @inbounds for x in a
        isfinite(x) && (m = max(m, abs(Float64(x))))
    end
    return m
end
function _slice_safe(ts, idx)
    a = Array(interior(ts[idx]))[:, 1, :]
    replace!(x -> isfinite(x) ? x : 0.0, a)
    return a
end

vmax = let v = 0.0
    for i in 1:Nt
        v = max(v,
                _safe_max_abs(wa_ts[mask_a[i]]),
                _safe_max_abs(we_ts[mask_e[i]]),
                _safe_max_abs(ws_ts[mask_s[i]]))
    end
    v > 0 ? v : 1.0
end
@info "shared color range" vmax

n = Observable(1)
wa_slice = @lift _slice_safe(wa_ts, mask_a[$n])
we_slice = @lift _slice_safe(we_ts, mask_e[$n])
ws_slice = @lift _slice_safe(ws_ts, mask_s[$n])

fig = Figure(size = (1800, 560), fontsize = 15)
title_node = @lift @sprintf("Dry thermal bubble (Ns=12 long run) — t = %5.1f s",
                            ta[mask_a[$n]])
fig[0, 1:4] = Label(fig, title_node, fontsize = 20, tellwidth = false)

ax_a = Axis(fig[1, 1]; title = "Anelastic",
            xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
ax_e = Axis(fig[1, 2]; title = "Compressible explicit (Δt = 0.1 s)",
            xlabel = "x (km)", aspect = DataAspect())
ax_s = Axis(fig[1, 3]; title = "Compressible substepper (Ns = 12)",
            xlabel = "x (km)", aspect = DataAspect())

hm = heatmap!(ax_a, x_km, z_km, wa_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_e, x_km, z_km, we_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(     ax_s, x_km, z_km, ws_slice; colormap = :balance, colorrange = (-vmax, vmax))
Colorbar(fig[1, 4], hm; label = "w (m/s)")

mp4 = joinpath(OUTDIR, "bubble_three_way.mp4")
CairoMakie.record(fig, mp4, 1:Nt; framerate = 15) do i; n[] = i; end
@info "wrote $mp4"

gif = joinpath(OUTDIR, "bubble_three_way.gif")
CairoMakie.record(fig, gif, 1:Nt; framerate = 15) do i; n[] = i; end
@info "wrote $gif"

# Time series of max|w|
wa_peak = [_safe_max_abs(wa_ts[mask_a[i]]) for i in 1:Nt]
we_peak = [_safe_max_abs(we_ts[mask_e[i]]) for i in 1:Nt]
ws_peak = [_safe_max_abs(ws_ts[mask_s[i]]) for i in 1:Nt]
fig2 = Figure(size = (900, 400))
ax2 = Axis(fig2[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
           title = "Bubble peak |w|: anelastic vs explicit vs Ns=12 substepper")
lines!(ax2, ta[mask_a[1:Nt]],   wa_peak; label = "anelastic",        linewidth = 2)
lines!(ax2, te[mask_e[1:Nt]],   we_peak; label = "explicit",         linewidth = 2, linestyle = :dash)
lines!(ax2, tsub[mask_s[1:Nt]], ws_peak; label = "Ns=12 substepper", linewidth = 2, linestyle = :dot)
axislegend(ax2, position = :rb)
save(joinpath(OUTDIR, "peak_w.png"), fig2)
@info "wrote peak_w.png"
