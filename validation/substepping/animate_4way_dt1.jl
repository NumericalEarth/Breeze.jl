##### Four-panel w animation: anelastic | Ns=12 | Ns=48 | explicit
##### All run with the same outer physics setup; only the integrator differs.
##### Cases that NaN'd produce short JLD2 files — the animation stops at the
##### shortest common frame count so we can watch them diverge.

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_4way_dt1")

labels = ("anelastic", "Ns12", "Ns48", "explicit")

titles = (
    "Anelastic\nΔt = 1.0 s",
    "Substepper Ns=12\nΔt = 1.0 s (ω=0.8, β=0.5)",
    "Substepper Ns=48\nΔt = 1.0 s (ω=0.8, β=0.5)",
    "Compressible explicit\nΔt = 0.1 s (acoustic CFL ≈ 0.45)",
)

# Load w time series.
w_series = [FieldTimeSeries(joinpath(OUTDIR, "$l.jld2"), "w") for l in labels]
times_per_case = [ts.times for ts in w_series]
for (l, t) in zip(labels, times_per_case)
    @info "$l: $(length(t)) frames, t in [$(minimum(t)), $(maximum(t))] s"
end

# Use the shortest run as the animation length so all panels have data each frame.
Nt = minimum(length.(times_per_case))
@info "Animation will use $Nt frames (shortest run)"

grid = w_series[1].grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

# Color range from the anelastic reference so substepper blowups don't saturate it.
vmax = let v = 0.0
    for i in 1:Nt
        v = max(v, maximum(abs, interior(w_series[1][i])))
    end
    v > 0 ? v : 1.0
end
@info "Shared color range ±$vmax m/s (from anelastic)"

n = Observable(1)
slices = [@lift Array(interior(w_series[k][$n]))[:, 1, :] for k in 1:4]

fig = Figure(size = (2000, 650), fontsize = 14)
title_node = @lift @sprintf("Dry thermal bubble (128², GPU) — t = %5.1f s",
                            times_per_case[1][$n])
fig[0, 1:5] = Label(fig, title_node, fontsize = 20, tellwidth = false)

local hm
for (k, (t, s)) in enumerate(zip(titles, slices))
    ax = Axis(fig[1, k]; title = t,
              xlabel = "x (km)",
              ylabel = k == 1 ? "z (km)" : "",
              aspect = DataAspect())
    hm = heatmap!(ax, x_km, z_km, s; colormap = :balance, colorrange = (-vmax, vmax))
    k == 4 && Colorbar(fig[1, 5], hm; label = "w (m/s)")
end

mp4 = joinpath(OUTDIR, "bubble_4way_dt1.mp4")
record(fig, mp4, 1:Nt; framerate = 10) do i
    n[] = i
end
@info "wrote $mp4"

# Still at final frame too.
n[] = Nt
save(joinpath(OUTDIR, "bubble_4way_dt1_final.png"), fig)
@info "wrote bubble_4way_dt1_final.png"
