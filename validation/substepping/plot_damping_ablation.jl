##### Render a 4-panel w snapshot comparing the damping variants.
##### Panels that crashed are plotted at their last valid frame.

using Oceananigans
using JLD2
using Printf
using CairoMakie

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_damping_ablation")

cases = [
    ("baseline_pressproj_0.5", "PressureProjectionDamping(0.5)"),
    ("mild_pressproj_0.1",     "PressureProjectionDamping(0.1)"),
    ("thermo_0.1",             "ThermodynamicDivergenceDamping(0.1)"),
    ("no_damping",             "NoDivergenceDamping"),
]

# Load each, take the last frame and its time.
function load_last(label)
    path = joinpath(OUTDIR, "$label.jld2")
    try
        ts = FieldTimeSeries(path, "w")
        return (ts[end], ts.times[end], length(ts.times))
    catch e
        @warn "failed to load $label: $e"
        return (nothing, nothing, 0)
    end
end

loaded = [(label, title, load_last(label)...) for (label, title) in cases]

# Shared color limits based on the BIGGEST |w| across surviving cases.
ws = [x[3] for x in loaded if x[3] !== nothing]
vmax = maximum(maximum(abs, interior(w)) for w in ws)

# Make a grid layout:
fig = Figure(size = (1400, 900))
for (idx, (label, title, w, t, n)) in enumerate(loaded)
    row, col = fldmod1(idx, 2)
    ax = Axis(fig[row, col];
              xlabel = "x (km)", ylabel = "z (km)",
              title = w === nothing ? "$(title) — NO DATA" :
                      @sprintf("%s — t=%.0fs (frame %d)", title, t, n))
    if w !== nothing
        x = xnodes(w) ./ 1e3
        z = znodes(w) ./ 1e3
        w_slice = interior(w)[:, 1, :]
        hm = heatmap!(ax, x, z, w_slice; colormap = :RdBu_9, colorrange = (-vmax, vmax))
        if idx == 1
            Colorbar(fig[:, 3], hm; label = "w (m/s)")
        end
    end
end
Label(fig[0, :], "Damping ablation: dry thermal bubble, final frame"; fontsize = 18, font = :bold)

outpath = joinpath(OUTDIR, "damping_ablation_final.png")
save(outpath, fig)
@info "Wrote $outpath"

# Also: a peak-|w|-vs-time plot showing the stability envelope of each variant.
fig2 = Figure(size = (900, 450))
ax = Axis(fig2[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
          title = "Damping ablation: peak |w| trajectory")
for (label, title) in cases
    path = joinpath(OUTDIR, "$label.jld2")
    try
        ts = FieldTimeSeries(path, "w")
        peaks = [maximum(abs, interior(ts[i])) for i in 1:length(ts.times)]
        lines!(ax, ts.times, peaks; label = title, linewidth = 2)
    catch
        # skip
    end
end
axislegend(ax; position = :lt)
save(joinpath(OUTDIR, "damping_ablation_peak_w.png"), fig2)
@info "Wrote peak_w plot"
