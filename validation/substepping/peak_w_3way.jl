#####
##### max|w|(t) comparison across anelastic / explicit / substepped (/tight).
#####

using Breeze, Oceananigans, Oceananigans.Units, CairoMakie

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")

function peaks_from(path, name = "w")
    ts = FieldTimeSeries(path, name)
    times = ts.times
    peaks = [maximum(abs, interior(ts[i])) for i in 1:length(times)]
    return times, peaks
end

entries = Dict{String,Any}()
for (label, file) in (
    ("anelastic",               "anelastic.jld2"),
    ("compressible explicit",   "explicit.jld2"),
    ("substepper cfl=0.3",      "compressible.jld2"),
    ("substepper Δt=0.1 fixed", "tight_dt01.jld2"),
    ("substepper Δt=0.25 N=12", "tight_dt025.jld2"),
    )
    path = joinpath(OUTDIR, file)
    if isfile(path)
        entries[label] = peaks_from(path)
    end
end

styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
ordered = [
    "anelastic",
    "compressible explicit",
    "substepper cfl=0.3",
    "substepper Δt=0.1 fixed",
    "substepper Δt=0.25 N=12",
]

fig = Figure(size = (1200, 900), fontsize = 14)
ax_full = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
               title = "Bubble peak w (full 1500 s)")
ax_zoom = Axis(fig[2, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
               title = "Zoom 0–500 s — ring event in wizard-cfl=0.3 substepper visible")

let i = 0
    for k in ordered
        haskey(entries, k) || continue
        i += 1
        t, w = entries[k]
        ls = styles[mod1(i, length(styles))]
        lines!(ax_full, t, w; linewidth = 2, linestyle = ls, label = k)
        mask = findall(<=(500.0), t)
        lines!(ax_zoom, t[mask], w[mask]; linewidth = 2, linestyle = ls)
    end
end

xlims!(ax_zoom, 0, 500)
axislegend(ax_full; position = :rb)
save(joinpath(OUTDIR, "peak_w_3way.png"), fig)
@info "wrote" path=joinpath(OUTDIR, "peak_w_3way.png") entries=collect(keys(entries))
