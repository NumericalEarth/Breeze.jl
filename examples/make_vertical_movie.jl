## High-resolution vertical-structure (x-z) movie of the distributed TC run. Animates
## the cross-section through the storm center (y = 0): vertical velocity w shows the
## eyewall updraft, convective towers, and rainband structure as the vortex evolves.
##
## Requires the x-z slice output (`w_xz`, `T_xz`, …) — produced by runs from the driver
## version that writes vertical cross-sections. Earlier x-y-only files are skipped.
##
## Usage:  julia --project=examples examples/make_vertical_movie.jl [dir] [stage_prefix]
##   COARSEN=n  subsample in x for speed (default 2)
##   FRAMERATE=n  fps (default 10)

using CairoMakie
using Printf

include(joinpath(@__DIR__, "tc_slice_reader.jl"))

dir          = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(get(ENV, "SCRATCH", "."), "tc_distributed")
stage_prefix = length(ARGS) ≥ 2 ? ARGS[2] : "tc_spinup_dx100m_gate_nz181"
coarsen      = parse(Int, get(ENV, "COARSEN", "2"))
framerate    = parse(Int, get(ENV, "FRAMERATE", "10"))

const Lx = 642.0e3

## Keep only runs that actually wrote the vertical cross-section.
allruns = discover_runs(dir, stage_prefix)
runs = filter(allruns) do run
    JLD2.jldopen(joinpath(run.dir, run.base * "_rank0.jld2")) do f
        haskey(f, "timeseries") && haskey(f["timeseries"], "w_xz")
    end
end
isempty(runs) && error("no x-z slice output (w_xz) found under $stage_prefix in $dir — " *
                       "only runs from the vertical-cross-section driver write it")
frames = build_timeline(runs; ref_field = "w_xz")
@info "Vertical movie: $(length(runs)) job(s), $(length(frames)) frames, " *
      "t = $(round(frames[1][3]/3600, digits=2))–$(round(frames[end][3]/3600, digits=2)) h"

Nx   = global_Nx(runs[1], Lx)
Lkm  = Lx / 2 / 1e3
xall = range(-Lkm, Lkm, length = Nx)
win  = findall(x -> abs(x) ≤ 200, xall)[1:coarsen:end]
xs   = xall[win]

_, z0 = assemble_xz(runs[1], "w_xz", frames[1][2], Lx)
zkm   = z0 ./ 1e3

n  = Observable(1)
wxz = @lift let fr = frames[$n]
    data, _ = assemble_xz(fr[1], "w_xz", fr[2], Lx)
    data[win, :]
end
ttl = @lift @sprintf("t = %.2f h", frames[$n][3] / 3600)

fig = Figure(size = (1500, 620))
Label(fig[0, :], @sprintf("Distributed TC vertical structure (y = 0) — %s", stage_prefix); fontsize = 18)
ax  = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "z (km)",
           title = @lift("Vertical velocity   " * $ttl),
           limits = (-200, 200, 0, 18))
hm  = heatmap!(ax, xs, zkm, wxz; colormap = :balance, colorrange = (-8, 8))
Colorbar(fig[1, 2], hm; label = "w (m/s)")

outdir = joinpath(@__DIR__, "tc_figures"); mkpath(outdir)
path = joinpath(outdir, "tc_vertical_movie.mp4")
record(fig, path, 1:length(frames); framerate) do i
    n[] = i
end
@info "Saved $path"
