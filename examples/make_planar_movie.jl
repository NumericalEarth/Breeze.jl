## High-resolution planar (x-y) movie of the distributed TC run. Reassembles the
## per-rank slice output into global horizontal fields and animates them over the
## full checkpoint/restart timeline. Two panels: surface wind speed (the vortex and
## its eye) and vertical velocity at 6 km (convective bursts and rainbands).
##
## Usage:  julia --project=examples examples/make_planar_movie.jl [dir] [stage_prefix]
##   COARSEN=n  subsample by n in each direction for speed (default 2)
##   FRAMERATE=n  fps (default 10)

using CairoMakie
using Printf

include(joinpath(@__DIR__, "tc_slice_reader.jl"))

dir          = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(get(ENV, "SCRATCH", "."), "tc_distributed")
stage_prefix = length(ARGS) ≥ 2 ? ARGS[2] : "tc_spinup_dx100m_gate_nz181"
coarsen      = parse(Int, get(ENV, "COARSEN", "2"))
framerate    = parse(Int, get(ENV, "FRAMERATE", "10"))

const Lx = 642.0e3   # 642 km square domain

runs   = discover_runs(dir, stage_prefix)
frames = build_timeline(runs)
@info "Planar movie: $(length(runs)) job(s), $(length(frames)) frames, " *
      "t = $(round(frames[1][3]/3600, digits=2))–$(round(frames[end][3]/3600, digits=2)) h"

## Coordinates (km), windowed to the storm-scale region and coarsened for display.
Nx   = global_Nx(runs[1], Lx)
Lkm  = Lx / 2 / 1e3
xall = range(-Lkm, Lkm, length = Nx)
win  = findall(x -> abs(x) ≤ 250, xall)[1:coarsen:end]
xs   = xall[win]

n      = Observable(1)
sub(a) = a[win, win]
speed  = @lift let fr = frames[$n]
    sub(sqrt.(assemble_xy(fr[1], "u_surface", fr[2], Lx) .^ 2 .+ assemble_xy(fr[1], "v_surface", fr[2], Lx) .^ 2))
end
w6km   = @lift sub(assemble_xy(frames[$n][1], "w_z6km", frames[$n][2], Lx))
ttl    = @lift @sprintf("t = %.2f h", frames[$n][3] / 3600)

fig = Figure(size = (1500, 820))
Label(fig[0, :], @sprintf("Distributed TC — %s  (100 m, GATE-181, 60 GPU)", stage_prefix); fontsize = 18)

ax1 = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
           title = @lift("Surface wind speed   " * $ttl), limits = (-250, 250, -250, 250))
hm1 = heatmap!(ax1, xs, xs, speed; colormap = :inferno, colorrange = (0, 45))
Colorbar(fig[1, 2], hm1; label = "|u| (m/s)")

ax2 = Axis(fig[1, 3]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
           title = @lift("Vertical velocity at 6 km   " * $ttl), limits = (-250, 250, -250, 250))
hm2 = heatmap!(ax2, xs, xs, w6km; colormap = :balance, colorrange = (-6, 6))
Colorbar(fig[1, 4], hm2; label = "w (m/s)")

outdir = joinpath(@__DIR__, "tc_figures"); mkpath(outdir)
path = joinpath(outdir, "tc_planar_movie.mp4")
record(fig, path, 1:length(frames); framerate) do i
    n[] = i
end
@info "Saved $path"
