## Reassemble the per-rank x-y slice output of the distributed TC run into global
## horizontal fields and plot them. The distributed run writes one file per rank
## (each an x-slab); this concatenates them along x and makes figures.
##
## Usage:  julia --project=examples examples/visualize_tc_slices.jl [dir] [prefix]

using CairoMakie
using Printf

## JLD2 is a transitive dep (via Oceananigans/CairoMakie) but not a direct project dep;
## load it from the manifest by UUID so we can read the raw slice data directly. We avoid
## FieldTimeSeries here: its getindex fills the DISTRIBUTED (FullyConnected) halos, which
## needs MPI and errors in this single-process post-processing read.
const JLD2 = Base.require(Base.PkgId(Base.UUID("033835bb-8acc-5ee8-8aae-3f567f8a3819"), "JLD2"))

dir    = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(get(ENV, "SCRATCH", "."), "tc_distributed")
prefix = length(ARGS) ≥ 2 ? ARGS[2] : "tc_spinup_dx100m_gate_nz181"

files = filter(f -> startswith(f, prefix * "_rank") && endswith(f, ".jld2") && !occursin("checkpoint", f),
               readdir(dir))
Nranks = length(files)
Nranks == 0 && error("no slice files $(prefix)_rank*.jld2 in $dir")

Δx_m   = parse(Int, match(r"dx(\d+)m", prefix).captures[1])
Lx     = 642.0e3 / Δx_m * Δx_m   # = 642 km (kept explicit)
Lx     = 642.0e3
Nx     = round(Int, Lx / Δx_m)   # global Nx
Ny     = Nx
Nxl    = Nx ÷ Nranks
@info "Reassembling $Nranks ranks → $(Nx)×$(Ny) global slices (Δx=$(Δx_m) m)"

## iteration keys + times from rank 0
file0 = joinpath(dir, "$(prefix)_rank0.jld2")
iters, times = JLD2.jldopen(file0) do f
    ks = filter(k -> k != "serialized", keys(f["timeseries/u_surface"]))
    its = sort(parse.(Int, ks))
    ts  = [f["timeseries/t/$i"] for i in its]
    its, ts
end
@info "Snapshots (h): $(round.(times ./ 3600, digits=3))"

## Read a rank's slice for one iteration, strip halos to interior (Nxl × Ny).
function read_slice(field, r, iter)
    a = JLD2.jldopen(joinpath(dir, "$(prefix)_rank$(r).jld2")) do f
        f["timeseries/$field/$iter"]
    end
    a = dropdims(a, dims = 3)                 # (Nxl[+2Hx], Ny[+2Hy])
    Hx = (size(a, 1) - Nxl) ÷ 2
    Hy = (size(a, 2) - Ny)  ÷ 2
    return Float32.(a[Hx+1:Hx+Nxl, Hy+1:Hy+Ny])
end

assemble(field, iter) = reduce(vcat, [read_slice(field, r, iter) for r in 0:(Nranks-1)])  # (Nx, Ny)

n = length(iters)

## Surface wind speed at first and last snapshot — the vortex and its spinup.
Lkm = Lx / 2 / 1e3
fig = Figure(size = (1300, 560))
for (col, ti) in enumerate((1, n))
    u = assemble("u_surface", iters[ti]); v = assemble("v_surface", iters[ti])
    speed = sqrt.(u .^ 2 .+ v .^ 2)
    xs = range(-Lkm, Lkm, length = size(speed, 1)); ys = range(-Lkm, Lkm, length = size(speed, 2))
    ax = Axis(fig[1, 2col-1]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
              title = @sprintf("Surface wind speed, t = %.1f h  (max %.1f m/s)", times[ti]/3600, maximum(speed)),
              limits = (-200, 200, -200, 200))
    hm = heatmap!(ax, xs, ys, speed; colormap = :inferno, colorrange = (0, 45))
    Colorbar(fig[1, 2col], hm; label = "|u| (m/s)")
end
Label(fig[0, :], @sprintf("Distributed TC spinup (100 m, GATE-181, 60 GPU) — %s", prefix); fontsize = 16)

outdir = joinpath(@__DIR__, "tc_figures"); mkpath(outdir)
path = joinpath(outdir, "tc_surface_windspeed.png")
save(path, fig)
@info "Saved $path"
