## Shared IO for reassembling the distributed TC slice output into global fields.
##
## The run writes one JLD2 file per rank (each an x-slab of the domain). A
## checkpoint/restart CHAIN tags each job's files with its SLURM job id, so the
## full integration is spread over several file sets. These helpers (1) discover
## every file set under a directory, (2) stitch them into one global, time-sorted
## timeline (dropping the duplicate frame each restart writes at its pickup time),
## and (3) concatenate ranks along x — stripping halos — into global 2D fields.
##
## We read raw JLD2 rather than FieldTimeSeries: the latter's getindex fills the
## distributed FullyConnected halos, which needs MPI and errors in single-process
## post-processing.

const JLD2 = Base.require(Base.PkgId(Base.UUID("033835bb-8acc-5ee8-8aae-3f567f8a3819"), "JLD2"))

include(joinpath(@__DIR__, "gate_vertical_grid.jl"))

## One job's output: a base prefix (…_nz181 or …_nz181_job12345) and its rank count.
struct SliceRun
    dir::String
    base::String
    nranks::Int
end

## Find every file set whose name starts with `stage_prefix` (e.g. "tc_spinup_dx100m_gate_nz181").
function discover_runs(dir, stage_prefix)
    entries = readdir(dir)
    rank0 = filter(f -> startswith(f, stage_prefix) && endswith(f, "_rank0.jld2") && !occursin("checkpoint", f), entries)
    runs = SliceRun[]
    for f in rank0
        base = replace(f, "_rank0.jld2" => "")
        n = count(g -> startswith(g, base * "_rank") && endswith(g, ".jld2"), entries)
        push!(runs, SliceRun(dir, base, n))
    end
    isempty(runs) && error("no slice files $(stage_prefix)*_rank0.jld2 in $dir")
    return runs
end

iteration_keys(f, field) = sort(parse.(Int, filter(k -> k != "serialized", keys(f["timeseries/$field"]))))

## Global timeline: a vector of (run, iteration, time_seconds), sorted by time, with
## the duplicate frame at each restart's pickup instant collapsed to the later job.
function build_timeline(runs; ref_field = "u_surface")
    frames = Tuple{SliceRun, Int, Float64}[]
    for run in runs
        f0 = joinpath(run.dir, run.base * "_rank0.jld2")
        its, ts = JLD2.jldopen(f0) do f
            its = iteration_keys(f, ref_field)
            its, [Float64(f["timeseries/t/$i"]) for i in its]
        end
        for (i, t) in zip(its, ts)
            push!(frames, (run, i, t))
        end
    end
    sort!(frames, by = fr -> fr[3])
    kept = Tuple{SliceRun, Int, Float64}[]
    for fr in frames
        if !isempty(kept) && abs(fr[3] - kept[end][3]) < 1.0   # same instant (restart overlap) ⇒ keep later
            kept[end] = fr
        else
            push!(kept, fr)
        end
    end
    return kept
end

Δx_of(run) = parse(Int, match(r"dx(\d+)m", run.base).captures[1])
Nz_of(run)  = parse(Int, match(r"nz(\d+)", run.base).captures[1])
global_Nx(run, Lx) = round(Int, Lx / Δx_of(run))

## Read one rank's slice for `field` at `iter`, squeeze the singleton (sliced) axis.
function read_rank_raw(run, field, r, iter)
    a = JLD2.jldopen(joinpath(run.dir, run.base * "_rank$(r).jld2")) do f
        f["timeseries/$field/$iter"]
    end
    return Float32.(a)
end

## Assemble a global x-y field (Nx, Ny): concatenate rank x-slabs, strip x/y halos.
function assemble_xy(run, field, iter, Lx)
    Nx  = global_Nx(run, Lx)
    Nxl = Nx ÷ run.nranks
    Ny  = Nx
    slabs = map(0:(run.nranks - 1)) do r
        a  = dropdims(read_rank_raw(run, field, r, iter), dims = 3)   # (Nxl+2Hx, Ny+2Hy)
        Hx = (size(a, 1) - Nxl) ÷ 2
        Hy = (size(a, 2) - Ny)  ÷ 2
        a[Hx+1:Hx+Nxl, Hy+1:Hy+Ny]
    end
    return reduce(vcat, slabs)   # (Nx, Ny)
end

## Assemble a global x-z field (Nx, Nzᶠ): concatenate rank x-slabs, strip x/z halos.
## Returns (data, z) where z matches the field's vertical location (faces for w, else centers).
function assemble_xz(run, field, iter, Lx; ztop = 27000)
    Nx  = global_Nx(run, Lx)
    Nxl = Nx ÷ run.nranks
    slabs = map(0:(run.nranks - 1)) do r
        a  = dropdims(read_rank_raw(run, field, r, iter), dims = 2)   # (Nxl+2Hx, Nzᶠ+2Hz)
        Hx = (size(a, 1) - Nxl) ÷ 2                                   # halo uniform across dims
        a[Hx+1:Hx+Nxl, Hx+1:end-Hx]
    end
    data = reduce(vcat, slabs)                                       # (Nx, Nzᶠ)
    z_faces   = gate_vertical_grid(ztop)
    z_centers = 0.5 .* (z_faces[1:end-1] .+ z_faces[2:end])
    z = size(data, 2) == length(z_faces) ? z_faces : z_centers       # w lives on faces, others on centers
    return data, z
end
