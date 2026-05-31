## NCCL correctness test for distributed (x-partitioned) Oceananigans grids.
##
## Verifies the two things the distributed atmosphere model relies on, against
## analytic single-GPU truth:
##   1. Global reductions (maximum / minimum / sum) — used by the CFL wizard.
##   2. Halo filling (Center and Face fields, x-partition) — used every step.
##   3. set! interior population — the suspected culprit behind the production
##      Δt=Inf crash (max|u| → 0 implies the field was left empty).
##
## Run the SAME test with `--no-nccl` (plain MPI, known-good) to compare. Because
## a periodic analytic field is the ground truth, each rank self-checks its own
## halos without a gather.
##
## Usage (see test_nccl_correctness.sh):
##   srun -n4 julia --project=examples examples/test_nccl_correctness.jl          # NCCL
##   srun -n4 julia --project=examples examples/test_nccl_correctness.jl --no-nccl # MPI

using MPI
MPI.Init()

## Cray MPICH inserts a malformed env entry after MPI_Init on multi-node srun,
## which breaks CUDA.jl → multi-node GPU hangs. Strip it (Oceananigans #5513).
include(joinpath(@__DIR__, "sanitize_environ.jl"))
SanitizeEnviron.sanitize_environ!()

using Oceananigans
using Oceananigans.Grids: xnode, ynode, znode, topology
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CUDA
using NCCL
using Printf

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
NCCLExt === nothing && error("OceananigansNCCLExt did not load")
const NCCLDistributed = NCCLExt.NCCLDistributed

## Precision matters: the TC run is Float32. A wrong NCCL buffer dtype/size would
## corrupt halos only in Float32, so default the test to Float32.
FT = ("--float64" in ARGS) ? Float64 : Float32
Oceananigans.defaults.FloatType = FT

use_nccl = !("--no-nccl" in ARGS)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)
arch  = use_nccl ?
    NCCLDistributed(GPU(); partition = Partition(Ngpus, 1)) :
    Distributed(GPU(); partition = Partition(Ngpus, 1))

backend = use_nccl ? "NCCL" : "MPI"
rank == 0 && @info "NCCL correctness test" backend Ngpus

## Global grid: x-partition needs global Nx divisible by Ngpus.
Nx = 16 * Ngpus
Ny = 24
Nz = 12
Hx = Hy = Hz = 3
Lx = 2π; Ly = 2π; Lz = 1.0   # periodic in x,y so the analytic field wraps cleanly

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz), halo = (Hx, Hy, Hz),
                       x = (0, Lx), y = (0, Ly), z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

## Smooth field, periodic in x and y so halo wrap-around has an exact analytic value.
f(x, y, z) = sin(x) * cos(2y) * (1 + z)

## Halo error for an x-partitioned field, checking west+east x-halos against f at
## the global node coordinates. ℓx is the field's x-location (Center()/Face()).
function max_xhalo_error(field, grid, f, ℓx, ℓy, ℓz, Hx, Hy, Hz, Ny, Nz)
    fp = Array(parent(field))
    Nxl = size(fp, 1) - 2Hx
    err = 0.0
    for kp in (Hz+1):(Hz+Nz), jp in (Hy+1):(Hy+Ny)
        k = kp - Hz; j = jp - Hy
        for ip in vcat(1:Hx, (Nxl+Hx+1):(Nxl+2Hx))
            i = ip - Hx
            xh = xnode(i, j, k, grid, ℓx, ℓy, ℓz)
            yh = ynode(i, j, k, grid, ℓx, ℓy, ℓz)
            zh = znode(i, j, k, grid, ℓx, ℓy, ℓz)
            err = max(err, abs(fp[ip, jp, kp] - f(xh, yh, zh)))
        end
    end
    return err
end

## ---- Reductions vs analytic global truth (on a Center field) ----
c = CenterField(grid); set!(c, f); fill_halo_regions!(c)
xC = [(i - 0.5) * Lx / Nx for i in 1:Nx]
yC = [(j - 0.5) * Ly / Ny for j in 1:Ny]
zC = [(k - 0.5) * Lz / Nz for k in 1:Nz]
ref_vals = [f(x, y, z) for x in xC, y in yC, z in zC]
ref_max = maximum(ref_vals); ref_min = minimum(ref_vals); ref_sum = sum(ref_vals)
got_max = maximum(c); got_min = minimum(c); got_sum = sum(c)
interior_absmax = maximum(abs, c)
field_scale = maximum(abs, ref_vals)

## ---- Halo correctness for Center, XFace, YFace fields ----
cerr = MPI.Allreduce(max_xhalo_error(c, grid, f, Center(), Center(), Center(), Hx, Hy, Hz, Ny, Nz), MPI.MAX, MPI.COMM_WORLD)

u = XFaceField(grid); set!(u, f); fill_halo_regions!(u)
uerr = MPI.Allreduce(max_xhalo_error(u, grid, f, Face(), Center(), Center(), Hx, Hy, Hz, Ny, Nz), MPI.MAX, MPI.COMM_WORLD)

v = YFaceField(grid); set!(v, f); fill_halo_regions!(v)
verr = MPI.Allreduce(max_xhalo_error(v, grid, f, Center(), Face(), Center(), Hx, Hy, Hz, Ny, Nz), MPI.MAX, MPI.COMM_WORLD)

if rank == 0
    tol = FT == Float32 ? 1f-3 : 1e-8   # discretization-exact; only roundoff expected
    abstol_sum = field_scale * length(ref_vals) * (FT == Float32 ? 1f-3 : 1e-10)
    @info @sprintf("[%s/%s] reductions: max |Δ|=%.2e  min |Δ|=%.2e  sum |Δ|=%.2e (tol %.1e)",
                   backend, FT, abs(got_max-ref_max), abs(got_min-ref_min), abs(got_sum-ref_sum), abstol_sum)
    @info @sprintf("[%s/%s] set! interior max|c| = %.6f (ref %.6f; must be > 0)", backend, FT, interior_absmax, field_scale)
    @info @sprintf("[%s/%s] halo fill max error:  Center=%.2e  XFace=%.2e  YFace=%.2e (tol %.1e)",
                   backend, FT, cerr, uerr, verr, tol)
    pass = abs(got_max-ref_max) < tol*field_scale && abs(got_min-ref_min) < tol*field_scale &&
           abs(got_sum-ref_sum) < abstol_sum && interior_absmax > 1e-6 &&
           cerr < tol && uerr < tol && verr < tol
    @info pass ? "[$backend/$FT] ===== ALL CHECKS PASSED =====" : "[$backend/$FT] ***** FAILURE DETECTED *****"
end

MPI.Barrier(MPI.COMM_WORLD)
