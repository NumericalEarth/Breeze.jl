# MWE: LLVM operand count mismatch on Julia 1.12 (works on 1.11)
#
# The Oceananigans periodic halo kernel has signature:
#   _fill_periodic_west_and_east_halo!(
#       c         :: OffsetArray{Float64, 3, CuTracedArray},
#       west_bc   :: BoundaryCondition{Periodic, Nothing},   # singleton / ghost
#       east_bc   :: BoundaryCondition{Periodic, Nothing},   # singleton / ghost
#       loc       :: Tuple{Center, Center, Center},           # singleton / ghost
#       grid      :: RectilinearGrid{...},                    # complex struct
#       args      :: Tuple{Clock, Tuple{}})                   # traced values
#
# On Julia 1.12, the LLVM function definition has 5 params but the call
# site passes 4 — the ghost/singleton types are lowered differently.
#
# This MWE reproduces the kernel argument structure without importing
# Oceananigans.

using CUDA, Reactant, Enzyme, KernelAbstractions, OffsetArrays
using Reactant: ConcreteRNumber
using GPUArraysCore: @allowscalar

Reactant.set_default_backend("cpu")
CUDA.allowscalar(true)

# ── Singleton types (ghost in 1.11, possibly non-ghost in 1.12) ──

struct Periodic end
struct Center end

struct BC{C, T}
    classification :: C
    condition :: T
end

const PBC = BC{Periodic, Nothing}
PBC() = BC(Periodic(), nothing)

# ── Grid (mirrors RectilinearGrid field layout) ──

struct MiniGrid{FT}
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
end

# ── Clock ──

mutable struct MiniClock{T, I}
    time          :: T
    last_Δt       :: T
    last_stage_Δt :: T
    iteration     :: I
    stage         :: Int
end

# ── Kernel: matches Oceananigans signature with all arg types ──
# 3D kernel (no inner loop) to avoid the scf.while issue and reach
# the LLVM lowering stage where the operand mismatch occurs.

@kernel function _fill_periodic_x!(c, west_bc, east_bc, loc, grid, args)
    i, j, k = @index(Global, NTuple)
    N = grid.Nx
    H = grid.Hx
    @inbounds parent(c)[i, j, k]          = parent(c)[N + i, j, k]
    @inbounds parent(c)[N + H + i, j, k] = parent(c)[H + i, j, k]
end

# ── Setup ──

Nx, Nz, H = 16, 8, 3

grid = MiniGrid(Nx, 1, Nz, H, 0, H, 10000.0, 0.0, 10000.0)

raw  = Reactant.to_rarray(zeros(Nx + 2H, 1, Nz + 2H))
c    = OffsetArray(raw, -H+1:Nx+H, 1:1, -H+1:Nz+H)

bc_w = PBC()
bc_e = PBC()
loc  = (Center(), Center(), Center())

clock = MiniClock(
    ConcreteRNumber(0.0),
    ConcreteRNumber(Inf),
    ConcreteRNumber(Inf),
    ConcreteRNumber(0),
    1)

mf = ()

function loss(c, bc_w, bc_e, loc, grid, clock, mf)
    backend = KernelAbstractions.get_backend(parent(c))
    args = (clock, mf)
    _fill_periodic_x!(backend)(c, bc_w, bc_e, loc, grid, args;
        ndrange=(grid.Hx, 1, grid.Nz + 2grid.Hz))
    KernelAbstractions.synchronize(backend)
    return 0.0
end

@info "Compiling..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(
    c, bc_w, bc_e, loc, grid, clock, mf)

@info compiled(c, bc_w, bc_e, loc, grid, clock, mf)
