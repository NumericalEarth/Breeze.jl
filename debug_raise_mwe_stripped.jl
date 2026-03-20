# MWE: fill_halo_regions! stripped of Oceananigans
#
# Original (debug_raise_mwe.jl) compiles fill_halo_regions! on:
#   CenterField on RectilinearGrid(ReactantState();
#       size=(16,8), topology=(Periodic, Flat, Bounded))
#
# Replacements:
#   RectilinearGrid → Grid struct (Int sizes only — Lx/Δx are ConcreteRNumber
#                     in the real grid but unused by halo fill)
#   CenterField     → raw 3D parent array, shape (Nx+2H, 1, Nz+2H)
#   Clock           → MiniClock (time/iteration as ConcreteRNumber)
#   fill_halo_regions! → direct loops (no KernelAbstractions, no BC dispatch)
#
# Topology (Periodic, Flat, Bounded):
#   x — periodic wrap    (H ghost cells each side)
#   y — flat, no halo    (dim size 1)
#   z — no-flux / mirror (H ghost cells each side)
#
# Parent array layout (1-based):
#   x-interior: [H+1 : H+Nx]     halos: [1:H] and [H+Nx+1 : 2H+Nx]
#   z-interior: [H+1 : H+Nz]     halos: [1:H] and [H+Nz+1 : 2H+Nz]

using CUDA, Reactant, Enzyme, KernelAbstractions
using Reactant: ConcreteRNumber
using GPUArraysCore: @allowscalar

Reactant.set_default_backend("cpu")
CUDA.allowscalar(true)

Reactant.Compiler.DUMP_LLVMIR[] = false

# ── Types ──

struct Grid
    Nx :: Int
    Nz :: Int
    Hx :: Int
    Hz :: Int
end

mutable struct MiniClock{T, I}
    time          :: T
    last_Δt       :: T
    last_stage_Δt :: T
    iteration     :: I
    stage         :: Int
end

# ── Halo fill (KernelAbstractions kernels) ──
# The inner `for i in 1:H` loop generates an scf.while in MLIR that the
# StableHLO raiser can't handle.  Fix: promote the halo index to a kernel
# dimension so each thread handles exactly one halo cell — no loops.
# N and H are passed directly as Int args (not through the struct).

@kernel function _fill_periodic_x!(c, N, H)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k]          = c[N + i, j, k]      # west ← east interior
    @inbounds c[N + H + i, j, k] = c[H + i, j, k]      # east ← west interior
end

@kernel function _fill_flux_z!(c, N, H)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k]          = c[i, j, 2H + 1 - k]      # bottom mirror
    @inbounds c[i, j, N + H + k] = c[i, j, N + H + 1 - k]    # top mirror
end

function fill_halos!(c, g, clock, mf)
    backend = KernelAbstractions.get_backend(c)
    _fill_periodic_x!(backend)(c, g.Nx, g.Hx;
        ndrange=(g.Hx, 1, g.Nz + 2g.Hz))
    _fill_flux_z!(backend)(c, g.Nz, g.Hz;
        ndrange=(g.Nx + 2g.Hx, 1, g.Hz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ── Setup ──

Nx, Nz, H = 16, 8, 3
g = Grid(Nx, Nz, H, H)

c = Reactant.to_rarray(zeros(Nx + 2H, 1, Nz + 2H))

clock = MiniClock(
    ConcreteRNumber(0.0),
    ConcreteRNumber(Inf),
    ConcreteRNumber(Inf),
    ConcreteRNumber(0),
    1)

mf = ()

function loss(c, clock, mf)
    fill_halos!(c, g, clock, mf)
    return 0.0
end

@info "Compiling..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(c, clock, mf)

@info compiled(c, clock, mf)