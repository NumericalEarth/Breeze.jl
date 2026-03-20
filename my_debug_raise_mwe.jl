using CUDA, Reactant, Enzyme
using Breeze, Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, Field, instantiated_location
using Reactant: TracedRNumber, ConcreteRNumber
using OffsetArrays: OffsetArray

Reactant.Compiler.DUMP_LLVMIR[] = false

Reactant.set_default_backend("cpu")

# ── Inlined from Oceananigans.BoundaryConditions ──

# ── Setup ──

Nx, Nz = 16, 8
grid = RectilinearGrid(ReactantState(); size=(Nx, Nz),
                       x=(-5000, 5000), z=(0, 10000),
                       topology=(Periodic, Flat, Bounded))

ρ      = CenterField(grid)
clock  = Clock(time=0.0, last_Δt=Inf, last_stage_Δt=Inf, iteration=ConcreteRNumber(0), stage=1)
mf     = ()
we_kernel = ρ.boundary_conditions.kernels.west_and_east
we_bcs    = ρ.boundary_conditions.ordered_bcs.west_and_east
loc       = instantiated_location(ρ)

function loss(ρ, clock, mf)
    we_kernel(ρ.data, we_bcs[1], we_bcs[2], loc, ρ.grid, (clock, mf))
    return 0.0
end

@info "Compiling with $(length(mf)) fields..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(ρ, clock, mf)
