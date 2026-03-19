using CUDA, Reactant, Enzyme
using Breeze, Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant: TracedRNumber, ConcreteRNumber

Reactant.set_default_backend("cpu")

Nx, Nz = 16, 8
grid = RectilinearGrid(ReactantState(); size=(Nx, Nz),
                       x=(-5000, 5000), z=(0, 10000),
                       topology=(Periodic, Flat, Bounded))

# ── Print what fields(model) contains so we can see the keys ──
# mf_full = fields(model)
# @info "fields(model) keys: $(keys(mf_full))"
# @info "fields(model) types: $(map(typeof, values(mf_full)))"

# ── Extract real objects ──

ρ      = CenterField(grid)
dρ     = Enzyme.make_zero(ρ)
clock  = Clock(time=0.0, last_Δt=Inf, last_stage_Δt=Inf, iteration=ConcreteRNumber(0), stage=1)
dclock = Enzyme.make_zero(clock)

mf = ()

dmf = Enzyme.make_zero(mf)

function loss(ρ, clock, mf)
    fill_halo_regions!(ρ, clock, mf)
    return 0.0
end

@info "Compiling with $(length(mf)) fields..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(
    ρ, clock, mf)
