using CUDA, Reactant, Enzyme
using Breeze, Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!

Reactant.set_default_backend("cpu")

Nx, Nz = 16, 8
grid = RectilinearGrid(ReactantState(); size=(Nx, Nz),
                       x=(-5000, 5000), z=(0, 10000),
                       topology=(Periodic, Flat, Bounded))

model  = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping()))
dmodel = Enzyme.make_zero(model)

# ── Print what fields(model) contains so we can see the keys ──
# mf_full = fields(model)
# @info "fields(model) keys: $(keys(mf_full))"
# @info "fields(model) types: $(map(typeof, values(mf_full)))"

# ── Extract real objects ──

ρ      = CenterField(grid)
dρ     = Enzyme.make_zero(ρ)
# clock = model.clock
# dclock = dmodel.clock
clock  = Clock(time=0.0, last_Δt=Inf, last_stage_Δt=Inf, iteration=0, stage=1)
dclock = Enzyme.make_zero(clock)

mf = ()

dmf = Enzyme.make_zero(mf)

function loss(ρ, clock, mf)
    fill_halo_regions!(ρ, clock, mf)
    return 0.0
end

function grad_loss(ρ, dρ, clock, dclock, mf, dmf)
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(ρ, dρ),
        Enzyme.Duplicated(clock, dclock),
        Enzyme.Duplicated(mf, dmf))
    return J
end

@info "Compiling with $(length(mf)) fields..."
@time compiled = Reactant.@compile raise=true raise_first=true grad_loss(
    ρ, dρ, clock, dclock, mf, dmf)

@info "Running..."
J = compiled(ρ, dρ, clock, dclock, mf, dmf)
@info "J = $J"
