using CUDA, Reactant, Enzyme
using Breeze, Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: CenterField, interior
using Reactant: @allowscalar, @trace

Reactant.set_default_backend("cpu")

# ── Tiny grid + model ──

Nx, Nz = 16, 8
grid = RectilinearGrid(ReactantState(); size=(Nx, Nz),
                       x=(-5000, 5000), z=(0, 10000),
                       topology=(Periodic, Flat, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping()))
dmodel = Enzyme.make_zero(model)

# ── Fields used by the loss ──

θ₀ = CenterField(grid);  set!(θ₀, 300.0)

# ── Loss and gradient ──

function loss(model, θ₀)
    set!(model; θ=θ₀)
    return 0.0
end

function grad_loss(model, dmodel, θ₀)
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Const(θ₀))
    return J
end

# ── Compile and run ──

@info "Compiling..."
@time compiled = Reactant.@compile raise=true raise_first=true grad_loss(
    model, dmodel, θ₀)

@info "Running..."
J = compiled(model, dmodel, θ₀)
@info "J = $J"
