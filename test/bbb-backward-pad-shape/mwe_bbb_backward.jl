# B.6.10 MWE: BBB backward pass pad-shape mismatch
# Reproduces stablehlo.pad tensor<0x4x5xf64> vs tensor<1x4x5xf64>
# Run: julia --check-bounds=no --project -e 'include("test/bbb-backward-pad-shape/mwe_bbb_backward.jl")'

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                       topology=(Bounded, Bounded, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
FT = eltype(grid)
set!(model; θ=FT(300), ρ=one(FT))

dmodel = Enzyme.make_zero(model)
θ_init = CenterField(grid)
set!(θ_init, (args...) -> FT(300))
dθ_init = CenterField(grid)
set!(dθ_init, FT(0))

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature).^2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, lv
end

Δt = FT(0.001)
nsteps = 4

@info "Compiling BBB backward pass (expect stablehlo.pad shape error)..."
compiled = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

dθ, lv = compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)
@info "Loss: $lv  max|∇θ|: $(maximum(abs, interior(dθ)))"
