#=
BUG 1: stablehlo.pad shape mismatch (BBB backward + checkpointing)
═══════════════════════════════════════════════════════════════════

ERROR:
  'stablehlo.pad' op inferred type(s) 'tensor<0x4x5xf64>' are incompatible
  with return type(s) of operation 'tensor<1x4x5xf64>'

TRIGGER: Enzyme reverse-mode through a checkpointed @trace loop containing
  time_step! on a BBB CompressibleDynamics model. The adjoint of the
  _fill_flux_top_halo! setindex! on an asymmetric Face,Center,Center field
  (ρu with NoFlux BCs) gets a z-slab of thickness 0 instead of 1.

MINIMAL TRIGGER CONDITIONS (all required):
  1. Topology = (Bounded, Bounded, Bounded)
  2. CompressibleDynamics (diagnostic velocity → fill_halo_regions! on Face fields)
  3. checkpointing=true on the @trace loop
  4. Enzyme reverse mode (backward pass)
  5. Full time_step! in the loop body (simpler subsets don't trigger it)

WITHOUT any one of these, compilation succeeds.

Run: julia --check-bounds=no --project -e 'include("test/bbb-backward-pad-shape/mwe_bug1_pad_shape.jl")'
=#

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant, Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4, 4, 4), extent=(1, 1, 1),
    topology=(Bounded, Bounded, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
FT = eltype(grid)

dmodel = Enzyme.make_zero(model)
θ_init  = CenterField(grid); set!(θ_init,  (args...) -> FT(300))
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
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
        Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dθ_init, lv
end

Δt = FT(0.02)
nsteps = 4

@info "Bug 1: Expecting stablehlo.pad shape error..."
compiled = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)
