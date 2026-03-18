# Minimal script: LatitudeLongitudeGrid + WENO + Reactant backward pass
#
# Purpose: verify that the BreezeReactantExt set_product! workaround is active
# and that a reverse-mode Enzyme pass compiles and runs on LatitudeLongitudeGrid.
#
# Note: loss writes directly to model internals (density, ρθ, ρu) rather than
# using set!(model; ρ=…, θ=…, u=…). This avoids the update_state! cascade that
# triggers global density reads inside set!, which produces spurious gradients via
# Enzyme even when those paths are dead. See discussion in acoustic_wave.jl.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

#####
##### Grid and model
#####

Nλ, Nφ, Nz = 8, 8, 4

grid = LatitudeLongitudeGrid(ReactantState();
                              size    = (Nλ, Nφ, Nz),
                              halo    = (5, 5, 5),
                              longitude = (0, 360),
                              latitude  = (-85, 85),
                              z         = (0, 1e3))

model = AtmosphereModel(grid; dynamics  = CompressibleDynamics(ExplicitTimeStepping()),
                              advection = WENO(order=5))

dmodel = Enzyme.make_zero(model)

#####
##### Input fields
#####

FT = eltype(grid)
Δt = FT(0.5)
nsteps = 1

# Initial density perturbation — the quantity we differentiate w.r.t.
δρ  = CenterField(grid)
dδρ = CenterField(grid)
set!(δρ, FT(0))
set!(dδρ, FT(0))

# Fixed background fields (Const in the backward pass)
ρ₀  = CenterField(grid);    set!(ρ₀, FT(1))
ρθ₀ = CenterField(grid);    set!(ρθ₀, FT(300))   # ρθ = ρ × θ = 1 × 300

#####
##### Loss and gradient wrapper
#####

# Direct-write initialisation: avoids the update_state! cascade inside
# set!(model; ρ=…, θ=…). The first time_step! call will run update_state!
# internally, where its outputs ARE used, making the backward pass correct.
function loss(model, δρ, ρ₀, ρθ₀, Δt, nsteps)
    ρ  = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density

    parent(ρ)  .= parent(ρ₀) .+ parent(δρ)
    parent(ρθ) .= parent(ρθ₀)
    parent(model.momentum.ρu) .= 0
    parent(model.momentum.ρw) .= 0

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    return mean(interior(ρ) .^ 2)
end

function grad_loss(model, dmodel, δρ, dδρ, ρ₀, ρθ₀, Δt, nsteps)
    parent(dδρ) .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(δρ, dδρ),
        Enzyme.Const(ρ₀),
        Enzyme.Const(ρθ₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dδρ, J
end

#####
##### Compile and run
#####

@info "Compiling backward pass (LatitudeLongitudeGrid + WENO + Enzyme)..."
compiled = Reactant.@compile raise=true raise_first=true sync=true dump_hlo=true grad_loss(
    model, dmodel, δρ, dδρ, ρ₀, ρθ₀, Δt, nsteps)
@info "Compilation done."

grad, J = compiled(model, dmodel, δρ, dδρ, ρ₀, ρθ₀, Δt, nsteps)

J_val = Float64(only(J))
grad_arr = Array(interior(grad, :, :, :))

@info "J = $J_val"
@info "‖∂J/∂δρ‖∞ = $(maximum(abs, grad_arr))"
@info "any NaN in gradient? $(any(isnan, grad_arr))"
@info "any Inf in gradient? $(any(isinf, grad_arr))"
