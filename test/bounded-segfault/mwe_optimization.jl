#=
MWE: Bounded Topology — Gradient Descent Optimization
=====================================================
Purpose: Minimal optimization loop using AD gradients on (Bounded, Bounded, Flat).
         Parameters (amplitude, x₀, y₀) control a Gaussian bump IC for θ.
         We minimize mean(T²) in a sub-region via gradient descent.

Run:  julia --project=test test/bounded-segfault/mwe_optimization.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.TimeSteppers: time_step!
using Breeze
using Breeze: CompressibleDynamics
using Reactant
using Reactant: @allowscalar
using Enzyme
using CUDA
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size = (5, 5), extent = (1e3, 1e3), halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

model  = AtmosphereModel(grid; dynamics = CompressibleDynamics())
dmodel = Enzyme.make_zero(model)

# Coordinate arrays for building IC inside compiled region
xc = Reactant.to_rarray(Array(xnodes(grid, Center())))
yc = Reactant.to_rarray(Array(ynodes(grid, Center())))

# Pack parameters: [amplitude, x₀, y₀]
params  = Reactant.to_rarray([5.0, 500.0, 500.0])
dparams = Reactant.to_rarray(zeros(3))

ix1, iy1 = 5,  5
ix2, iy2 = 15, 15

function loss(model, params, xc, yc, nsteps)
    amp = @allowscalar params[1]
    x₀  = @allowscalar params[2]
    y₀  = @allowscalar params[3]

    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = (X .- x₀).^2 .+ (Y .- y₀).^2
    θ_vals = 300 .+ amp .* exp.(-r² ./ (2 * 150^2))

    ρ  = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density
    interior(ρ)  .= reshape(ones(size(θ_vals)), size(interior(ρ)))
    interior(ρθ) .= reshape(θ_vals, size(interior(ρθ)))

    @trace track_numbers = false mincut = true checkpointing = true for _ in 1:nsteps
        time_step!(model, 0.05)
    end
    T = interior(model.temperature)
    return mean(T[ix1:ix2, iy1:iy2, 1:1] .^ 2)
end

function grad_loss(model, dmodel, params, dparams, xc, yc, nsteps)
    dparams .= 0
    _, val = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(xc),
        Enzyme.Const(yc),
        Enzyme.Const(nsteps))
    return val, dparams
end

# ─── Compile ─────────────────────────────────────────────────────────────────

nsteps = 10

println("Compiling …")
@time "compile" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
    model, dmodel, params, dparams, xc, yc, nsteps)

# ─── Gradient descent loop ───────────────────────────────────────────────────

lr     = 1e-4
niters = 8
p_val  = [5.0, 500.0, 500.0]  # [amplitude, x₀, y₀]

println("\nOptimizing [amp, x₀, y₀] (lr=$lr, $niters iters, $nsteps steps/iter):")
for i in 1:niters
    Reactant.set!(params, p_val)
    Reactant.set!(dparams, zeros(3))

    L, dp = compiled(model, dmodel, params, dparams, xc, yc, nsteps)
    g = Float64.(Array(dp))

    p_val .-= lr .* g
    println("  iter $i  loss=$(Float64(Array(L)[]))  ∇=[$(join(round.(g, sigdigits=4), ", "))]  params=$(round.(p_val, sigdigits=5))")
end

println("\n✓ Optimization complete")
