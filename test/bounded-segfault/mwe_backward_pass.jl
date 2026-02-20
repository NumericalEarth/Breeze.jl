#=
MWE: Bounded Topology — Backward Pass (AD)
===========================================
Investigation: B.6.4 SinkDUS / Bounded Topology
Purpose: Test compilation of the backward pass (Enzyme AD) on (Bounded, Bounded, Flat).
         Plots initial θ, final T, and ∂loss/∂θ side by side.
Fix branch: Oceananigans.jl  dkz/fix-bounded-segfault

Run:  julia --project=test test/bounded-segfault/mwe_backward_pass.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Breeze
using Breeze: CompressibleDynamics
using Reactant
using Enzyme
using CUDA
using Reactant: @allowscalar
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size = (80, 80), extent = (1e3, 1e3), halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())
dmodel = Enzyme.make_zero(model)

θ = CenterField(grid)
set!(θ, (x, y) -> begin
    r = sqrt((x - 500)^2 + (y - 500)^2)
    ϕ = atan(y - 500, x - 500)
    spiral = 3sin(3ϕ + r / 80) * exp(-r^2 / (2 * 300^2))
    vortex = 5exp(-r^2 / (2 * 120^2))
    hotspot = -2exp(-((x - 750)^2 + (y - 300)^2) / 8e3)
    dipole  = 2exp(-((x - 200)^2 + (y - 750)^2) / 6e3) -
              2exp(-((x - 350)^2 + (y - 750)^2) / 6e3)
    ripple  = 0.8sin(2π * x / 100) * sin(2π * y / 100) * exp(-r^2 / (2 * 400^2))
    300 + vortex + spiral + hotspot + dipole + ripple
end)
dθ = CenterField(grid)
set!(dθ, 0.0)

θ_init = Array(interior(θ)[:, :, 1])

ix1, iy1 = 20, 10
ix2, iy2 = 30, 70

function loss(model, θ, nsteps)
    set!(model, θ = θ, ρ = 1.0)
    @trace track_numbers = false for _ in 1:nsteps
        time_step!(model, 0.005)
    end
    T = interior(model.temperature)
    return mean(T[ix1:ix2, iy1:iy2, 1:1] .^ 2)
end

function grad_loss(model, dmodel, θ, dθ, nsteps)
    _, val = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ, dθ),
        Enzyme.Const(nsteps))
    return val, interior(model.temperature), interior(dθ)
end

println("Compiling backward pass (Bounded, Bounded, Flat) …")
nsteps = 2000
@time "compile" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
    model, dmodel, θ, dθ, nsteps)

println("Running …")
@time "run" loss_val, T_final, grad_θ = compiled(model, dmodel, θ, dθ, nsteps)

println("Loss:      $loss_val")
println("T range:   [$(minimum(T_final)), $(maximum(T_final))]")
println("∂L/∂θ range: [$(minimum(grad_θ)), $(maximum(grad_θ))]")

# ─── Plot: initial θ | final T | gradient ∂L/∂θ ──────────────────────────────

outdir = joinpath(@__DIR__, "results")
mkpath(outdir)

try
    using CairoMakie

    T_slice = Array(T_final[:, :, 1])
    g_slice = Array(grad_θ[:, :, 1])

    fig = Figure(size = (1400, 400))

    ax1 = Axis(fig[1, 1]; title = "Initial θ", xlabel = "x", ylabel = "y")
    hm1 = heatmap!(ax1, θ_init; colormap = :thermal)
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3]; title = "Final T  (after $nsteps steps)", xlabel = "x", ylabel = "y")
    hm2 = heatmap!(ax2, T_slice; colormap = :thermal)
    Colorbar(fig[1, 4], hm2)

    ax3 = Axis(fig[1, 5]; title = "∂L/∂θ  (L = mean T[$ix1:$ix2,$iy1:$iy2]²)", xlabel = "x", ylabel = "y")
    hm3 = heatmap!(ax3, g_slice; colormap = :balance)
    Colorbar(fig[1, 6], hm3)

    path = joinpath(outdir, "bounded_backward.png")
    save(path, fig; px_per_unit = 2)
    println("Saved → $path")
catch e
    if e isa ArgumentError || e isa ErrorException
        println("CairoMakie not available — skipping plot (install with: Pkg.add(\"CairoMakie\"))")
    else
        rethrow(e)
    end
end

println("✓ Backward pass complete")
