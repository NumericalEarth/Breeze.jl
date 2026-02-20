#=
MWE: Bounded Topology — Forward Pass Only
==========================================
Investigation: B.6.4 SinkDUS / Bounded Topology
Purpose: Test compilation of the forward pass (no AD) on (Bounded, Bounded, Flat).
Fix branch: Oceananigans.jl  dkz/fix-bounded-segfault

Run:  julia --project=test test/bounded-segfault/mwe_forward_pass.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Breeze
using Breeze: CompressibleDynamics
using Reactant
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size = (5, 5), extent = (1e3, 1e3), halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

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

function forward(model, θ, nsteps)
    set!(model, θ = θ, ρ = 1.0)
    @trace track_numbers = false for _ in 1:nsteps
        time_step!(model, 0.1)
    end
    return interior(model.temperature)
end

println("Compiling forward pass (Bounded, Bounded, Flat) …")
nsteps = 30
@time "compile" compiled = Reactant.@compile raise_first=true raise=true sync=true forward(model, θ, nsteps)

println("Running …")
@time "run" T_final = compiled(model, θ, nsteps)

println("T_final range: [$(minimum(T_final)), $(maximum(T_final))]")
println("✓ Forward pass complete")
