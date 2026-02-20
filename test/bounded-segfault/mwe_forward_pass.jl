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
set!(θ, (x, y) -> 300 + 0.01x)

function loss(model, θ, nsteps)
    set!(model, θ = θ, ρ = 1.0)
    @trace track_numbers = false for _ in 1:nsteps
        time_step!(model, 0.1)
    end
    return mean(interior(model.temperature) .^ 2)
end

println("Compiling forward pass (Bounded, Bounded, Flat) …")
nsteps = 3
@time "compile" compiled = Reactant.@compile raise_first=true raise=true sync=true loss(model, θ, nsteps)

println("Running …")
@time "run" result = compiled(model, θ, nsteps)

println("Result: $result  ✓")
