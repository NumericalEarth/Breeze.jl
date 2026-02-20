#=
MWE: Bounded Topology Slice Bounds Error — x-Momentum Tendency
===============================================================
Investigation: B.6.4 SinkDUS / Bounded Topology
Status: FAILING
Purpose: Minimal reproduction of slice bounds error triggered by
         compute_momentum_tendencies! on (Bounded, Bounded, Flat) grid.
Related: investigations/bounded-sinkdus-segfault.md
Synchronized with: Breeze.jl/test/bounded-segfault/

Findings from bisection (Rounds 1–7):
  ✓ update_state!(model, compute_tendencies=false) at 3x → PASS
  ✗ update_state!(model, compute_tendencies=true)  at 3x → FAIL
  ✗ update_no_tend + compute_momentum_tendencies!   at 3x → FAIL
  ✗ update_no_tend + compute_x_momentum_tendency!   at 3x → FAIL (Mx)
  ✓ update_no_tend + compute_y_momentum_tendency!   at 3x → PASS (My)

Error: "limit index 6 is larger than dimension size 5 in dimension 1"
  → Face field on Bounded x-axis has N+1=6 interior points,
    but work_layout returns N=5. The x-momentum stencil (div_𝐯u)
    accesses u[i+1] at i=5, reaching index 6 in dimension 1.

Run:  julia --project=test test/bounded-segfault/mwe_x_momentum_genuine.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_momentum_tendencies!
using Reactant
using Statistics: mean
using CUDA
using Pkg

Reactant.set_default_backend("cpu")

println("Package versions:")
for pkg in ["Oceananigans", "Breeze", "Reactant", "Enzyme"]
    v = Pkg.dependencies()[Base.UUID(Pkg.project().dependencies[pkg])].version
    println("  $pkg: v$v")
end
println()

# ─── Grid + Model ─────────────────────────────────────────────────────────────

@time "Constructing grid" grid = RectilinearGrid(ReactantState();
    size = (5, 5),
    extent = (1e3, 1e3),
    halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

@time "Constructing model" model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

θ = CenterField(grid)
set!(θ, (x, y) -> 300 + 0.01x)

nsteps = 2

# ─── Loss function ────────────────────────────────────────────────────────────
# Mirrors the pattern from Round 6 "M" test: all args built inside the trace.

function loss(model, θ, nsteps)
    set!(model, θ = θ, ρ = 1.0)

    @trace track_numbers = false for _ in 1:nsteps
        for _ in 1:3
            update_state!(model, compute_tendencies = false)
            compute_momentum_tendencies!(model, Oceananigans.fields(model))
            parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        end
    end

    return mean(interior(model.temperature) .^ 2)
end

# ─── Compile + run ────────────────────────────────────────────────────────────

println("Expected: SLICE BOUNDS error (limit index 6 > dimension size 5 in dim 1)\n")

@time "Compiling loss" compiled = Reactant.@compile raise_first = true raise = true sync = true loss(
    model, θ, nsteps)

@time "Running compiled loss" result = compiled(model, θ, nsteps)

println("\nResult: $result")
