#####
##### Narrow down: What about the second stage triggers the failure?
#####
# F (1 stage: store + substep + update_state!) → PASSES
# G (2 stages: store + substep + update + substep + update) → FAILS
#
# This script tests each piece in isolation to find the trigger.
#
# Run: julia --project -e 'include("test/bbb-backward-pad-shape/mwe_topdown_narrow.jl")'

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Breeze.TimeSteppers: store_initial_state!, ssp_rk3_substep!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size = (4, 4, 4), extent = (1e3, 1e3, 1e3),
    topology = (Bounded, Bounded, Bounded))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())
FT = eltype(grid)

dmodel = Enzyme.make_zero(model)
θ_init = CenterField(grid); set!(θ_init, (args...) -> FT(300))
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

Δt = FT(0.02)
nsteps = 4

# ── N1: Two update_state! calls, no substep in between ──
function loss_N1(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        update_state!(model; compute_tendencies = true)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ── N2: Two substeps (different α), no update_state! in between ──
function loss_N2(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        ssp_rk3_substep!(model, Δt, α²)
    end
    return mean(interior(model.temperature).^2)
end

# ── N3: One substep + two update_state! calls ──
function loss_N3(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ── N4: Two substeps + one update_state! at end ──
function loss_N4(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        ssp_rk3_substep!(model, Δt, α²)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ── N5: substep + update + substep (no second update) ──
function loss_N5(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α²)
    end
    return mean(interior(model.temperature).^2)
end

# ── N6: substep + update + just update (no second substep) ──
# This is G minus the second substep — is the second update_state! alone the problem?
function loss_N6(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        # Just the second update_state!, no second substep:
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ──────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────

function make_grad(loss_fn)
    function grad_fn(model, dmodel, θ_init, dθ_init, Δt, nsteps)
        parent(dθ_init) .= 0
        _, lv = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss_fn, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(θ_init, dθ_init),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps))
        return dθ_init, lv
    end
    return grad_fn
end

variants = [
    ("N1: 2× update_state! (no substep)",              make_grad(loss_N1)),
    ("N2: 2× substep (no update_state!)",              make_grad(loss_N2)),
    ("N3: 1× substep + 2× update_state!",             make_grad(loss_N3)),
    ("N4: 2× substep + 1× update_state!",             make_grad(loss_N4)),
    ("N5: substep + update + substep (no 2nd update)", make_grad(loss_N5)),
    ("N6: substep + update + update (no 2nd substep)", make_grad(loss_N6)),
]

results = Dict{String, Bool}()

for (label, grad_fn) in variants
    @info "Testing $label..."
    try
        compiled = Reactant.@compile raise=true raise_first=true sync=true grad_fn(
            model, dmodel, θ_init, dθ_init, Δt, nsteps)
        @info "  ✓ $label PASSED"
        results[label] = true
    catch e
        @warn "  ✗ $label FAILED" exception=(e, catch_backtrace())
        results[label] = false
    end
end

println("\n", "=" ^ 65)
println("RESULTS SUMMARY (recall: F=1 stage PASSES, G=2 stages FAILS)")
println("=" ^ 65)
for (label, _) in variants
    status = get(results, label, false) ? "✓ PASS" : "✗ FAIL"
    println("  $status  $label")
end
println("=" ^ 65)
