#####
##### Top-down MWE: Build up from 1 RK3 stage toward the full time_step!
#####
# From mwe_topdown.jl we know:
#   - Variant F (1 stage: store + substep + update_state!) → PASSES
#   - Variant A (full time_step! with 3 stages) → FAILS
#
# This script adds complexity incrementally to find the trigger.
#
# Run: julia --project -e 'include("test/bbb-backward-pad-shape/mwe_topdown_stages.jl")'

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!, tick!, compute_flux_bc_tendencies!
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

# ──────────────────────────────────────────────────────────────────
# G: Two stages (store + substep + update_state!, repeated twice)
# ──────────────────────────────────────────────────────────────────

function loss_G(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α²)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ──────────────────────────────────────────────────────────────────
# H: Three stages (store + 3× substep + update_state!), no tick!, no flux BCs
# ──────────────────────────────────────────────────────────────────

function loss_H(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    α³ = model.timestepper.α³
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α²)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α³)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ──────────────────────────────────────────────────────────────────
# I: Three stages + tick! calls (like the real time_step!)
# ──────────────────────────────────────────────────────────────────

function loss_I(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    α³ = model.timestepper.α³
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        tick!(model.clock, Δt; stage = true)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α²)
        update_state!(model; compute_tendencies = true)
        ssp_rk3_substep!(model, Δt, α³)
        tick!(model.clock, Δt)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ──────────────────────────────────────────────────────────────────
# J: Three stages + compute_flux_bc_tendencies! (no tick!)
# ──────────────────────────────────────────────────────────────────

function loss_J(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    α¹ = model.timestepper.α¹
    α² = model.timestepper.α²
    α³ = model.timestepper.α³
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies = true)
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, α²)
        update_state!(model; compute_tendencies = true)
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, α³)
        update_state!(model; compute_tendencies = true)
    end
    return mean(interior(model.temperature).^2)
end

# ──────────────────────────────────────────────────────────────────
# K: Full time_step! (= variant A, for reference)
# ──────────────────────────────────────────────────────────────────

function loss_K(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
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
    ("G: 2 stages (no tick, no flux BCs)",         make_grad(loss_G)),
    ("H: 3 stages (no tick, no flux BCs)",         make_grad(loss_H)),
    ("I: 3 stages + tick!",                        make_grad(loss_I)),
    ("J: 3 stages + flux BCs (no tick)",           make_grad(loss_J)),
    ("K: full time_step!",                         make_grad(loss_K)),
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
println("RESULTS SUMMARY")
println("=" ^ 65)
for (label, _) in variants
    status = get(results, label, false) ? "✓ PASS" : "✗ FAIL"
    println("  $status  $label")
end
println("=" ^ 65)
