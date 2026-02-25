#=
Bug 2 distillation: What inside the second update_state! triggers the failure?
═══════════════════════════════════════════════════════════════════════════════

From mwe_topdown_narrow.jl we know:
  substep + 1× update_state!  → PASS (variant F)
  substep + 2× update_state!  → FAIL (variant N3)

update_state! for AtmosphereModel does (in order):
  1. tracer_density_to_specific!
  2. fill_halo_regions!(prognostic_fields, ...)
  3. compute_auxiliary_variables!   ← includes compute_velocities! → fill_halo_regions!(velocities)
  4. update_radiation!
  5. compute_forcings!
  6. microphysics_model_update!
  7. compute_tendencies!
  8. tracer_specific_to_density!

This script replaces the second update_state! with individual pieces to find which one
is the minimal trigger when preceded by substep + full update_state!.

Run: julia --check-bounds=no --project -e 'include("test/bbb-backward-pad-shape/mwe_bug2_distill.jl")'
=#

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Breeze.TimeSteppers: store_initial_state!, ssp_rk3_substep!
using Breeze.AtmosphereModels: compute_auxiliary_variables!, compute_velocities!, compute_tendencies!
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4, 4, 4), extent=(1e3, 1e3, 1e3),
    topology=(Bounded, Bounded, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
FT = eltype(grid)

dmodel = Enzyme.make_zero(model)
θ_init  = CenterField(grid); set!(θ_init,  (args...) -> FT(300))
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

Δt = FT(0.02)
nsteps = 4

# ── D1: substep + update_state! + fill_halo_regions!(prognostic_fields) ──
function loss_D1(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
    end
    return mean(interior(model.temperature).^2)
end

# ── D2: substep + update_state! + compute_velocities! ──
function loss_D2(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        compute_velocities!(model)
    end
    return mean(interior(model.temperature).^2)
end

# ── D3: substep + update_state! + fill_halo_regions!(velocities) ──
function loss_D3(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        fill_halo_regions!(model.velocities)
    end
    return mean(interior(model.temperature).^2)
end

# ── D4: substep + update_state! + fill_halo_regions!(u only) ──
function loss_D4(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        fill_halo_regions!(model.velocities.u)
    end
    return mean(interior(model.temperature).^2)
end

# ── D5: substep + update_state! + compute_tendencies! ──
function loss_D5(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        compute_tendencies!(model)
    end
    return mean(interior(model.temperature).^2)
end

# ── D6: substep + update_state! + fill_halo_regions!(v only) ──
# Control: v is at (Center,Face,Center) — different from u at (Face,Center,Center)
function loss_D6(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=true)
        fill_halo_regions!(model.velocities.v)
    end
    return mean(interior(model.temperature).^2)
end

# ── D7: substep + update_state!(no tendencies) + update_state!(no tendencies) ──
# Is compute_tendencies! needed to trigger the bug, or just the halo fills?
function loss_D7(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=false)
        update_state!(model; compute_tendencies=false)
    end
    return mean(interior(model.temperature).^2)
end

# ── D8: substep + update_state!(no tendencies) + update_state!(with tendencies) ──
function loss_D8(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    α¹ = model.timestepper.α¹
    @trace checkpointing=true track_numbers=false for _ in 1:nsteps
        store_initial_state!(model)
        ssp_rk3_substep!(model, Δt, α¹)
        update_state!(model; compute_tendencies=false)
        update_state!(model; compute_tendencies=true)
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
            Enzyme.Const(Δt), Enzyme.Const(nsteps))
        return dθ_init, lv
    end
    return grad_fn
end

variants = [
    ("D1: update + fill_halo!(prognostic_fields)",  make_grad(loss_D1)),
    ("D2: update + compute_velocities!",            make_grad(loss_D2)),
    ("D3: update + fill_halo!(velocities)",         make_grad(loss_D3)),
    ("D4: update + fill_halo!(u only)",             make_grad(loss_D4)),
    ("D5: update + compute_tendencies!",            make_grad(loss_D5)),
    ("D6: update + fill_halo!(v only)",             make_grad(loss_D6)),
    ("D7: update(no tend) + update(no tend)",        make_grad(loss_D7)),
    ("D8: update(no tend) + update(with tend)",      make_grad(loss_D8)),
]

results = Dict{String, Any}()

for (label, grad_fn) in variants
    @info "Testing $label..."
    try
        compiled = Reactant.@compile raise=true raise_first=true sync=true grad_fn(
            model, dmodel, θ_init, dθ_init, Δt, nsteps)
        @info "  ✓ $label PASSED"
        results[label] = true
    catch e
        errmsg = sprint(showerror, e)
        short = first(errmsg, 120)
        @warn "  ✗ $label FAILED: $short"
        results[label] = false
    end
end

println("\n", "=" ^ 72)
println("BUG 2 DISTILLATION: Which piece of update_state! triggers the failure?")
println("All tests: substep + full update_state! + <piece>")
println("=" ^ 72)
for (label, _) in variants
    status = get(results, label, false) == true ? "✓ PASS" : "✗ FAIL"
    println("  $status  $label")
end
println("=" ^ 72)
println()
println("Expected pattern:")
println("  D3/D4 fail → the velocity halo fill (Open BC on Face fields) is the trigger")
println("  D1 fails   → even prognostic field halo fill (NoFlux on ρu) triggers it")
println("  D7 fails   → compute_tendencies! is NOT needed; just duplicate halo fills suffice")
