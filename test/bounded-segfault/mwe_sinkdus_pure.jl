#=
MWE: SinkDUS Segfault — Diagnostic Velocity Investigation (B.6.4)
==================================================================
Isolates whether Breeze's diagnostic velocity computation (u = ρu/ρ)
triggers the SinkDUS segfault on Bounded grids.

Previous findings (Tests 1-5 all PASSED):
  - Simple field ops on Bounded grid → safe
  - Two-field cross-updates → safe
  - Enzyme AD gradients of the above → safe
  - set! inside traced loop → safe

Hypothesis: The diagnostic velocity computation in Breeze's update_state!
is the trigger. It is unique because:
  1. Uses interpolation operators (ℑxᶠᵃᵃ) that access NEIGHBOR cells
  2. Runs over EXTENDED indices (1:N+1 for Bounded, not just 1:N)
  3. Writes to Face-located fields from Center-located data
  4. Involves fill_halo_regions! both BEFORE and AFTER the kernel
  5. Called 3x per time_step (once per SSPRK3 stage)

Tests in order of increasing Breeze complexity:
  6. fill_halo_regions! inside loop (many halo-fill kernel_calls)
  7. Breeze model + ONLY compute_velocities! in loop (GRADIENT)
  8. Breeze model + full update_state! in loop (GRADIENT)
  9. Breeze model + full time_step! in loop (known crash case)
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_velocities!
using Breeze.TimeSteppers: store_initial_state!, ssp_rk3_substep!
using Oceananigans.TimeSteppers: update_state!, compute_flux_bc_tendencies!, tick!
using Reactant
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

# ── MLIR dump setup ──────────────────────────────────────────────────────────
mlir_dump_dir = joinpath(@__DIR__, "mlir_dump_diag")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir

# ── Grid setup (Bounded topology, matching Breeze crash case) ────────────────
grid = RectilinearGrid(ReactantState();
    size = (4, 4),
    extent = (1000.0, 1000.0),
    halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

nsteps = 2
Δt = 0.01

# ── Helper ───────────────────────────────────────────────────────────────────
function run_test(label, fn; timeout_hint="")
    print("  $label: ")
    flush(stdout)
    try
        result = fn()
        println("OK (result=$result)")
        flush(stdout)
        return true
    catch e
        if e isa InterruptException
            rethrow()
        end
        msg = sprint(showerror, e)
        first_line = first(split(msg, '\n'))
        println("ERROR: $first_line")
        flush(stdout)
        return false
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — fill_halo_regions! inside loop (many halo fill kernel_calls)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests whether many fill_halo_regions! ops in the while body trigger the crash.
# Breeze's update_state! calls fill_halo_regions! on density, momentum (3 fields),
# velocities (3 fields), temperature, moisture, etc.

function loss_halo_heavy(a, b, c, nsteps)
    set!(a, 0.01)
    set!(b, 0.02)
    set!(c, 0.03)
    @trace track_numbers=false for i in 1:nsteps
        fill_halo_regions!(a)
        fill_halo_regions!(b)
        fill_halo_regions!(c)
        a .= a .* 0.9 .+ b .* 0.05 .+ c .* 0.05
        b .= b .* 0.8 .+ a .* 0.1 .+ c .* 0.1
        c .= c .* 0.7 .+ a .* 0.15 .+ b .* 0.15
        fill_halo_regions!(a)
        fill_halo_regions!(b)
        fill_halo_regions!(c)
    end
    return mean(interior(a) .^ 2) + mean(interior(b) .^ 2) + mean(interior(c) .^ 2)
end

function grad_loss_halo_heavy(a, da, b, db, c, dc, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_halo_heavy, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(b, db),
        Enzyme.Duplicated(c, dc),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Breeze model: ONLY compute_velocities! in loop (GRADIENT)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the key test. Breeze's compute_velocities! does:
#   1. fill_halo_regions!(ρ)  — sync halo fill of density
#   2. fill_halo_regions!(momentum)  — sync halo fill of ρu, ρv, ρw
#   3. launch! _compute_velocities! with extended range (1:N+1 for Bounded)
#      → u[i,j,k] = ρu[i,j,k] / ℑxᶠᵃᵃ(i,j,k,grid,ρ)  (reads ρ[i-1,j,k])
#      → v[i,j,k] = ρv[i,j,k] / ℑyᵃᶠᵃ(i,j,k,grid,ρ)  (reads ρ[i,j-1,k])
#   4. mask_immersed_field! on velocities
#   5. fill_halo_regions!(velocities)

function loss_diag_vel(model, θ_init, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        # Only the diagnostic velocity computation — no tendencies, no time stepping
        compute_velocities!(model)
        # Evolve momentum slightly to keep the loop non-trivial
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        parent(model.momentum.ρv) .= parent(model.momentum.ρv) .* 0.99
    end
    return mean(interior(model.velocities.u) .^ 2) + mean(interior(model.velocities.v) .^ 2)
end

function grad_loss_diag_vel(model, dmodel, θ_init, dθ_init, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_diag_vel, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Breeze model: full update_state! in loop (GRADIENT)
# ═══════════════════════════════════════════════════════════════════════════════
# update_state! = compute_velocities! + compute_auxiliary_thermodynamic_variables!
#               + compute_auxiliary_dynamics_variables! + compute_diffusivities!
#               + compute_tendencies!

function loss_update_state(model, θ_init, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        Oceananigans.TimeSteppers.update_state!(model)
        # Perturb prognostic fields slightly so loop is non-trivial
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_update_state(model, dmodel, θ_init, dθ_init, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_update_state, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8a — Breeze model: 2x update_state! per iteration
# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 (1x) passed. Does 2x trigger it?

function loss_2x_update_state(model, θ_init, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_2x_update_state(model, dmodel, θ_init, dθ_init, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_2x_update_state, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8a2 — Breeze model: 3x update_state! per iteration (like 3 RK3 stages)
# ═══════════════════════════════════════════════════════════════════════════════
# Known to CRASH from previous run.

function loss_3x_update_state(model, θ_init, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_3x_update_state(model, dmodel, θ_init, dθ_init, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_3x_update_state, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8b — Breeze model: store_initial + substep + update_state (1 RK3 stage)
# ═══════════════════════════════════════════════════════════════════════════════
# Adds the RK3 substep kernel: u = (1-α)*u⁰ + α*(u + Δt*G)
# This involves reading from U⁰ and Gⁿ fields, plus the substep kernel.

function loss_one_rk3_stage(model, θ_init, Δt, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        store_initial_state!(model)
        update_state!(model)
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, 1.0)
        update_state!(model)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_one_rk3_stage(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_one_rk3_stage, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8c — Breeze model: all 3 RK3 stages (= time_step! without tick/clock)
# ═══════════════════════════════════════════════════════════════════════════════
# This is essentially time_step! minus tick! and step_lagrangian_particles!.
# If this crashes, we've confirmed it's purely the 3-stage RK3 pattern.

function loss_three_rk3_stages(model, θ_init, Δt, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace track_numbers=false for i in 1:nsteps
        store_initial_state!(model)

        # Stage 1: α = 1
        update_state!(model)
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, 1.0)
        update_state!(model)

        # Stage 2: α = 1/4
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, 0.25)
        update_state!(model)

        # Stage 3: α = 2/3
        compute_flux_bc_tendencies!(model)
        ssp_rk3_substep!(model, Δt, 2/3)
        update_state!(model)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_three_rk3_stages(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_three_rk3_stages, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Breeze model: full time_step! in loop (known crash case)
# ═══════════════════════════════════════════════════════════════════════════════

function loss_full_timestep(model, θ_init, Δt, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    @trace mincut=true checkpointing=false track_numbers=false for i in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_full_timestep(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_full_timestep, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

println("=" ^ 72)
println("SinkDUS MWE — Diagnostic Velocity Investigation")
println("Grid: $(summary(grid))")
println("MLIR dumps → $mlir_dump_dir")
println("nsteps = $nsteps, Δt = $Δt")
println("=" ^ 72)

# ── Test 6: Halo-heavy gradient (SKIP — known PASS) ──────────────────────────
println("\nTest 6 — SKIPPED (known PASS from previous run)")

# ── Create Breeze model (shared for Tests 7-9) ──────────────────────────────
println("\nCreating Breeze AtmosphereModel...")
model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)
θ_init = CenterField(grid)
set!(θ_init, (x, y) -> 300.0 + 0.01 * x)
dθ_init = CenterField(grid)
set!(dθ_init, 0.0)
println("  Model created: $(typeof(model))")
println("  Prognostic fields: $(keys(Oceananigans.prognostic_fields(model)))")

# ── Test 7: Diagnostic velocities ONLY (SKIP — known PASS) ───────────────────
println("\nTest 7 — SKIPPED (known PASS from previous run)")

# ── Test 8: Full update_state! (SKIP — known PASS) ──────────────────────────
println("\nTest 8 — SKIPPED (known PASS from previous run)")

# ── Test 8a: 2x update_state! ─────────────────────────────────────────────────
println("\nTest 8a — Gradient: 2x update_state! per iteration:")
run_test("Compile+Run", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_2x_update_state(
        model, dmodel, θ_init, dθ_init, nsteps)
    compiled(model, dmodel, θ_init, dθ_init, nsteps)
end)

# ── Test 8a2: 3x update_state! (known crash) ────────────────────────────────
println("\nTest 8a2 — Gradient: 3x update_state! per iteration (KNOWN CRASH):")
run_test("Compile+Run", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_3x_update_state(
        model, dmodel, θ_init, dθ_init, nsteps)
    compiled(model, dmodel, θ_init, dθ_init, nsteps)
end)

# ── Tests 8b, 8c, 9: SKIPPED (focus on 2x vs 3x boundary) ────────────────────
println("\nTests 8b, 8c, 9 — SKIPPED (focus on 2x vs 3x threshold)")

println("\n" * "=" ^ 72)
println("RESULTS SUMMARY:")
println("  Test 8  (1x update_state!): KNOWN PASS")
println("  Test 8a (2x update_state!): ???")
println("  Test 8a2(3x update_state!): KNOWN CRASH")
println()
println("INTERPRETATION:")
println("  8a PASS  + 8a2 CRASH → crash threshold is between 2x and 3x update_state!")
println("  8a CRASH             → crash threshold is between 1x and 2x update_state!")
println("  The SinkDUS bug is a SCALE issue: too many ops in the while body")
println("=" ^ 72)
