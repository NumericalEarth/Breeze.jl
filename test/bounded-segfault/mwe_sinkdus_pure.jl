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

# ── Test 6: Halo-heavy gradient ──────────────────────────────────────────────
println("\nTest 6 — Gradient: 3 fields + many fill_halo_regions!:")
let a = CenterField(grid)
    b = CenterField(grid)
    c = CenterField(grid)
    da = Enzyme.make_zero(a)
    db = Enzyme.make_zero(b)
    dc = Enzyme.make_zero(c)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_halo_heavy(
            a, da, b, db, c, dc, nsteps)
        compiled(a, da, b, db, c, dc, nsteps)
    end)
end

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

# ── Test 7: Diagnostic velocities ONLY ───────────────────────────────────────
println("\nTest 7 — Gradient: compute_velocities! ONLY in loop (THE diagnostic velocity test):")
run_test("Compile+Run", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_diag_vel(
        model, dmodel, θ_init, dθ_init, nsteps)
    compiled(model, dmodel, θ_init, dθ_init, nsteps)
end)

# ── Test 8: Full update_state! ───────────────────────────────────────────────
println("\nTest 8 — Gradient: full update_state! in loop:")
run_test("Compile+Run", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_update_state(
        model, dmodel, θ_init, dθ_init, nsteps)
    compiled(model, dmodel, θ_init, dθ_init, nsteps)
end)

# ── Test 9: Full time_step! (known crash) ─────────────────────────────────────
println("\nTest 9 — Gradient: full time_step! in loop (KNOWN CRASH CASE):")
run_test("Compile+Run", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_full_timestep(
        model, dmodel, θ_init, dθ_init, Δt, nsteps)
    compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)
end)

println("\n" * "=" ^ 72)
println("INTERPRETATION:")
println("  Test 6 PASS + Test 7 CRASH  → diagnostic velocities alone trigger it")
println("  Test 7 PASS + Test 8 CRASH  → thermodynamic/pressure/diffusion ops needed")
println("  Test 8 PASS + Test 9 CRASH  → tendency computation or RK3 substep needed")
println("  All PASS                    → crash needs full model complexity")
println("=" ^ 72)
