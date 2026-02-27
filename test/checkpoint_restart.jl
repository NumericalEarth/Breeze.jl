using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior
using Random: seed!

function build_model()
    Nx = Nz = 64
    Lz = 4 * 1024
    grid = RectilinearGrid(size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))
    p₀, θ₀ = 1e5, 288
    reference_state = ReferenceState(grid, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    Q₀ = 1000
    ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀))
    ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-2))
    advection = WENO()
    model = AtmosphereModel(grid; advection, dynamics,
                                  boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))
    return model, reference_state
end

function apply_initial_conditions!(model, reference_state)
    Δθ = 2
    Tₛ = reference_state.potential_temperature
    θᵢ(x, z) = Tₛ + Δθ * z / model.grid.Lz + 2e-2 * Δθ * (rand() - 0.5)
    set!(model, θ=θᵢ)
end

# ── No-restart run ──────────────────────────────────────────────────────────

seed!(42)
model_no_restart, reference_state = build_model()
apply_initial_conditions!(model_no_restart, reference_state)
sim_no_restart = Simulation(model_no_restart, Δt=10, stop_time=2hours)
sim_no_restart.output_writers[:checkpointer] = Checkpointer(model_no_restart,
                                                             schedule=IterationInterval(100),
                                                             prefix="no_restart_checkpoint")
conjure_time_step_wizard!(sim_no_restart, cfl=0.7)
run!(sim_no_restart, checkpoint_at_end=true)

# ── Restarted run (two segments) ────────────────────────────────────────────

# Segment 1: run to iteration 2000, writing checkpoint
seed!(42)
model_run0, reference_state = build_model()
apply_initial_conditions!(model_run0, reference_state)
sim_run0 = Simulation(model_run0, Δt=10, stop_iteration=2000)
sim_run0.output_writers[:checkpointer] = Checkpointer(model_run0,
                                                       schedule=IterationInterval(100),
                                                       prefix="run0_checkpoint")
conjure_time_step_wizard!(sim_run0, cfl=0.7)
run!(sim_run0)

# Segment 2: restart from checkpoint, run to stop_time
model_restart, _ = build_model()
sim_restart = Simulation(model_restart, Δt=10, stop_time=2hours)
sim_restart.output_writers[:checkpointer] = Checkpointer(model_restart,
                                                          schedule=IterationInterval(100),
                                                          prefix="run0_restart_checkpoint")
conjure_time_step_wizard!(sim_restart, cfl=0.7)
run!(sim_restart, pickup="run0_checkpoint_iteration2000.jld2", checkpoint_at_end=true)

# ── Compare final states ─────────────────────────────────────────────────────

println("\nComparing no-restart vs restarted final states:")
all_match_bitwise = true
all_match_approx = true

# check moisture density
ρqᵗ1 = model_no_restart.moisture_density
ρqᵗ2 = model_restart.moisture_density
max_diff = maximum(abs, interior(ρqᵗ1) .- interior(ρqᵗ2))
approx_match = (max_diff ≈ 0)
all_match_approx &= approx_match
all_match_bitwise &= (max_diff == 0)
println("  ρqᵗ: max|Δ| = $max_diff  $(approx_match ? "✓" : "✗")")

# check thermo
ρθ1 = model_no_restart.formulation.potential_temperature_density
ρθ2 = model_restart.formulation.potential_temperature_density
max_diff = maximum(abs, interior(ρθ1) .- interior(ρθ2))
approx_match = (max_diff ≈ 0)
all_match_approx &= approx_match
all_match_bitwise &= (max_diff == 0)
println("  ρθ: max|Δ| = $max_diff  $(approx_match ? "✓" : "✗")")

# check momenta
momentum_diffs = [maximum(abs, interior(model_no_restart.momentum[n]) .- interior(model_restart.momentum[n]))
                  for n in propertynames(model_no_restart.momentum)]
for (name, max_diff) in zip(propertynames(model_no_restart.momentum), momentum_diffs)
    println("  $name: max|Δ| = $max_diff  $(max_diff ≈ 0 ? "✓" : "✗")")
end
all_match_approx  &= all(d -> d ≈ 0, momentum_diffs)
all_match_bitwise &= all(d -> d == 0, momentum_diffs)

println(all_match_bitwise ? "\nPASS: restart is bitwise identical to no-restart." :
                            "\nFAIL: restart differs from no-restart.")
println(all_match_approx  ? "\nPASS: restart is approximately identical to no-restart." :
                            "\nFAIL: restart significantly differs from no-restart.")
