#=
MWE: Slice Bounds Error — Round 5: The Missing Tendencies
==========================================================
Reactant v0.2.221+, Julia 1.11+

CRITICAL REALIZATION:
  The ONLY confirmed failure is: update_state!(model) × 3
  update_state! calls compute_tendencies!(model) by default.

  In ALL our manual decompositions (Rounds 2-4), we tested aux_vars
  WITHOUT compute_tendencies!. Individual tests of compute_tendencies!
  alone passed (Round 2, Test D). But the COMBINATION of all components
  INCLUDING tendencies was never tested.

  The tendency kernels are the heaviest part of update_state!:
    - 3 momentum tendency kernels (x_momentum, y_momentum, z_momentum)
    - scalar tendency kernels for each tracer
    Each reads from Face-located velocity fields and computes advection,
    pressure gradients, and diffusion via stencil operators.

  This round tests whether compute_tendencies! is the missing ingredient.

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_round5.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_velocities!, compute_auxiliary_thermodynamic_variables!,
    compute_auxiliary_dynamics_variables!, compute_tendencies!,
    tracer_density_to_specific!, tracer_specific_to_density!
using Reactant
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(5, 5), extent=(1e3, 1e3), halo=(3, 3),
    topology=(Bounded, Bounded, Flat))

nsteps = 2

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
θ = CenterField(grid); set!(θ, (x, y) -> 300 + 0.01x)

function run_test(label, fn)
    print("  $label: ")
    flush(stdout)
    try
        result = fn()
        println("PASS (result=$result)")
        flush(stdout)
        return :pass
    catch e
        e isa InterruptException && rethrow()
        msg = sprint(showerror, e)
        if contains(msg, "limit index") || contains(msg, "dimension size")
            m = match(r"limit index (\d+) is larger than dimension size (\d+) in dimension (\d+)", msg)
            if m !== nothing
                println("SLICE BOUNDS: limit=$(m[1]) > dim_size=$(m[2]) in dim $(m[3])")
            else
                println("SLICE BOUNDS ERROR")
            end
        elseif contains(msg, "signal (11)") || contains(msg, "SinkDUS")
            println("SEGFAULT")
        else
            first_line = first(split(msg, '\n'))
            println("ERROR: $(first_line[1:min(end, 120)])")
        end
        flush(stdout)
        return :fail
    end
end

println_header(s) = println("\n", "─" ^ 72, "\n", s, "\n", "─" ^ 72)

println("=" ^ 72)
println("Slice Bounds Error — Round 5: The Missing Tendencies")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded,Bounded,Flat)")
println("All tests: forward-only, 3x per loop, nsteps=$nsteps")
println("=" ^ 72)

# ═════════════════════════════════════════════════════════════════════════════
# CONTROL: Confirm update_state! at 3x fails (fresh session)
# ═════════════════════════════════════════════════════════════════════════════
println_header("CONTROL — Confirm update_state! × 3 failure")

run_test("CTRL  3x update_state!(model) [default compute_tendencies=true]", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                update_state!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# Also test: update_state! with compute_tendencies=false
run_test("CTRL2 3x update_state!(model, compute_tendencies=false)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                update_state!(model, compute_tendencies=false)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: aux_vars + compute_tendencies! (no extra halo fill B)
# ═════════════════════════════════════════════════════════════════════════════
println_header("TEST 1-3: Adding compute_tendencies! to previous combos")

run_test("T1  aux_vars + tendencies (no halo(prog))", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                compute_tendencies!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: B + aux_vars + compute_tendencies!
# ═════════════════════════════════════════════════════════════════════════════

run_test("T2  halo(prog,clock,fields) + aux_vars + tendencies", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                compute_tendencies!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Full manual update_state! (ALL components)
# ═════════════════════════════════════════════════════════════════════════════

run_test("T3  full manual: A+B+C+D+E (all components inline)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                tracer_density_to_specific!(model)
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                compute_tendencies!(model)
                tracer_specific_to_density!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4-5: Tendencies sub-components
# ═════════════════════════════════════════════════════════════════════════════
# compute_tendencies! = momentum_tendencies (3 kernels) + scalar tendencies
# If T1 fails, bisect what part of tendencies is needed.
println_header("TEST 4-5: Tendency sub-components (if T1 fails)")

run_test("T4  aux_vars + momentum_tendencies only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                Breeze.AtmosphereModels.compute_momentum_tendencies!(model, Oceananigans.fields(model))
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

run_test("T5  aux_vars + compute_tendencies! + tracer conversions", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                tracer_density_to_specific!(model)
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                compute_tendencies!(model)
                tracer_specific_to_density!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Does the async=true kwarg matter?
# ═════════════════════════════════════════════════════════════════════════════
println_header("TEST 6: async kwarg")

run_test("T6  halo(prog, clock, fields, async=true) + aux_vars + tend", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model), async=true)
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                compute_tendencies!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("INTERPRETATION:")
println()
println("CTRL FAIL + CTRL2 PASS → compute_tendencies is the key!")
println("CTRL FAIL + CTRL2 FAIL → tendencies don't matter, something")
println("  else in update_state! that we haven't isolated.")
println()
println("T1 FAIL → aux_vars + tendencies at 3x is enough, no extra halo")
println("  needed. Tendencies are the missing ingredient.")
println("T1 PASS + T2 FAIL → halo(prog) + aux_vars + tendencies needed.")
println("T1 PASS + T2 PASS + T3 FAIL → tracer conversions also needed.")
println("T1-T3 PASS → update_state! dispatch does something we can't")
println("  reproduce manually (check callbacks, default args, etc.)")
println()
println("T4: If momentum tendencies alone + aux_vars fail → momentum")
println("  tendency kernels (advection, pressure gradient) that read")
println("  from Face velocity fields are the trigger.")
println()
println("T6: Tests if async=true changes behavior under Reactant tracing.")
println("=" ^ 72)
