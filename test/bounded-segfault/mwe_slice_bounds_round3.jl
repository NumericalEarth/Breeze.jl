#=
MWE: Slice Bounds Error — Round 3 Bisection
=============================================
Reactant v0.2.221+, Julia 1.11+

FINDINGS SO FAR:
  Round 1: All individual sub-components pass at 3x
  Round 2: compute_auxiliary_variables! alone passes at 3x
           Adding fill_halo_regions!(prognostic_fields) before it → FAILS

This round answers THREE questions:

Q1: Which prognostic field subsets in the halo fill trigger it?
    → momentum (Face fields) vs non-momentum (Center fields) vs all

Q2: Which aux_vars components are needed alongside the halo fill?
    → C1 only, C1+C2, etc.

Q3: Is it purely an MLIR op-count threshold?
    → Test aux_vars × 4 (no B) — if that fails, it's just complexity

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_round3.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_velocities!, compute_auxiliary_thermodynamic_variables!,
    compute_auxiliary_dynamics_variables!, compute_tendencies!
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

# Extract individual prognostic fields for targeted halo fills
ρ_field = Oceananigans.prognostic_fields(model).ρ       # Center,Center,Center
ρu_field = model.momentum.ρu                            # Face,Center,Center
ρv_field = model.momentum.ρv                            # Center,Face,Center
ρw_field = model.momentum.ρw                            # Center,Center,Face (but z=Flat)
prog = Oceananigans.prognostic_fields(model)

println("=" ^ 72)
println("Slice Bounds Error — Round 3 Bisection")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded,Bounded,Flat)")
println("All tests: forward-only, 3x per loop, nsteps=$nsteps")
println()
println("Prognostic fields: $(keys(prog))")
for (name, f) in pairs(prog)
    println("  $name: loc=$(Oceananigans.Fields.location(f)), parent_size=$(size(parent(f)))")
end
println("=" ^ 72)

# Helper: aux_vars = compute_velocities! + thermo + dynamics + diffusivities
function do_aux_vars!(model)
    compute_velocities!(model)
    compute_auxiliary_thermodynamic_variables!(model)
    compute_auxiliary_dynamics_variables!(model)
    compute_diffusivities!(model.closure_fields, model.closure, model)
end

# ═════════════════════════════════════════════════════════════════════════════
# Q1: Which prognostic field subsets trigger the error?
# ═════════════════════════════════════════════════════════════════════════════
# All tests: 3x (halo_subset + aux_vars) per loop body
println_header("Q1 — Which prognostic field halo fills trigger it? (+ aux_vars)")

# Momentum only (Face fields: ρu, ρv, ρw)
run_test("Q1a  halo(momentum) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(model.momentum)
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ρu only (Face in x on Bounded — the primary suspect)
run_test("Q1b  halo(ρu only) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(model.momentum.ρu)
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ρv only (Face in y on Bounded)
run_test("Q1c  halo(ρv only) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(model.momentum.ρv)
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# Non-momentum (Center fields only: ρ, ρθ, ρqᵗ)
run_test("Q1d  halo(ρ + ρθ + ρqᵗ) + aux_vars [all Center, control]", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        ρ = Oceananigans.prognostic_fields(model).ρ
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(ρ)
                fill_halo_regions!(model.moisture_density)
                fill_halo_regions!(Oceananigans.prognostic_fields(model.formulation))
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# Q2: Which aux_vars components are needed alongside full prognostic halo?
# ═════════════════════════════════════════════════════════════════════════════
println_header("Q2 — Which aux_vars components? (+ full prognostic halo)")

function halo_prog!(model)
    fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
end

# B + C1 only
run_test("Q2a  halo(prog) + compute_velocities! only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                halo_prog!(model)
                compute_velocities!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# B + C2 only
run_test("Q2b  halo(prog) + compute_thermo! only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                halo_prog!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# B + C1 + C2
run_test("Q2c  halo(prog) + velocities + thermo", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                halo_prog!(model)
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# B + C1 + C2 + C3
run_test("Q2d  halo(prog) + vel + thermo + dynamics", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                halo_prog!(model)
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# Q3: Is it a pure MLIR op-count threshold?
# ═════════════════════════════════════════════════════════════════════════════
# aux_vars at 3x passes. If 4x or 5x fails WITHOUT B, it's just op-count.
println_header("Q3 — Pure complexity threshold? (aux_vars only, more reps)")

run_test("Q3a  aux_vars × 4 (no prognostic halo)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:4
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

run_test("Q3b  aux_vars × 5 (no prognostic halo)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:5
                do_aux_vars!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

run_test("Q3c  aux_vars × 6 (no prognostic halo)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:6
                do_aux_vars!(model)
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
println("Q1: If Q1a/Q1b/Q1c fail but Q1d passes → Face-field halo fills")
println("    with wrong worksize are the trigger. The specific Face field")
println("    (ρu vs ρv) tells us which Bounded axis is involved.")
println()
println("    If Q1d also fails → it's not Face-specific; any extra halo")
println("    fills push past the complexity threshold.")
println()
println("Q2: The first failure identifies which compute kernel READS from")
println("    the Face fields and generates the conflicting MLIR slice op.")
println("    Likely suspect: compute_velocities! (C1) reads ρu/ρv and")
println("    writes to u/v velocity Face fields via interpolation.")
println()
println("Q3: If Q3a-c fail → pure complexity threshold, not Face-specific.")
println("    If Q3a-c pass → the prognostic halo fill adds something")
println("    qualitatively different (Face-field worksize inconsistency),")
println("    not just more ops.")
println("=" ^ 72)
