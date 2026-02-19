#=
MWE: Slice Bounds Error Bisection — Forward-Only at 3x
========================================================
Reactant v0.2.221+, Julia 1.11+
Assumes :xyz fix for compute_velocities! (Bug 1 resolved).

FINDING FROM ISOLATION MWE:
  - Levels 1-3 (pure halo fills, mixed Face+Center, even 3x): ALL PASS
  - Level 4 (Breeze update_state!): 1x PASS, 2x PASS, 3x FAIL
  - Forward AND gradient fail at 3x → error is NOT AD-specific
  - Since Enzyme is irrelevant, all tests here are forward-only (faster)

This MWE bisects what's inside update_state! at 3x per loop iteration
to find the minimal combination that triggers the slice bounds error.

update_state! internals:
  A: tracer_density_to_specific! (parent(ρc) ./= parent(ρ))
  B: fill_halo_regions!(prognostic_fields(model))
  C1: compute_velocities! (fill_halo + kernel + fill_halo)
  C2: compute_auxiliary_thermodynamic_variables! (kernel + fill_halo)
  C3: compute_auxiliary_dynamics_variables! (fill_halo + kernel + fill_halo)
  C4: compute_diffusivities!
  D: compute_tendencies! (momentum×3 + scalar + tracer)
  E: tracer_specific_to_density! (parent(c) .*= parent(ρ))

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_bisect.jl
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
println("Slice Bounds Error Bisection — Forward-Only at 3x")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded, Bounded, Flat)")
println("nsteps=$nsteps, 3 repetitions per loop body")
println("=" ^ 72)

# ═════════════════════════════════════════════════════════════════════════════
# ROUND 1: Individual sub-components × 3
# ═════════════════════════════════════════════════════════════════════════════
# Each test calls a single sub-component 3 times per loop iteration.
# If none fail individually, the error requires COMBINATION.

# println_header("ROUND 1 — Individual sub-components × 3 per iteration")

# # A: tracer_density_to_specific! (just element-wise division)
# run_test("A   3x tracer_density_to_specific!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             tracer_density_to_specific!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             tracer_density_to_specific!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             tracer_density_to_specific!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # B: fill_halo_regions!(prognostic_fields(model))
# run_test("B   3x fill_halo!(prognostic_fields)", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # C1: compute_velocities!
# run_test("C1  3x compute_velocities!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             compute_velocities!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_velocities!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_velocities!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # C2: compute_auxiliary_thermodynamic_variables!
# run_test("C2  3x compute_aux_thermo!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             compute_auxiliary_thermodynamic_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_auxiliary_thermodynamic_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_auxiliary_thermodynamic_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # C3: compute_auxiliary_dynamics_variables!
# run_test("C3  3x compute_aux_dynamics!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             compute_auxiliary_dynamics_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_auxiliary_dynamics_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_auxiliary_dynamics_variables!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # C4: compute_diffusivities!
# run_test("C4  3x compute_diffusivities!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             compute_diffusivities!(model.closure_fields, model.closure, model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_diffusivities!(model.closure_fields, model.closure, model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_diffusivities!(model.closure_fields, model.closure, model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# # D: compute_tendencies!
# run_test("D   3x compute_tendencies!", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             compute_tendencies!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_tendencies!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             compute_tendencies!(model)
#             parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# ═════════════════════════════════════════════════════════════════════════════
# ROUND 2: Cumulative sub-components × 3
# ═════════════════════════════════════════════════════════════════════════════
# Progressively add sub-components to find the tipping point.
# compute_auxiliary_variables! = C1 + C2 + C3 + C4

println_header("ROUND 2 — Cumulative sub-components × 3 per iteration")

# C1 only (same as Round 1, but establishes baseline for cumulative)
run_test("C1          3x compute_velocities! only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                compute_velocities!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# C1 + C2
run_test("C1+C2       3x velocities + thermo", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
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

# C1 + C2 + C3
run_test("C1+C2+C3    3x velocities + thermo + dynamics", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
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

# C1 + C2 + C3 + C4 = compute_auxiliary_variables!
run_test("C1+C2+C3+C4 3x compute_auxiliary_variables!", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# aux_vars + B (prognostic halo fill)
run_test("+B          3x aux_vars + prognostic halo fill", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
                compute_velocities!(model)
                compute_auxiliary_thermodynamic_variables!(model)
                compute_auxiliary_dynamics_variables!(model)
                compute_diffusivities!(model.closure_fields, model.closure, model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# aux_vars + B + D (tendencies)
run_test("+B+D        3x aux_vars + halo + tendencies", () -> begin
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

# # Full update_state! (known FAIL, control)
# run_test("FULL        3x update_state! (known FAIL)", () -> begin
#     function f(model, θ, n)
#         set!(model, θ=θ, ρ=1.0)
#         @trace track_numbers=false for _ in 1:n
#             for _ in 1:3
#                 update_state!(model)
#                 parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
#             end
#         end
#         return mean(interior(model.temperature) .^ 2)
#     end
#     c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
#     c(model, θ, nsteps)
# end)

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("INTERPRETATION:")
println()
println("ROUND 1: If any single component fails at 3x, that component alone")
println("  is sufficient to trigger the error. The bug is in that component.")
println()
println("ROUND 1 all pass: The error requires COMBINATION of sub-components.")
println()
println("ROUND 2: The first cumulative test that fails identifies the minimal")
println("  combination. E.g., if C1+C2 fails but C1 alone passes, then the")
println("  interaction between compute_velocities! and compute_thermo! is key.")
println()
println("Key suspect: compute_velocities! fills halos on Face-located velocity")
println("  fields (u, v) with wrong StaticSize, and later compute kernels READ")
println("  from those fields, creating MLIR slice ops that reference the wrong")
println("  tensor dimensions. At 3x, the accumulated references exceed what")
println("  MLIR can reconcile.")
println("=" ^ 72)
