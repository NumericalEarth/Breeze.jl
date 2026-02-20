#=
MWE: Slice Bounds Error — Round 4: Call Style & Extra Args
===========================================================
Reactant v0.2.221+, Julia 1.11+

FINDINGS SO FAR:
  Round 1-2: halo(all_prog, clock, fields) + aux_vars × 3 → FAIL
  Round 3:   halo(momentum) + aux_vars × 3 → PASS
             halo(Center fields) + aux_vars × 3 → PASS
             aux_vars × 6 → PASS  (NOT an op-count issue)

CRITICAL OBSERVATION:
  The FAILING test used:
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
  The PASSING tests used:
    fill_halo_regions!(model.momentum)      # no extra args
    fill_halo_regions!(ρ_field)             # no extra args

  Oceananigans passes the extra args (clock, fields(model)) through to
  every halo fill kernel — even though NoFluxBoundaryCondition ignores
  them. During Reactant tracing, `fields(model)` is a NamedTuple of ALL
  model fields, generating MLIR ops in each kernel for tracing these
  unused arguments.

  This round discriminates between:
    A) ALL fields together vs subsets
    B) NamedTuple call vs individual calls
    C) Extra args (clock, fields) vs no extra args

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_round4.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_velocities!, compute_auxiliary_thermodynamic_variables!,
    compute_auxiliary_dynamics_variables!
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

function do_aux_vars!(model)
    compute_velocities!(model)
    compute_auxiliary_thermodynamic_variables!(model)
    compute_auxiliary_dynamics_variables!(model)
    compute_diffusivities!(model.closure_fields, model.closure, model)
end

prog = Oceananigans.prognostic_fields(model)
all_fields = Oceananigans.fields(model)

println("=" ^ 72)
println("Slice Bounds Error — Round 4: Call Style & Extra Args")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded,Bounded,Flat)")
println("All tests: forward-only, 3x per loop, nsteps=$nsteps")
println()
println("Prognostic fields: $(keys(prog))")
println("All fields: $(keys(all_fields))")
println("=" ^ 72)

# ═════════════════════════════════════════════════════════════════════════════
# Test 1: ALL prognostic fields, INDIVIDUAL calls, NO extra args
# ═════════════════════════════════════════════════════════════════════════════
println_header("Test 1-4: Varying call style and extra args")

run_test("T1  individual calls, NO extra args + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        pf = Oceananigans.prognostic_fields(model)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                for i in eachindex(pf)
                    @inbounds fill_halo_regions!(pf[i])
                end
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
# Test 2: ALL prognostic fields, as NamedTuple, NO extra args
# ═════════════════════════════════════════════════════════════════════════════

run_test("T2  NamedTuple call, NO extra args + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model))
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
# Test 3: ALL prognostic fields, INDIVIDUAL calls, WITH clock + fields args
# ═════════════════════════════════════════════════════════════════════════════

run_test("T3  individual calls, WITH clock+fields + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        pf = Oceananigans.prognostic_fields(model)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                for i in eachindex(pf)
                    @inbounds fill_halo_regions!(pf[i], model.clock, Oceananigans.fields(model))
                end
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
# Test 4: ALL prognostic fields, NamedTuple call, WITH clock + fields args
#         (= Round 2 "+B", known FAIL)
# ═════════════════════════════════════════════════════════════════════════════

run_test("T4  NamedTuple call, WITH clock+fields + aux_vars  [known FAIL]", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
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
# Test 5: Isolate extra args WITHOUT aux_vars
# ═════════════════════════════════════════════════════════════════════════════
# If extra args are the trigger, do they also fail without aux_vars?
println_header("Test 5-6: Extra args alone (no aux_vars)")

run_test("T5  NamedTuple + clock+fields, NO aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(Oceananigans.prognostic_fields(model), model.clock, Oceananigans.fields(model))
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# Test 6: Extra args on momentum ONLY + aux_vars
# ═════════════════════════════════════════════════════════════════════════════
# Round 3 Q1a passed: halo(momentum) without extra args.
# Does adding extra args to momentum halo + aux_vars fail?
println_header("Test 6-7: Extra args on field subsets + aux_vars")

run_test("T6  halo(momentum, clock, fields) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(model.momentum, model.clock, Oceananigans.fields(model))
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
# Test 7: Extra args on Center fields ONLY + aux_vars
# ═════════════════════════════════════════════════════════════════════════════

run_test("T7  halo(ρ+ρθ+ρqᵗ, clock, fields) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        ρ = Oceananigans.prognostic_fields(model).ρ
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(ρ, model.clock, Oceananigans.fields(model))
                fill_halo_regions!(model.moisture_density, model.clock, Oceananigans.fields(model))
                fill_halo_regions!(Oceananigans.prognostic_fields(model.formulation), model.clock, Oceananigans.fields(model))
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
# Test 8: Momentum + Center fields separately, both WITH extra args + aux_vars
# ═════════════════════════════════════════════════════════════════════════════
println_header("Test 8: All fields separately with extra args + aux_vars")

run_test("T8  momentum(+args) + center(+args) + aux_vars", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        ρ = Oceananigans.prognostic_fields(model).ρ
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(model.momentum, model.clock, Oceananigans.fields(model))
                fill_halo_regions!(ρ, model.clock, Oceananigans.fields(model))
                fill_halo_regions!(model.moisture_density, model.clock, Oceananigans.fields(model))
                fill_halo_regions!(Oceananigans.prognostic_fields(model.formulation), model.clock, Oceananigans.fields(model))
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
println("DECISION TREE:")
println()
println("T1 PASS + T2 PASS + T3 FAIL → extra args on individual calls trigger it")
println("T1 PASS + T2 PASS + T3 PASS + T4 FAIL → NamedTuple dispatch + args needed")
println("T1 PASS + T2 FAIL → NamedTuple dispatch alone (no args needed)")
println("T1 FAIL → having ALL fields halo-filled (any style) triggers it")
println()
println("T5 FAIL → extra args alone (without aux_vars) is enough")
println("T5 PASS → extra args + aux_vars interaction needed")
println()
println("T6 FAIL + T7 PASS → extra args on Face fields (momentum) is key")
println("T6 PASS + T7 FAIL → extra args on Center fields is key")
println("T6 PASS + T7 PASS + T8 FAIL → both subsets + args needed together")
println("T6 PASS + T7 PASS + T8 PASS → only full prognostic tuple + args triggers it")
println("=" ^ 72)
