#=
MWE: Slice Bounds Error — Round 6: Tendency Bisection
======================================================
Reactant v0.2.221+, Julia 1.11+

CONFIRMED:
  update_state!(model)                        at 3x → FAIL
  update_state!(model, compute_tendencies=false) at 3x → PASS

compute_tendencies!(model) has these sub-components:
  M:  compute_momentum_tendencies! — 3 kernels (x,y,z momentum)
      Each reads from Face velocity fields (u,v,w), uses advection,
      pressure gradient (∂xᶠᶜᶜ), Coriolis, diffusion operators.
  Th: compute_thermodynamic_tendency! — 1 kernel (ρθ or ρe)
      Scalar tendency, reads velocities for advection.
  Q:  moisture tendency — 1 kernel (ρqᵗ), scalar tendency.
  D:  compute_dynamics_tendency! — 1 kernel (density ρ)
      Computes divᶜᶜᶜ(ρu, ρv, ρw), reads Face momentum fields.

All tendency kernels read from Face-located fields (velocities or momentum).
All are launched with :xyz worksize via launch!.

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_round6.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Utils: launch!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_velocities!, compute_auxiliary_thermodynamic_variables!,
    compute_auxiliary_dynamics_variables!, compute_tendencies!,
    compute_momentum_tendencies!, compute_thermodynamic_tendency!,
    compute_dynamics_tendency!, dynamics_density,
    compute_x_momentum_tendency!, compute_y_momentum_tendency!,
    compute_z_momentum_tendency!, compute_scalar_tendency!
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
println("Slice Bounds Error — Round 6: Tendency Bisection")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded,Bounded,Flat)")
println("All tests: forward-only, 3x per loop, nsteps=$nsteps")
println("Base = update_state!(compute_tendencies=false)")
println("=" ^ 72)

# Helper: the part of update_state! that passes (no tendencies)
function do_update_no_tendencies!(model)
    update_state!(model, compute_tendencies=false)
end

# ═════════════════════════════════════════════════════════════════════════════
# ROUND 1: Individual tendency sub-components
# ═════════════════════════════════════════════════════════════════════════════
# Each test: 3x (update_state_no_tend + ONE tendency component) per iteration
println_header("ROUND 1 — Individual tendency sub-components at 3x")

# M: Momentum tendencies only (3 kernels: x,y,z momentum)
run_test("M   momentum tendencies only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_momentum_tendencies!(model, Oceananigans.fields(model))
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# Th: Thermodynamic tendency only (1 kernel: ρθ)
run_test("Th  thermodynamic tendency only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        common_args = (model.dynamics, model.formulation, model.thermodynamic_constants,
            model.specific_moisture, model.velocities, model.microphysics,
            model.microphysical_fields, model.closure, model.closure_fields,
            model.clock, Oceananigans.fields(model))
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_thermodynamic_tendency!(model, common_args)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# Q: Moisture tendency only (1 kernel: ρqᵗ)
run_test("Q   moisture tendency only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                # Inline the moisture tendency kernel launch
                common_args = (model.dynamics, model.formulation, model.thermodynamic_constants,
                    model.specific_moisture, model.velocities, model.microphysics,
                    model.microphysical_fields, model.closure, model.closure_fields,
                    model.clock, Oceananigans.fields(model))
                ρq_args = (model.specific_moisture, Val(2), Val(:ρqᵗ),
                    model.forcing.ρqᵗ, model.advection.ρqᵗ, common_args...)
                Gρqᵗ = model.timestepper.Gⁿ.ρqᵗ
                launch!(model.grid.architecture, model.grid, :xyz,
                    compute_scalar_tendency!, Gρqᵗ, model.grid, ρq_args)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# D: Density tendency only (1 kernel: divᶜᶜᶜ of momentum)
run_test("D   density tendency only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_dynamics_tendency!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# ROUND 2: Cumulative tendency sub-components
# ═════════════════════════════════════════════════════════════════════════════
println_header("ROUND 2 — Cumulative tendencies at 3x")

# M only
run_test("M           momentum only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_momentum_tendencies!(model, Oceananigans.fields(model))
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# M + Th
run_test("M+Th        momentum + thermodynamic", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        common_args = (model.dynamics, model.formulation, model.thermodynamic_constants,
            model.specific_moisture, model.velocities, model.microphysics,
            model.microphysical_fields, model.closure, model.closure_fields,
            model.clock, Oceananigans.fields(model))
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_momentum_tendencies!(model, Oceananigans.fields(model))
                compute_thermodynamic_tendency!(model, common_args)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# M + D (momentum + density — both read from Face fields)
run_test("M+D         momentum + density", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_momentum_tendencies!(model, Oceananigans.fields(model))
                compute_dynamics_tendency!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# All tendencies (= compute_tendencies!)
run_test("ALL         compute_tendencies! (control)", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                compute_tendencies!(model)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# Full update_state! (ultimate control)
run_test("FULL        update_state! (ultimate control)", () -> begin
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

# ═════════════════════════════════════════════════════════════════════════════
# ROUND 3: Momentum tendency sub-components (if M alone fails)
# ═════════════════════════════════════════════════════════════════════════════
println_header("ROUND 3 — Individual momentum tendency kernels (if M fails)")

# x-momentum only
run_test("Mx  x_momentum_tendency only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        mf = Oceananigans.fields(model)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                grid = model.grid
                arch = grid.architecture
                Gρu = model.timestepper.Gⁿ.ρu
                momentum_args = (dynamics_density(model.dynamics),
                    model.advection.momentum, model.velocities, model.closure,
                    model.closure_fields, model.momentum, model.coriolis, model.clock, mf)
                u_args = tuple(momentum_args..., model.forcing.ρu, model.dynamics)
                launch!(arch, grid, :xyz,
                    compute_x_momentum_tendency!, Gρu, grid, u_args)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# y-momentum only
run_test("My  y_momentum_tendency only", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        mf = Oceananigans.fields(model)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                grid = model.grid
                arch = grid.architecture
                Gρv = model.timestepper.Gⁿ.ρv
                momentum_args = (dynamics_density(model.dynamics),
                    model.advection.momentum, model.velocities, model.closure,
                    model.closure_fields, model.momentum, model.coriolis, model.clock, mf)
                v_args = tuple(momentum_args..., model.forcing.ρv, model.dynamics)
                launch!(arch, grid, :xyz,
                    compute_y_momentum_tendency!, Gρv, grid, v_args)
                parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
            end
        end
        return mean(interior(model.temperature) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(model, θ, nsteps)
    c(model, θ, nsteps)
end)

# x + y momentum
run_test("Mxy x+y momentum tendencies", () -> begin
    function f(model, θ, n)
        set!(model, θ=θ, ρ=1.0)
        mf = Oceananigans.fields(model)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                do_update_no_tendencies!(model)
                grid = model.grid
                arch = grid.architecture
                Gρu = model.timestepper.Gⁿ.ρu
                Gρv = model.timestepper.Gⁿ.ρv
                momentum_args = (dynamics_density(model.dynamics),
                    model.advection.momentum, model.velocities, model.closure,
                    model.closure_fields, model.momentum, model.coriolis, model.clock, mf)
                u_args = tuple(momentum_args..., model.forcing.ρu, model.dynamics)
                v_args = tuple(momentum_args..., model.forcing.ρv, model.dynamics)
                launch!(arch, grid, :xyz,
                    compute_x_momentum_tendency!, Gρu, grid, u_args)
                launch!(arch, grid, :xyz,
                    compute_y_momentum_tendency!, Gρv, grid, v_args)
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
println("ROUND 1: Which tendency sub-component alone (+ update_no_tend)")
println("  triggers the error? If M alone fails, momentum tendencies are it.")
println("  If none fail individually, it's the COMBINATION.")
println()
println("ROUND 2: Cumulative addition finds the tipping point.")
println()
println("ROUND 3: If M fails, which momentum direction (x, y, or both)?")
println("  x_momentum reads from u (Face,Center,Center) via advection.")
println("  y_momentum reads from v (Center,Face,Center) via advection.")
println("  If only one direction fails, it's about that specific Face axis.")
println("=" ^ 72)
