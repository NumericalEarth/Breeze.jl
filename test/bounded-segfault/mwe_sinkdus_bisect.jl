# Breeze-level MWE: compute_velocities! at varying nsteps and reps.
#
# The nsteps-dominate-error.md says compute_velocities!
# (7 halo fills + 1 kernel per call) fails at nsteps >= 6.
#
# We test:
#   - compute_velocities! alone at nsteps = 1..12
#   - compute_velocities! × 3 reps per iter at nsteps = 1..8
#   - compute_velocities! + compute_auxiliary_thermodynamic_variables! at nsteps = 1..8
#   - full baseline C (+ bookkeeping) at nsteps = 1..8

using Oceananigans, Breeze, Reactant, Enzyme
using Oceananigans.Architectures: ReactantState
using Oceananigans: fields, prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Statistics: mean

using Breeze.AtmosphereModels: compute_velocities!,
    compute_auxiliary_thermodynamic_variables!,
    compute_auxiliary_dynamics_variables!,
    tracer_density_to_specific!,
    tracer_specific_to_density!

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4,4), extent=(1e3,1e3), halo=(3,3),
    topology=(Bounded, Bounded, Flat))

model  = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)
θ  = CenterField(grid); set!(θ, (x,y) -> 300 + 0.01x)
dθ = CenterField(grid); set!(dθ, 0)

# ─── A: compute_velocities! only, 1× per iter ───
function loss_A(model, θ, n)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        compute_velocities!(model)
    end
    return mean(interior(model.velocities.u) .^ 2)
end

# ─── B: compute_velocities! × 3 per iter ───
function loss_B(model, θ, n)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        compute_velocities!(model)
        compute_velocities!(model)
        compute_velocities!(model)
    end
    return mean(interior(model.velocities.u) .^ 2)
end

# ─── C: compute_velocities! + thermo, 1× per iter ───
function loss_C(model, θ, n)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        compute_velocities!(model)
        compute_auxiliary_thermodynamic_variables!(model)
    end
    return mean(interior(model.temperature) .^ 2)
end

# ─── D: compute_velocities! + thermo + dynamics, 1× per iter ───
function loss_D(model, θ, n)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        compute_velocities!(model)
        compute_auxiliary_thermodynamic_variables!(model)
        compute_auxiliary_dynamics_variables!(model)
    end
    return mean(interior(model.temperature) .^ 2)
end

function make_grad(loss_fn)
    function grad(model, dmodel, θ, dθ, n)
        Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss_fn, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ, dθ),
            Enzyme.Const(n))
    end
    return grad
end

function tag(e)
    msg = string(e)
    occursin("dominate", msg)    && return "DOMINANCE"
    occursin("donated", msg)     && return "DONATION"
    occursin("Segmentation", msg) && return "SEGFAULT"
    occursin("descendant", msg)  && return "SINKDUS"
    return first(msg, 60)
end

println("="^60)
println("A: compute_velocities! only, 1× per iter")
println("="^60)
grad_A = make_grad(loss_A)
for n in [1, 2, 4, 6, 8, 10, 12]
    try
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_A(model, dmodel, θ, dθ, n)
        compiled(model, dmodel, θ, dθ, n)
        println("  n=$n: ✓")
    catch e
        println("  n=$n: ✗ [$(tag(e))]")
    end
end

println("\n" * "="^60)
println("B: compute_velocities! × 3, per iter")
println("="^60)
grad_B = make_grad(loss_B)
for n in [1, 2, 4, 6, 8]
    try
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_B(model, dmodel, θ, dθ, n)
        compiled(model, dmodel, θ, dθ, n)
        println("  n=$n: ✓")
    catch e
        println("  n=$n: ✗ [$(tag(e))]")
    end
end

println("\n" * "="^60)
println("C: velocities + thermo, 1× per iter")
println("="^60)
grad_C = make_grad(loss_C)
for n in [1, 2, 4, 6, 8]
    try
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_C(model, dmodel, θ, dθ, n)
        compiled(model, dmodel, θ, dθ, n)
        println("  n=$n: ✓")
    catch e
        println("  n=$n: ✗ [$(tag(e))]")
    end
end

println("\n" * "="^60)
println("D: velocities + thermo + dynamics, 1× per iter")
println("="^60)
grad_D = make_grad(loss_D)
for n in [1, 2, 4, 6, 8]
    try
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_D(model, dmodel, θ, dθ, n)
        compiled(model, dmodel, θ, dθ, n)
        println("  n=$n: ✓")
    catch e
        println("  n=$n: ✗ [$(tag(e))]")
    end
end
