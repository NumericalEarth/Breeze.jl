# Which parts of the Lagrangian droplet microphysics trace and compile with Reactant?
#
#   julia --project=~/droplet-tasks-env droplets_reactant.jl level [Nd]
#
# Levels (run one per process to isolate memory use):
#   build     build the model on ReactantState
#   condense  the droplet condensation kernel alone (_condense_droplets!)
#   update    microphysics_model_update! (condensation, apply, deposit, update_state!)
#   advect    one RK stage of Lagrangian particle advection
#   step0     time_step! of the same box without droplets (baseline)
#   step      time_step! with droplets
# Each level prints the StableHLO size (lines, number of ops) before compiling.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Grids: halo_size
using Oceananigans.Models.LagrangianParticleTracking: advect_lagrangian_particles!, particle_stage
using CUDA
using Reactant
using Random
using Printf

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
level = ARGS[1]
Nd = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 100
const PM = Breeze.ParticleMicrophysics

T₀, p₀, L = 282.0, 85000.0, 0.02
fixed = get(ENV, "FIXED", "0")
physics = fixed == "0" ? DropletPhysicsParameters(Float64) :
          DropletPhysicsParameters(Float64; solver=Breeze.Solvers.FixedIterations(parse(Int, fixed)))
println("droplet solver: ", summary(physics.solver))

function droplet_box(arch; Nd=100, M=8, droplets_on=true)
    # DYNAMICS=compressible avoids the eager Boussinesq pressure-solver kernel compile at construction,
    # whose MLIR optimization exhausts memory on ReactantState (see REACTANT_DYNAMIC_PARTICLES.md)
    compressible = get(ENV, "DYNAMICS", "boussinesq") == "compressible"
    topology = compressible ? (Periodic, Periodic, Bounded) : (Periodic, Periodic, Periodic)
    grid = RectilinearGrid(arch; size=(M, M, M), x=(0, L), y=(0, L), z=(0, L), topology)
    dynamics = compressible ? CompressibleDynamics() : BoussinesqDynamics(grid; pressure=p₀, temperature=T₀)
    rng = Xoshiro(1)
    droplets = (x = L .* rand(rng, Nd), y = L .* rand(rng, Nd), z = L .* rand(rng, Nd),
                r = 5e-6 .+ 5e-6 .* rand(rng, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    closure = ScalarDiffusivity(; ν=1.6e-5, κ=(ρθ=2.22e-5, ρqᵛ=2.55e-5))
    microphysics = droplets_on ? LagrangianMicrophysics(; droplets, physics) : nothing
    return AtmosphereModel(grid; dynamics, closure, microphysics)
end

function condense!(model, Δt)
    μ = model.microphysics
    grid = model.grid
    droplets = μ.droplets.properties
    ρ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
    p = Breeze.AtmosphereModels.dynamics_pressure(model.dynamics)
    C = model.microphysical_fields.C
    launch!(architecture(grid), grid, KernelParameters(1:length(droplets)), PM._condense_droplets!, parent(C),
            halo_size(grid), droplets, grid, Δt, model.temperature, ρ, model.moisture_density, p,
            model.thermodynamic_constants, μ.physics, nothing)
    return nothing
end

function update!(model, Δt)
    model.clock.last_Δt = Δt
    Breeze.AtmosphereModels.microphysics_model_update!(model.microphysics, model)
    return nothing
end

function advect!(model, Δt)
    advect_lagrangian_particles!(model.microphysics.droplets, model, Δt, particle_stage(model.timestepper, 1, true))
    return nothing
end

# Single-droplet physics in a bare kernel over particle arrays (no grid)
const KA = Oceananigans.Architectures.KA
KA.@kernel function _rate!(Φ, r, rᵈ, κ, T, p, 𝒮, constants, physics)
    n = KA.@index(Global)
    @inbounds Φ[n] = PM.diffusional_growth_rate(r[n], rᵈ[n], κ[n], 𝒮, 1.0, T, p, constants, physics)
end
KA.@kernel function _growth!(r, rᵈ, κ, T, p, 𝒮, Δt, constants, physics)
    n = KA.@index(Global)
    @inbounds r[n] = PM.condensational_growth_step(r[n], rᵈ[n], κ[n], 𝒮, 1.0, T, p, Δt, constants, physics).radius
end
function rate!(Φ, r, rᵈ, κ)
    _rate!(KA.get_backend(r))(Φ, r, rᵈ, κ, T₀, p₀, 0.01, ThermodynamicConstants(Float64), physics; ndrange=length(r))
    return nothing
end
function growth!(r, rᵈ, κ)
    _growth!(KA.get_backend(r))(r, rᵈ, κ, T₀, p₀, 0.01, 0.01, ThermodynamicConstants(Float64), physics; ndrange=length(r))
    return nothing
end

function hlo_stats(f, args...)
    t = @elapsed hlo = repr(Reactant.@code_hlo optimize=false f(args...))
    ops = [m.match for m in eachmatch(r"(stablehlo|enzymexla|func|llvm|scf|affine|arith|memref)\.[a-z_\.]+", hlo)]
    counts = sort([(k, count(==(k), ops)) for k in unique(ops)], by=last, rev=true)
    @printf("trace %.1f s, unoptimized MLIR: %d lines, %d ops; top: %s\n", t, count(==('\n'), hlo), length(ops),
            join(["$k×$n" for (k, n) in first(counts, 8)], ", "))
end

function run_level(f, args...; raise=false)
    println("tracing ", f); flush(stdout)
    hlo_stats(f, args...)
    flush(stdout)
    GC.gc()
    t = @elapsed compiled = Reactant.@compile raise=raise f(args...)
    @printf("compiled %s in %.1f s (raise=%s), Sys.maxrss = %.2f GB\n", f, t, raise, Sys.maxrss() / 1e9)
    compiled(args...)
    println("ran ", f)
end

if level in ("rate", "growth")
    rng = Xoshiro(1)
    r, rᵈ, κ = Reactant.to_rarray.((5e-6 .+ 5e-6 .* rand(rng, Nd), fill(5e-8, Nd), fill(0.6, Nd)))
    try
        level == "rate" ? run_level(rate!, similar(r), r, rᵈ, κ) : run_level(growth!, r, rᵈ, κ)
    catch err
        println("FAILED ", level, ": ", first(sprint(showerror, err), 3000))
    end
    exit()
end

t = @elapsed model = droplet_box(ReactantState(); Nd, droplets_on = level != "step0")
@printf("built model (Nd = %d) in %.1f s\n", Nd, t); flush(stdout)
level == "build" && exit()
println(summary(model.microphysics))

Δt = 0.01
raise = get(ENV, "RAISE", "false") == "true"
try
    if level == "condense"
        run_level(condense!, model, Δt; raise)
    elseif level == "update"
        run_level(update!, model, Δt; raise)
    elseif level == "advect"
        run_level(advect!, model, Δt; raise)
    elseif level in ("step", "step0")
        run_level(time_step!, model, Δt; raise)
    end
catch err
    println("FAILED ", level, ": ", first(sprint(showerror, err), 3000))
    bt = catch_backtrace()
    st = stacktrace(bt)
    for fr in first(filter(fr -> occursin(r"Breeze|Oceananigans|LagrangianParticle", string(fr.file)), st), 15)
        println("  ", fr)
    end
end
