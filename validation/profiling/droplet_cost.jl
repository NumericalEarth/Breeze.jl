# Cost of Lagrangian droplet microphysics per time step, in an L3-like setup (NCAR parcel DNS:
# forced turbulence in a rising periodic box, Centered(4) advection, constant diffusivities,
# Boussinesq rising frame with adiabatic expansion). Droplets are placed uniformly at random.
#
#   julia --project=~/droplet-tasks-env droplet_cost.jl "64:0,64:6000,128:1000000" [profile]
#
# Each entry M:Nd times time_step! at M³ with Nd droplets (Nd = -1: no microphysics at all; Nd = -2: Oceananigans NonhydrostaticModel).
# Env SORT=1 sorts droplets by cell every step (CountingSort).

using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, PlanarLiquidSurface
using Oceananigans
using Oceananigans.Models.LagrangianParticleTracking: CountingSort
using CUDA
using Printf
using Random

const L = 0.2 # m, roughly the L3 box

function build(M, Nd; arch=GPU(), sorting=nothing, Δt=1e-3)
    grid = RectilinearGrid(arch; size=(M, M, M), halo=(3, 3, 3), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))
    if Nd == -2 # Oceananigans baseline: NonhydrostaticModel with two tracers, no forcing
        closure = ScalarDiffusivity(; ν=1.6e-5, κ=(T=2.22e-5, q=2.55e-5))
        model = NonhydrostaticModel(grid; advection=Centered(order=4), closure, tracers=(:T, :q),
                                    timestepper=:RungeKutta3)
        set!(model; u=0.1 .* randn(Xoshiro(2), M, M, M), T=282, q=0.01)
        return model
    end
    T₀, p₀ = 282.0, 85000.0
    dynamics = BoussinesqDynamics(grid; pressure=p₀, temperature=T₀, frame_vertical_velocity=2.0)
    closure = ScalarDiffusivity(; ν=1.6e-5, κ=(ρθ=2.22e-5, ρqᵛ=2.55e-5))
    forcing = ConstantPowerBandForcing(grid; energy_injection_rate=0.05)
    forcing = (u=forcing, v=forcing, w=forcing)

    if Nd < 0
        microphysics = nothing
    else
        rng = Xoshiro(1)
        droplets = (x = L .* rand(rng, Nd), y = L .* rand(rng, Nd), z = L .* rand(rng, Nd),
                    r = 5e-6 .+ 5e-6 .* rand(rng, Nd), rᵈ = fill(5e-8, Nd), κ = fill(1.2, Nd))
        kw = sorting === nothing ? (;) : (; sorting)
        self_interaction = get(ENV, "SELF", "1") == "1" ? DiffusiveSelfInteraction() : nothing
        microphysics = LagrangianMicrophysics(; droplets, self_interaction, kw...,
                                              physics=DropletPhysicsParameters(Float64; terminal_velocity=StokesTerminalVelocity()))
    end

    model = AtmosphereModel(grid; dynamics, microphysics, closure, advection=Centered(order=4), forcing)
    constants = model.thermodynamic_constants
    pᵛ = 1.01 * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    q₀ = ε * pᵛ / (p₀ - (1 - ε) * pᵛ)
    rng = Xoshiro(2)
    u₀ = 0.1 .* randn(rng, M, M, M)
    v₀ = 0.1 .* randn(rng, M, M, M)
    set!(model; T=T₀, qᵗ=q₀, u=u₀, v=v₀)
    return model
end

function time_steps!(model, n; Δt=1e-3)
    CUDA.synchronize()
    t = @elapsed begin
        for _ in 1:n
            time_step!(model, Δt)
        end
        CUDA.synchronize()
    end
    return t / n
end

cases = [Tuple(parse.(Int, split(c, ':'))) for c in split(ARGS[1], ',')]
profile = length(ARGS) ≥ 2 && ARGS[2] == "profile"
sorting = get(ENV, "SORT", "0") == "1" ? CountingSort() : nothing
println("sorting = ", sorting)

base = Dict{Int, Float64}()
for (M, Nd) in cases
    model = build(M, Nd; sorting)
    for _ in 1:3; time_step!(model, 1e-3); end # compile
    n = M ≥ 256 ? 10 : 30
    t = time_steps!(model, n)
    a = CUDA.@allocated time_step!(model, 1e-3)
    Nd == 0 && (base[M] = t)
    per = Nd > 0 && haskey(base, M) ? 1e9 * (t - base[M]) / Nd : NaN
    @printf("RESULT M=%d Nd=%d: %.3f ms/step  (%.2f ns/droplet/step above Nd=0)  gpu-alloc=%d B\n",
            M, Nd, 1e3t, per, a)
    if profile
        println("PROFILE M=$M Nd=$Nd")
        res = CUDA.@profile trace=false begin
            for _ in 1:5; time_step!(model, 1e-3); end
        end
        mkpath(joinpath(@__DIR__, "output"))
        file = joinpath(@__DIR__, "output", "profile_M$(M)_Nd$(Nd)_sort$(sorting !== nothing)$(get(ENV, "LABEL", "")).txt")
        open(file, "w") do io
            show(IOContext(io, :displaysize => (1000, 400)), res)
        end
        println("profile written to ", file)
    end
    # locality drift: run more steps then re-time (unsorted particles diffuse away from initial order;
    # here initial order is already random, so this checks sorting benefit over time)
    model = nothing
    GC.gc(true); CUDA.reclaim()
end
