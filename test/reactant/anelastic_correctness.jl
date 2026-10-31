include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant correctness — anelastic AtmosphereModel parity over a compiled
##### `run!`. Construction and `set!` solve for the anelastic pressure eagerly
##### (`enforce_mass_conservation!`), which exercises Reactant's FFT plans on
##### concrete arrays; the compiled `run!` exercises them on traced arrays.
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Periodic, Bounded
using Reactant
using Printf: @printf
using Test
using CUDA

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

function compare_interior(name, ψ₁, ψ₂; rtol, atol)
    a₁ = Array(interior(ψ₁))
    a₂ = Array(interior(ψ₂))
    max_δ = maximum(abs, a₁ .- a₂)
    ok = isapprox(a₁, a₂; rtol, atol)
    @printf("  %-12s ok=%-5s  max|ψ₁|=%.6e  max|δ|=%.6e\n", string(name), ok, maximum(abs, a₁), max_δ)
    return ok
end

function compare_models(vmodel, rmodel; rtol, atol)
    ok = true
    Ψv = Oceananigans.fields(vmodel)
    Ψr = Oceananigans.fields(rmodel)
    for name in keys(Ψv)
        ok &= compare_interior(name, Ψv[name], Ψr[name]; rtol, atol)
    end
    ok &= compare_interior(:p′, vmodel.dynamics.pressure_anomaly, rmodel.dynamics.pressure_anomaly; rtol, atol)
    return ok
end

function anelastic_model(arch)
    grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5), extent=(1e3, 1e3, 1e3),
                           topology=(Periodic, Periodic, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=1e5, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    return AtmosphereModel(grid; dynamics, advection=WENO(order=5))
end

@testset "Reactant anelastic correctness" begin
    Δt = 1.0
    stop_iteration = 3
    rtol = 1e-8
    atol = sqrt(eps(Float64))

    vmodel = anelastic_model(default_arch)
    rmodel = anelastic_model(ReactantState())
    @test rmodel.grid.architecture isa ReactantState

    Nx, Ny, Nz = size(vmodel.grid)
    u_init = randn(size(vmodel.velocities.u)...)
    v_init = randn(size(vmodel.velocities.v)...)
    θ_init = 300 .+ 0.1 .* randn(Nx, Ny, Nz)

    # `set!` projects the velocity onto the anelastic constraint with a pressure solve
    set!(vmodel; u=u_init, v=v_init, θ=θ_init)
    set!(rmodel; u=u_init, v=v_init, θ=θ_init)
    @test compare_models(vmodel, rmodel; rtol, atol)

    vsimulation = Simulation(vmodel; Δt, stop_iteration, verbose=false)
    run!(vsimulation)

    rsimulation = Simulation(rmodel; Δt, stop_iteration, verbose=false)
    r_run! = Reactant.@compile raise=true sync=true run!(rsimulation)
    r_run!(rsimulation)

    @test iteration(rsimulation) == stop_iteration
    @test compare_models(vmodel, rmodel; rtol, atol)
end
