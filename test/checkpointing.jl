include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

const NSPIN = 6
const NRESTART = 4
const Δt = 0.4

function test_grid(FT)
    Oceananigans.defaults.FloatType = FT
    return RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 1_000), y=(0, 1_000), z=(0, 1_000),
                           topology=(Periodic, Periodic, Bounded))
end

θᵢ(x, y, z) = 300 + 2 * sin(2π * x / 1_000) * cos(2π * y / 1_000) * exp(-z / 500)
uᵢ(x, y, z) = 0.5 * cos(2π * x / 1_000) * sin(2π * z / 1_000)

function anelastic_model(FT; microphysics=nothing, closure=nothing)
    grid = test_grid(FT)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, microphysics, closure)
    set!(model; θ=θᵢ, qᵗ=0.015, u=uᵢ)
    return model
end

function compressible_model(FT; time_discretization=ExplicitTimeStepping(), microphysics=nothing, closure=nothing)
    grid = test_grid(FT)
    dynamics = CompressibleDynamics(time_discretization; reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    set!(model; ρ=model.dynamics.reference_state.density, θ=θᵢ, qᵗ=0.015, u=uᵢ)
    return model
end

snapshot(model) = Dict(string(name) => Array(interior(field))
                       for (name, field) in pairs(Oceananigans.prognostic_fields(model)))

function run_continuously(build)
    model = build()
    simulation = Simulation(model; Δt, stop_iteration=NSPIN + NRESTART, verbose=false)
    run!(simulation)
    return snapshot(model)
end

function run_with_restart(build, dir)
    model = build()
    simulation = Simulation(model; Δt, stop_iteration=NSPIN, verbose=false)
    simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=IterationInterval(NSPIN), dir)
    run!(simulation)

    checkpoint = joinpath(dir, "checkpoint_iteration$(NSPIN).jld2")
    @test isfile(checkpoint)

    model = build()
    simulation = Simulation(model; Δt, stop_iteration=NSPIN + NRESTART, verbose=false)
    run!(simulation, pickup=checkpoint)
    return snapshot(model)
end

function test_bitwise_restart(build)
    continuous = run_continuously(build)
    restarted = mktempdir(dir -> run_with_restart(build, dir))

    @test issetequal(keys(continuous), keys(restarted))
    for name in sort(collect(keys(continuous)))
        @testset let field = name
            @test maximum(abs, continuous[name] - restarted[name]) == 0
        end
    end
end

@testset "Checkpoint restart [$FT]" for FT in test_float_types()
    @testset "Anelastic dynamics" begin
        @testset "no microphysics, no closure" begin
            test_bitwise_restart(() -> anelastic_model(FT))
        end

        @testset "equilibrium microphysics" begin
            test_bitwise_restart(() -> anelastic_model(FT; microphysics=SaturationAdjustment()))
        end

        @testset "equilibrium microphysics with precipitation" begin
            test_bitwise_restart(() -> anelastic_model(FT; microphysics=InstantaneousPrecipitation()))
        end

        @testset "non-equilibrium microphysics, two species" begin
            test_bitwise_restart(() -> anelastic_model(FT; microphysics=OneMomentCloudMicrophysics()))
        end

        @testset "non-equilibrium microphysics, five species" begin
            test_bitwise_restart(() -> anelastic_model(FT; microphysics=TwoMomentCloudMicrophysics()))
        end

        @testset "Smagorinsky closure" begin
            test_bitwise_restart(() -> anelastic_model(FT; closure=SmagorinskyLilly()))
        end

        @testset "prognostic TKE closure" begin
            test_bitwise_restart(() -> anelastic_model(FT; closure=TKEBasedTurbulenceClosure()))
        end
    end

    @testset "Compressible dynamics" begin
        @testset "explicit, no microphysics, no closure" begin
            test_bitwise_restart(() -> compressible_model(FT))
        end

        @testset "split-explicit, no microphysics, no closure" begin
            test_bitwise_restart(() -> compressible_model(FT; time_discretization=SplitExplicitTimeDiscretization()))
        end

        @testset "non-equilibrium microphysics, two species" begin
            test_bitwise_restart(() -> compressible_model(FT; microphysics=OneMomentCloudMicrophysics()))
        end

        @testset "prognostic TKE closure" begin
            test_bitwise_restart(() -> compressible_model(FT; closure=TKEBasedTurbulenceClosure()))
        end
    end
end
