#####
##### Tests for Reactant compilation of CompressibleDynamics models
##### (No differentiation - just forward compilation)
#####

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Breeze
using Reactant
using Test
using CUDA

Reactant.set_default_backend("cpu")

#####
##### Test configurations
#####

test_topologies = [
    # 2D cases (Flat in z)
    (topology = (Periodic, Periodic, Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (3, 3),    name = "(Periodic, Periodic, Flat)"),
    (topology = (Periodic, Bounded,  Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (3, 3),    name = "(Periodic, Bounded, Flat)"),
    (topology = (Bounded,  Bounded,  Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (3, 3),    name = "(Bounded, Bounded, Flat)"),

    # 3D cases
    (topology = (Periodic, Periodic, Periodic), size = (4, 4, 4), extent = (1000.0, 1000.0, 1000.0), halo = (3, 3, 3), name = "(Periodic, Periodic, Periodic)"),
    (topology = (Periodic, Periodic, Bounded),  size = (4, 4, 4), extent = (1000.0, 1000.0, 1000.0), halo = (3, 3, 3), name = "(Periodic, Periodic, Bounded)"),
]

#####
##### Helper functions
#####

function make_grid(arch, config)
    return RectilinearGrid(arch;
                           size = config.size,
                           extent = config.extent,
                           halo = config.halo,
                           topology = config.topology)
end

function run_time_steps!(model, Δt, nsteps)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return nothing
end

get_temperature(model) = Array(interior(model.temperature))

#####
##### Part (a): Model construction tests
#####

@testset "Reactant model construction" begin
    @info "Testing Reactant model construction..."
    for config in test_topologies
        @testset "$(config.name)" begin
            @info "  Testing $(config.name)..."
            grid = make_grid(ReactantState(), config)
            model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

            @test model.grid === grid
            @test model.dynamics isa CompressibleDynamics

            # Initialize with simple constant values
            set!(model; θ = 300.0, ρ = 1.0)

            T = get_temperature(model)
            @test all(isfinite, T)
            @test all(T .> 0)
        end
    end
end

#####
##### Part (b): Time step compilation tests
#####

@testset "Reactant time_step! compilation" begin
    @info "Testing Reactant time_step! compilation..."
    for config in test_topologies
        @testset "$(config.name)" begin
            @info "  Compiling time_step! for $(config.name)..."
            grid = make_grid(ReactantState(), config)
            model = AtmosphereModel(grid; dynamics = CompressibleDynamics())
            set!(model; θ = 300.0, ρ = 1.0)

            Δt = 0.01
            nsteps = 2

            compiled_run = Reactant.@compile sync=true run_time_steps!(model, Δt, nsteps)
            @test compiled_run !== nothing

            @info "  Running compiled time_step! for $(config.name)..."
            compiled_run(model, Δt, nsteps)

            T = get_temperature(model)
            @test all(isfinite, T)
            @test all(T .> 0)
        end
    end
end
