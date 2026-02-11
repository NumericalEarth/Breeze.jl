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

test_grids = [
    # 1D cases
    (topology = (Periodic, Flat, Flat), size = (8,),       extent = (1000.0,),             halo = (5,),    name = "(Periodic, Flat, Flat)"),
    (topology = (Flat, Flat, Bounded), size = (8,),       extent = (1000.0,),             halo = (5,),    name = "(Flat, Flat, Bounded)"),

    # 2D cases (Flat in z)
    (topology = (Periodic, Periodic, Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (5, 5),    name = "(Periodic, Periodic, Flat)"),
    (topology = (Bounded,  Bounded,  Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (5, 5),    name = "(Bounded, Bounded, Flat)"),
    # TODO: Make mixed topologies work
    # (topology = (Periodic, Flat, Bounded), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (3, 3),    name = "(Periodic, Flat, Bounded)"),

    # 3D cases
    (topology = (Periodic, Periodic, Periodic), size = (8, 8, 8), extent = (1000.0, 1000.0, 1000.0), halo = (5, 5, 5), name = "(Periodic, Periodic, Periodic)"),
    (topology = (Periodic, Periodic, Bounded),  size = (8, 8, 8), extent = (1000.0, 1000.0, 1000.0), halo = (5, 5, 5), name = "(Periodic, Periodic, Bounded)"),
]

test_advection_schemes = [
    (scheme = Centered(order=2), name = "Centered(order=2)"),
]

# WENO schemes to test on reduced set of topologies (1D and 2D only)
weno_advection_schemes = [
    (scheme = WENO(order=5), name = "WENO(order=5)"),
    (scheme = WENO(order=9), name = "WENO(order=9)"),
    (scheme = WENO(order=5, bounds=(0, 1)), name = "WENO(order=5, bounds=(0, 1))"),
]

# Reduced grid set for WENO tests (1D and 2D only to keep test time reasonable)
weno_test_grids = [
    # 1D case
    (topology = (Flat, Flat, Bounded), size = (8,),       extent = (1000.0,),             halo = (5,),    name = "(Periodic, Flat, Flat)"),

    # 2D case
    (topology = (Periodic, Periodic, Flat), size = (8, 8),       extent = (1000.0, 1000.0),             halo = (5, 5),    name = "(Periodic, Periodic, Flat)"),
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
##### Tests grouped by topology
#####

@testset "Reactant CompressibleDynamics - Centered Advection" begin
    @info "Testing Reactant CompressibleDynamics compilation with Centered advection..."

    for grid_config in test_grids
        @testset "$(grid_config.name)" begin
            @info "  Testing $(grid_config.name)..."
            for scheme_config in test_advection_schemes
                @testset "$(scheme_config.name)" begin
                    @info "    Testing $(scheme_config.name)..."
                    # Build grid and model once per grid configuration
                    grid = make_grid(ReactantState(), grid_config)
                    model = AtmosphereModel(grid; dynamics = CompressibleDynamics(), advection = scheme_config.scheme)

                    @testset "Construction" begin
                        @test model.grid === grid
                        @test model.dynamics isa CompressibleDynamics

                        # Initialize with simple constant values
                        set!(model; θ = 300.0, ρ = 1.0)

                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end

                    @testset "Compiled time_step!" begin
                        @info "    Compiling time_step!..."
                        Δt = 0.01
                        nsteps = 2

                        compiled_run = Reactant.@compile sync=true run_time_steps!(model, Δt, nsteps)
                        @test compiled_run !== nothing

                        @info "    Running compiled time_step!..."
                        compiled_run(model, Δt, nsteps)

                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end
                end
            end
        end
    end
end

@testset "Reactant CompressibleDynamics - WENO Advection" begin
    @info "Testing Reactant CompressibleDynamics compilation with WENO advection (1D and 2D only)..."

    for grid_config in weno_test_grids
        @testset "$(grid_config.name)" begin
            @info "  Testing $(grid_config.name)..."
            for scheme_config in weno_advection_schemes
                @testset "$(scheme_config.name)" begin
                    @info "    Testing $(scheme_config.name)..."
                    # Build grid and model once per grid configuration
                    grid = make_grid(ReactantState(), grid_config)
                    model = AtmosphereModel(grid; dynamics = CompressibleDynamics(), advection = scheme_config.scheme)

                    @testset "Construction" begin
                        @test model.grid === grid
                        @test model.dynamics isa CompressibleDynamics

                        # Initialize with simple constant values
                        set!(model; θ = 300.0, ρ = 1.0)

                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end

                    @testset "Compiled time_step!" begin
                        @info "    Compiling time_step!..."
                        Δt = 0.01
                        nsteps = 2

                        compiled_run = Reactant.@compile sync=true run_time_steps!(model, Δt, nsteps)
                        @test compiled_run !== nothing

                        @info "    Running compiled time_step!..."
                        compiled_run(model, Δt, nsteps)

                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end
                end
            end
        end
    end
end
