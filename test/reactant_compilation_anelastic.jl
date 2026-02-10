#####
##### Tests for Reactant compilation of AnelasticDynamics models
##### (No differentiation - just forward compilation)
#####
# AnelasticDynamics requires Bounded z (FourierTridiagonalPoissonSolver).
# B.6.6 workaround: BreezeReactantExt overrides enforce_mass_conservation!
# with @jit to route the FFT pressure solve through Reactant compilation.
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
##### Test configurations (all require Bounded z for AnelasticDynamics)
#####

test_topologies = [
    # 2D cases (x-z plane)
    (topology = (Periodic, Flat, Bounded), size = (8, 8), extent = (1000.0, 1000.0), halo = (3, 3), name = "(Periodic, Flat, Bounded)"),

    # 3D cases
    (topology = (Periodic, Periodic, Bounded), size = (4, 4, 4), extent = (1000.0, 1000.0, 1000.0), halo = (3, 3, 3), name = "(Periodic, Periodic, Bounded)"),
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

@testset "Reactant AnelasticDynamics" begin
    @info "Testing Reactant AnelasticDynamics compilation..."

    for config in test_topologies
        @testset "$(config.name)" begin
            @info "  Testing $(config.name)..."

            grid = make_grid(ReactantState(), config)

            @testset "Construction" begin
                # B.6.6: requires BreezeReactantExt enforce_mass_conservation! override
                @time "Constructing model" model = AtmosphereModel(grid;
                    dynamics = AnelasticDynamics(ReferenceState(grid))
                )

                @test model.grid === grid
                @test model.dynamics isa AnelasticDynamics
                @test model.dynamics.pressure_anomaly isa Field

                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            @testset "Compiled time_step!" begin
                model = AtmosphereModel(grid;
                    dynamics = AnelasticDynamics(ReferenceState(grid))
                )

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
