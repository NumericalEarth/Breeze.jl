#####
##### Reactant CompressibleDynamics tests
#####
# Tests construction and compiled time-stepping of AtmosphereModel
# with CompressibleDynamics on selected topologies including Bounded.
# Uses raise=true and raise_first=true to surface any MLIR compilation errors.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState, GPU
using Reactant
using Reactant: @trace
using CUDA
using Test

@testset "Reactant CompressibleDynamics" begin
    @info "Performing Reactant CompressibleDynamics tests..."


    Reactant.set_default_backend("cpu")
    
    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for _ in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    @testset "3D (Periodic, Periodic, Flat) — CompressibleDynamics" begin
        @info "  Testing 3D (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        set!(model; θ=300.0, ρ=1.0)

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end

    @testset "3D (Periodic, Periodic, Periodic) — CompressibleDynamics" begin
        @info "  Testing 3D (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        set!(model; θ=300.0, ρ=1.0)

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end

    @testset "3D (Periodic, Periodic, Bounded) — CompressibleDynamics" begin
        @info "  Testing 3D (Periodic, Periodic, Bounded)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        set!(model; θ=300.0, ρ=1.0)

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end

    @testset "3D (Bounded, Bounded, Bounded) — CompressibleDynamics" begin
        @info "  Testing 3D (Bounded, Bounded, Bounded)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Bounded, Bounded, Bounded))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        set!(model; θ=300.0, ρ=1.0)

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end
end
