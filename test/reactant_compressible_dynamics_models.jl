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
using Enzyme
using Statistics: mean
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

    function make_init_fields(grid)
        FT = eltype(grid)
        θ_init = CenterField(grid)
        set!(θ_init, (args...) -> FT(300))
        dθ_init = CenterField(grid)
        set!(dθ_init, FT(0))
        return θ_init, dθ_init
    end

    function loss(model, θ_init, Δt, nsteps)
        FT = eltype(model.grid)
        set!(model; θ=θ_init, ρ=one(FT))
        @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
            time_step!(model, Δt)
        end
        return mean(interior(model.temperature).^2)
    end

    function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
        parent(dθ_init) .= 0
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(θ_init, dθ_init),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps))
        return dθ_init, loss_value
    end

    @testset "3D (Periodic, Periodic, Flat) — CompressibleDynamics" begin
        @info "  Testing 3D (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)
        set!(model; θ=FT(300), ρ=FT(1))

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = FT(0.001)
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Compiled backward pass (Enzyme)" begin
            @info "    Compiling and running backward pass..."
            Δt = FT(0.001)
            nsteps = 4
            dmodel = Enzyme.make_zero(model)
            θ_init, dθ_init = make_init_fields(grid)
            compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test compiled_grad !== nothing

            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test loss_val > 0
            @test isfinite(loss_val)
            @test maximum(abs, interior(dθ)) > 0
            @test !any(isnan, interior(dθ))
        end
    end

    @testset "3D (Bounded, Bounded, Flat) — CompressibleDynamics" begin
        @info "  Testing 3D (Bounded, Bounded, Flat)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1),
                               topology=(Bounded, Bounded, Flat))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)
        set!(model; θ=FT(300), ρ=FT(1))

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = FT(0.001)
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Compiled backward pass (Enzyme)" begin
            @info "    Compiling and running backward pass..."
            Δt = FT(0.001)
            nsteps = 4
            dmodel = Enzyme.make_zero(model)
            θ_init, dθ_init = make_init_fields(grid)
            compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test compiled_grad !== nothing

            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test loss_val > 0
            @test isfinite(loss_val)
            @test maximum(abs, interior(dθ)) > 0
            @test !any(isnan, interior(dθ))
        end
    end

    @testset "3D (Periodic, Periodic, Periodic) — CompressibleDynamics" begin
        @info "  Testing 3D (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)
        set!(model; θ=FT(300), ρ=FT(1))

        @testset "Construction" begin
            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = FT(0.001)
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Compiled backward pass (Enzyme)" begin
            @info "    Compiling and running backward pass..."
            Δt = FT(0.001)
            nsteps = 4
            dmodel = Enzyme.make_zero(model)
            θ_init, dθ_init = make_init_fields(grid)
            compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test compiled_grad !== nothing

            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test loss_val > 0
            @test isfinite(loss_val)
            @test maximum(abs, interior(dθ)) > 0
            @test !any(isnan, interior(dθ))
        end
    end

    # @testset "3D (Bounded, Bounded, Bounded) — CompressibleDynamics" begin
    #     @info "  Testing 3D (Bounded, Bounded, Bounded)..."
    #     grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
    #                            topology=(Bounded, Bounded, Bounded))
    #     model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    #     FT = eltype(grid)
    #     set!(model; θ=FT(300), ρ=FT(1))

    #     @testset "Construction" begin
    #         @test model isa AtmosphereModel
    #         @test model.grid.architecture isa ReactantState
    #         @test model.dynamics isa CompressibleDynamics
    #     end

    #     @testset "Compiled time_step!" begin
    #         @info "    Compiling and running time_step!..."
    #         Δt = FT(0.001)
    #         Nt = 4
    #         compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
    #         compiled_run!(model, Δt, Nt)
    #         @test model.clock.iteration == Nt
    #     end

    #     @testset "Compiled backward pass (Enzyme)" begin
    #         @info "    Compiling and running backward pass..."
    #         Δt = FT(0.001)
    #         nsteps = 4
    #         dmodel = Enzyme.make_zero(model)
    #         θ_init, dθ_init = make_init_fields(grid)
    #         compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    #             model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #         @test compiled_grad !== nothing

    #         dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #         @test loss_val > 0
    #         @test isfinite(loss_val)
    #         @test maximum(abs, interior(dθ)) > 0
    #         @test !any(isnan, interior(dθ))
    #     end
    # end

    # @testset "3D (Periodic, Periodic, Bounded) — CompressibleDynamics" begin
    #     @info "  Testing 3D (Periodic, Periodic, Bounded)..."
    #     grid = RectilinearGrid(ReactantState(); size=(2, 2, 2), extent=(1, 1, 1),
    #                            topology=(Periodic, Periodic, Bounded))
    #     @time "Constructing model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    #     FT = eltype(grid)

    #     @testset "Construction" begin
    #         @test model isa AtmosphereModel
    #         @test model.grid.architecture isa ReactantState
    #         @test model.dynamics isa CompressibleDynamics
    #     end

    #     @testset "Compiled time_step!" begin
    #         @info "    Compiling and running time_step!..."
    #         Δt = FT(0.001)
    #         Nt = 1
    #         @time "Compiling time_step!" compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
    #         @time "Running time_step!" compiled_run!(model, Δt, Nt)
    #         @test model.clock.iteration == Nt
    #     end

    #     @testset "Compiled backward pass (Enzyme)" begin
    #         @info "    Compiling and running backward pass..."
    #         Δt = FT(0.001)
    #         nsteps = 1
    #         dmodel = Enzyme.make_zero(model)
    #         θ_init, dθ_init = make_init_fields(grid)
    #         @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    #             model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #         @test compiled_grad !== nothing

    #         @time "Running grad_loss" dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #         @test loss_val > 0
    #         @test isfinite(loss_val)
    #         @test maximum(abs, interior(dθ)) > 0
    #         @test !any(isnan, interior(dθ))
    #     end
    # end
end
