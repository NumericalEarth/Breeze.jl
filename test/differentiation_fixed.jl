#####
##### Integration tests for Reactant/Enzyme automatic differentiation, where the number of steps is fixed
#####

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Breeze
using Reactant
using Enzyme
using Statistics: mean
using Test
using CUDA

Reactant.set_default_backend("cpu")

@testset "Reactant/Enzyme differentiation with SSPRungeKutta3" begin
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
    Reactant.MLIR.IR.DUMP_MLIR_DIR[] = "/Users/danielkz/Aeolus2/Breeze.jl/mlir_dump/differentiation_fixed"
    grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1000, 1000),
                           halo=(3, 3), topology=(Periodic, Periodic, Flat))
    model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    dmodel = Enzyme.make_zero(model)

    θ_init = CenterField(grid)
    set!(θ_init, (x, y) -> 0.01 * x + 0.01 * y)
    dθ_init = CenterField(grid)
    set!(dθ_init, 0.0)

    function loss(model, θ_init, Δt)
        set!(model, θ=θ_init, ρ=1.0)
        @trace mincut=true checkpointing=false track_numbers=false for i in 1:1
            time_step!(model, Δt)
        end
        return mean(interior(model.temperature).^2)
    end

    function grad_loss(model, dmodel, θ_init, dθ_init, Δt)
        parent(dθ_init) .= 0
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(θ_init, dθ_init),
            Enzyme.Const(Δt))
        return dθ_init, loss_value
    end

    Δt = 0.01

    @testset "Compilation succeeds" begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, θ_init, dθ_init, Δt)
        @test compiled !== nothing
    end

    # @testset "Gradient computation" begin
    #     compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
    #         model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #     dθ, loss_val = compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)

    #     @test loss_val > 0
    #     @test !isnan(loss_val)
    #     @test maximum(abs, interior(dθ)) > 0  # Gradient should be non-zero
    #     @test !any(isnan, interior(dθ))
    # end
end
