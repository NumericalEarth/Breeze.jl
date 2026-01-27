#####
##### MWE: DelinearizeIndexingPass segfault with Reactant/Enzyme autodiff
#####
#
# This is the simplest possible reproducer for the segfault in
# DelinearizeIndexingPass::runOnOperation() when compiling autodiff.
#
# The crash occurs during MLIR pass execution, not at runtime.
#
# To dump MLIR for debugging:
#   Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#   Reactant.MLIR.IR.DUMP_MLIR_DIR[] = "/path/to/dump"
#

using Reactant
using Enzyme
using Statistics: mean
using Test

Reactant.set_default_backend("cpu")

# Uncomment to dump MLIR for debugging:
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = "/Users/danielkz/Aeolus2/Breeze.jl/mlir_dump/reactant_autodiff_mwe"

@testset "Minimal Reactant/Enzyme autodiff MWE" begin

    @testset "Simple array operations" begin
        # Create ConcreteRArray (Reactant's traced array type)
        x = Reactant.ConcreteRArray(rand(4, 4))
        dx = Reactant.ConcreteRArray(zeros(4, 4))

        function simple_loss(x)
            return sum(x.^2)
        end

        function grad_simple(x, dx)
            Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                simple_loss, Enzyme.Active,
                Enzyme.Duplicated(x, dx))
            return dx
        end

        @testset "Basic autodiff compiles" begin
            compiled = Reactant.@compile raise_first=true raise=true sync=true grad_simple(x, dx)
            @test compiled !== nothing
        end
    end

    @testset "With traced loop" begin
        x = Reactant.ConcreteRArray(rand(4, 4))
        dx = Reactant.ConcreteRArray(zeros(4, 4))

        function loss_with_loop(x, nsteps)
            result = x
            @trace for i in 1:nsteps
                result = result .+ x .* 0.1
            end
            return sum(result.^2)
        end

        function grad_with_loop(x, dx, nsteps)
            Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss_with_loop, Enzyme.Active,
                Enzyme.Duplicated(x, dx),
                Enzyme.Const(nsteps))
            return dx
        end

        @testset "Autodiff with traced loop compiles" begin
            compiled = Reactant.@compile raise_first=true raise=true sync=true grad_with_loop(
                x, dx, 2)
            @test compiled !== nothing
        end
    end

    @testset "With multiple arrays and reshape" begin
        # This pattern is closer to what Oceananigans does
        x = Reactant.ConcreteRArray(rand(4, 4, 1))
        y = Reactant.ConcreteRArray(rand(4, 4, 1))
        dx = Reactant.ConcreteRArray(zeros(4, 4, 1))
        dy = Reactant.ConcreteRArray(zeros(4, 4, 1))

        function multi_array_loss(x, y)
            # Operations that might create complex indexing patterns
            z = x .+ y
            w = z .* x
            return sum(w.^2)
        end

        function grad_multi(x, y, dx, dy)
            Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                multi_array_loss, Enzyme.Active,
                Enzyme.Duplicated(x, dx),
                Enzyme.Duplicated(y, dy))
            return dx, dy
        end

        @testset "Multi-array autodiff compiles" begin
            compiled = Reactant.@compile raise_first=true raise=true sync=true grad_multi(
                x, y, dx, dy)
            @test compiled !== nothing
        end
    end

    @testset "With slicing operations" begin
        # Slicing can create linearized indexing patterns
        x = Reactant.ConcreteRArray(rand(10, 10, 1))
        dx = Reactant.ConcreteRArray(zeros(10, 10, 1))

        function sliced_loss(x)
            # Interior slice (like Oceananigans' interior())
            interior = x[4:7, 4:7, 1:1]
            return sum(interior.^2)
        end

        function grad_sliced(x, dx)
            Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                sliced_loss, Enzyme.Active,
                Enzyme.Duplicated(x, dx))
            return dx
        end

        @testset "Sliced autodiff compiles" begin
            compiled = Reactant.@compile raise_first=true raise=true sync=true grad_sliced(x, dx)
            @test compiled !== nothing
        end
    end
end
