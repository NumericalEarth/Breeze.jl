include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Breeze.Advection: ScaledWENOWeights
using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: biased_weno_weights, weno_differences
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using CUDA
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

@inline function first_weno_weight(i, j, k, grid, scheme, values)
    stencil = ntuple(r -> @inbounds(values[r]), Val(5))
    return biased_weno_weights(weno_differences(scheme, stencil), grid, scheme)[1]
end

function weno_objective!(field, scheme, values)
    field .= KernelFunctionOperation{Center, Center, Center}(first_weno_weight, field.grid, scheme, values)
    return sum(abs2, interior(field))
end

function weno_gradient!(field, dfield, scheme, values, dvalues)
    Enzyme.remake_zero!(dfield)
    dvalues .= 0
    _, value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                              weno_objective!, Enzyme.Active,
                              Enzyme.Duplicated(field, dfield), Enzyme.Const(scheme),
                              Enzyme.Duplicated(values, dvalues))
    return value, dvalues
end

@testset "Scaled WENO weights and gradients" begin
    grid = RectilinearGrid(ReactantState(), Float64; size=2, z=(0, 1), topology=(Flat, Flat, Bounded))
    field = CenterField(grid)
    scheme = WENO(Float64; order=5, weight_computation=ScaledWENOWeights)
    stencil = [1.0, 2, 6, 3, 8]
    controls = Reactant.to_rarray(stencil)
    dcontrols, dfield = zero(controls), Enzyme.make_zero(field)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true weno_gradient!(
        field, dfield, scheme, controls, dcontrols)
    value, derivative = compiled(field, dfield, scheme, controls, dcontrols)
    gradient = Array(derivative)
    cpu_grid = RectilinearGrid(CPU(), Float64; size=2, z=(0, 1), topology=(Flat, Flat, Bounded))
    objective(values) = 2first_weno_weight(1, 1, 1, cpu_grid, scheme, values)^2
    direction = [0.2, -0.3, 0.5, -0.1, 0.4]
    increment = 1e-5
    numerical = (objective(stencil .+ increment .* direction) -
                 objective(stencil .- increment .* direction)) / (2increment)
    @test Float64(value) ≈ objective(stencil)
    @test all(isfinite, gradient) && any(!iszero, gradient)
    @test sum(gradient .* direction) ≈ numerical rtol=1e-6
end
