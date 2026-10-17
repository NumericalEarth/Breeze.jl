include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using CUDA
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function profile_objective!(field, profile, values)
    profile.values .= values
    set!(field, profile, 0.5)
    return sum(abs2, interior(field))
end

function profile_gradient!(field, dfield, profile, dprofile, values, dvalues)
    Enzyme.remake_zero!(dfield)
    Enzyme.remake_zero!(dprofile)
    dvalues .= 0
    _, value = Enzyme.autodiff(Enzyme.ReverseWithPrimal, profile_objective!, Enzyme.Active,
                              Enzyme.Duplicated(field, dfield), Enzyme.Duplicated(profile, dprofile),
                              Enzyme.Duplicated(values, dvalues))
    return value, dvalues
end

@testset "Batched time-height profile gradients" begin
    grid = RectilinearGrid(ReactantState(), Float64; size=3, z=[0, 1, 3, 5],
                           topology=(Flat, Flat, Bounded))
    field = CenterField(grid)
    values = [0.0 2; 1 3; 9 11; 25 27]
    profile = TimeHeightProfile([0.0, 1, 3, 5], [0.0, 1], values; architecture=ReactantState())
    controls = Reactant.to_rarray(values)
    dcontrols = zero(controls)
    dfield, dprofile = Enzyme.make_zero(field), Enzyme.make_zero(profile)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true profile_gradient!(
        field, dfield, profile, dprofile, controls, dcontrols)
    # Queries are at z = 0.5, 2, 4. At t=0.5 each interpolated value
    # is one above the t=0 column, with equal weights in both times.
    expected = repeat([0.75, 3.75, 12, 9], 1, 2)
    for repetition in 1:2
        # Reusing the executable must clear the previous adjoints.
        value, gradient = compiled(field, dfield, profile, dprofile, controls, dcontrols)
        @test Float64(value) ≈ sum(abs2, [1.5, 6, 18])
        @test Array(gradient) ≈ expected
    end
end
