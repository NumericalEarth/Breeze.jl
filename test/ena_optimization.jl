include(joinpath(@__DIR__, "setup.jl"))
using Test
include(joinpath(@__DIR__, "..", "examples", "setups", "ena_controls.jl"))
include(joinpath(@__DIR__, "..", "examples", "setups", "ena_observations.jl"))

@testset "ENA inverse observation operator" begin
    height = [50, 150, 250, 350]
    values = [280.0 0.01 2.0 -1.0; 282.0 0.012 3.0 -2.0;
              284.0 0.014 4.0 -3.0; 286.0 0.016 5.0 -4.0]
    valid = trues(4, 4)
    valid[1, 2] = false
    values[1, 2] = NaN
    sounding = (; height, time=[0.0, 0.5, 1.0, 1.5], values, valid)
    target = ena_sounding_target([0, 200, 400]; sounding, scales=(1, 1, 1, 1))
    @test target.values[:, 1] == [281, 285]
    @test target.values[:, 2] ≈ [0.012, 0.015]
    @test target.times[:, 1] == [0.25, 1.25]
    @test target.times[:, 2] == [0.5, 1.25]
    @test all(isfinite, target.values)
    @test sum(target.weights) == 1
    sampling = ena_sampling_weights(target, 0.5, 3)
    @test dropdims(sum(sampling; dims=3); dims=3) ≈ ones(2, 4)
    @test dropdims(sum(sampling .* reshape(collect(0:0.5:1.5), 1, 1, :); dims=3); dims=3) ≈ target.times
    @test_throws ArgumentError ena_sampling_weights(target, 0.5, 2)
    @test_throws ArgumentError ena_sampling_weights(target, 0, 3)
    @test_throws ArgumentError ena_sounding_target([0, 200, 400]; sounding, scales=(0, 1, 1, 1))
    @test_throws ArgumentError ena_sounding_target([0, 400, 200]; sounding)
    @test ena_fit_loss(target.values, target, zeros(3), 0.01) == 0
    @test ena_fit_loss(target.values .+ 1, target, zeros(3), 0.01) ≈ 0.5
    @test ena_fit_loss(target.values, target, ones(3), 0.01) ≈ 0.005
    basis = ena_vertical_basis(collect(0:100:3000))
    @test all(basis .≥ 0)
    @test vec(sum(basis; dims=2)) ≈ ones(31)
end

@testset "ENA bounded optimizer and numerical gradient reference" begin
    value(p) = sum(abs2, p .- [0.5, -0.25]) / 2
    gradient(p) = (value(p), p .- [0.5, -0.25])
    @test ena_central_difference(value, [0.0, 0.0]) ≈ [-0.5, 0.25]
    result = ena_optimize(gradient, value, [0.0, 0.0]; iterations=3)
    @test result.controls ≈ [0.5, -0.25]
    @test result.history[end].loss < result.history[1].loss
    bounded = ena_optimize(gradient, value, [0.0, 0.0]; bound=0.1)
    @test bounded.controls ≈ [0.1, -0.1]
    @test_throws ErrorException ena_optimize(p -> (NaN, p), value, zeros(2))
end
