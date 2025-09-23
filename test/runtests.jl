include("runtests_setup.jl")

using Test
using Breeze
using Oceananigans

@testset "Breeze.jl" begin
    @testset "Atmosphere Thermodynamics" begin
        include("test_atmosphere_thermodynamics.jl")
    end

    @testset "Atmosphere Model Unit" begin
        include("test_atmosphere_model_unit.jl")
    end

    @testset "Moist Air Buoyancy" begin
        include("test_moist_air_buoyancy.jl")
    end

    @testset "Anelastic Pressure Solver" begin
        include("test_anelastic_pressure_solver.jl")
    end
end
