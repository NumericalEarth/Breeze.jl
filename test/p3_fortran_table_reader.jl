using Test
using Oceananigans
using Oceananigans.Utils: TabulatedFunction
using Breeze.Microphysics.PredictedParticleProperties

@testset "FortranTabulatedFunction5D rime density transform" begin
    # Create a 5D table that returns its 4th argument (rime density index)
    identity_4th = (x1, x2, x3, x4, x5) -> Float64(x4)
    table = TabulatedFunction(identity_4th, CPU(), Float64;
                              range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 5.0), (0.0, 20.0)),
                              points=(2, 2, 2, 5, 2))
    wrapped = FortranTabulatedFunction5D(table)

    # Physical rime densities should map to Fortran indices
    # rho=50 -> index 1, rho=250 -> index 2, rho=450 -> index 3,
    # rho=650 -> index 4, rho=900 -> index 5
    @test wrapped(0.5, 0.5, 0.5, 50.0, 10.0) ≈ 1.0
    @test wrapped(0.5, 0.5, 0.5, 250.0, 10.0) ≈ 2.0
    @test wrapped(0.5, 0.5, 0.5, 450.0, 10.0) ≈ 3.0
    @test wrapped(0.5, 0.5, 0.5, 650.0, 10.0) ≈ 4.0
    @test wrapped(0.5, 0.5, 0.5, 900.0, 10.0) ≈ 5.0
    # Intermediate value
    @test wrapped(0.5, 0.5, 0.5, 150.0, 10.0) ≈ 1.5
end
