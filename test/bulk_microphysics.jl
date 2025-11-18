using Breeze
using Test

@testset "BulkMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
    # Test default construction
    bμp_default = BulkMicrophysics(FT)
    @test bμp_default.nucleation isa SaturationAdjustment
    @test bμp_default.categories === nothing
    @test bμp_default isa BulkMicrophysics{<:SaturationAdjustment, Nothing}

    # Test construction with explicit clouds scheme
    nucleation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    bμp_warm = BulkMicrophysics(; nucleation)
    @test bμp_warm.nucleation === nucleation
    @test bμp_warm.categories === nothing

    # Test construction with mixed-phase equilibrium
    nucleation_mixed = SaturationAdjustment(; equilibrium=MixedPhaseEquilibrium(FT))
    bμp_mixed = BulkMicrophysics(; nucleation=nucleation_mixed)
    @test bμp_mixed.nucleation === nucleation_mixed
    @test bμp_mixed.categories === nothing
end

