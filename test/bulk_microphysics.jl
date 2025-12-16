using Breeze
using Test

@testset "BulkMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
    # Test default construction
    bμp_default = BulkMicrophysics(FT)
    @test bμp_default.cloud_formation isa SaturationAdjustment
    @test bμp_default.categories === nothing
    @test bμp_default isa BulkMicrophysics{<:SaturationAdjustment, Nothing}

    # Test construction with explicit clouds scheme
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    bμp_warm = BulkMicrophysics(; cloud_formation)
    @test bμp_warm.cloud_formation === cloud_formation
    @test bμp_warm.categories === nothing

    # Test construction with mixed-phase equilibrium
    cloud_formation_mixed = SaturationAdjustment(; equilibrium=MixedPhaseEquilibrium(FT))
    bμp_mixed = BulkMicrophysics(; cloud_formation=cloud_formation_mixed)
    @test bμp_mixed.cloud_formation === cloud_formation_mixed
    @test bμp_mixed.categories === nothing
end

