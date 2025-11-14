using Breeze
using Oceananigans
using Test

using Breeze.Microphysics:
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium

# BulkMicrophysics is not exported, so we access it via the module
const BulkMicrophysics = Breeze.Microphysics.BulkMicrophysics

@testset "BulkMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
    # Test default construction
    bμp_default = BulkMicrophysics(FT)
    @test bμp_default.clouds isa SaturationAdjustment
    @test bμp_default.precipitation === nothing
    @test bμp_default isa BulkMicrophysics{<:SaturationAdjustment, Nothing}

    # Test construction with explicit clouds scheme
    clouds = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    bμp_warm = BulkMicrophysics(FT, clouds)
    @test bμp_warm.clouds === clouds
    @test bμp_warm.precipitation === nothing

    # Test construction with mixed-phase equilibrium
    clouds_mixed = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    bμp_mixed = BulkMicrophysics(FT, clouds_mixed)
    @test bμp_mixed.clouds === clouds_mixed
    @test bμp_mixed.precipitation === nothing

    # Test construction with custom clouds and no precipitation
    bμp_custom = BulkMicrophysics(FT, clouds, nothing)
    @test bμp_custom.clouds === clouds
    @test bμp_custom.precipitation === nothing
end

