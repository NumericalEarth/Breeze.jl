include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.TurbulenceClosures.SHOC: HardClipping, clip_max, clip_min, clip_below, clip_between, clip_negative,
                                      stable_damping, hard_floor
using Test

# Bitwise equality, with NaN equal to NaN and the two zeros distinct
same_bits(a, b) = a === b

@testset "SHOC clipping policies [$FT]" for FT in test_float_types()
    values = FT[-Inf, -100, -1, -floatmin(FT), -0.0, 0.0, floatmin(FT), 4e-4, 1, 100, 101, Inf, NaN]
    scale = one(FT)

    @testset "HardClipping is the idiom of each call site" begin
        policy = HardClipping()
        for a in values, b in values
            @test same_bits(clip_max(policy, a, b, scale), max(a, b))
            @test same_bits(clip_min(policy, a, b, scale), min(a, b))
            @test same_bits(clip_below(policy, a, b, scale), ifelse(a < b, b, a))
            @test same_bits(hard_floor(a, b), ifelse(a < b, b, a))
            @test same_bits(stable_damping(policy, a, b, scale), ifelse(b <= 0, zero(FT), a) * b)
        end

        for x in values
            @test same_bits(clip_negative(policy, x, scale), ifelse(x >= 0, x, zero(FT)))
            @test same_bits(clip_between(policy, x, -1, 1, 1), max(-1, min(1, x)))
            # The two sites that nest the other way in shoc.F90 have the same value
            @test same_bits(clip_between(policy, x, 0, 100, 1), min(100, max(0, x)))
            @test same_bits(clip_between(policy, x, FT(0.01), FT(0.99), 1), max(FT(0.01), min(x, FT(0.99))))
        end

        # An integer bound keeps the type of the clipped value
        @test clip_max(policy, 0, FT(-1), scale) isa FT
        @test clip_between(policy, FT(2), 0, 100, 1) isa FT
    end
end
