include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: clip_max, clip_min, clip_below, clip_between, clip_negative,
                                      stable_damping, hard_floor, third_moment_bound, bound_third_moment
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

    @testset "SmoothClipping" begin
        @test_throws ArgumentError SmoothClipping(FT; width = 0)
        @test_throws ArgumentError SmoothClipping(-one(FT))
        @test SmoothClipping(FT(0.1)) === SmoothClipping(FT; width = 0.1)

        hard = HardClipping()
        finite = filter(isfinite, values)
        for width in FT[1e-1, 1e-2, 1e-3]
            policy = SmoothClipping(width)
            δ = width * scale
            for a in finite, b in finite
                # One-sided, and within δ / 2 of the hard bound (to round-off)
                roundoff = 4eps(FT) * (abs(a) + abs(b))
                @test clip_max(policy, a, b, scale) ≥ max(a, b) - roundoff
                @test clip_min(policy, a, b, scale) ≤ min(a, b) + roundoff
                @test clip_below(policy, a, b, scale) ≥ max(a, b) - roundoff
                @test clip_max(policy, a, b, scale) - max(a, b) ≤ δ / 2 + roundoff
                @test min(a, b) - clip_min(policy, a, b, scale) ≤ δ / 2 + roundoff
                @test clip_max(policy, a, b, scale) isa FT
            end
            for x in finite
                @test FT(0.01) ≤ clip_between(policy, x, FT(0.01), FT(0.99), 1) ≤ FT(0.99)
                @test -1 ≤ clip_between(policy, x, -1, 1, 1) ≤ 1
                @test abs(clip_between(policy, x, -1, 1, 1) - clamp(x, -1, 1)) ≤ δ
                @test clip_negative(policy, x, scale) ≥ 0
                @test stable_damping(policy, FT(2), x, scale) ≥ 2 * max(x, 0) - 8eps(FT) * abs(x)
            end
            # Strictly inside the bounds where the argument is within a few bounds of them
            @test FT(0.01) < clip_between(policy, FT(-1), FT(0.01), FT(0.99), 1) < clip_between(policy, FT(2), FT(0.01), FT(0.99), 1) < FT(0.99)
            @test clip_negative(policy, -scale, scale) > 0
        end

        # The width → 0 limit is the hard bound
        narrow = SmoothClipping(sqrt(eps(FT)))
        for a in finite, b in finite
            tolerance = sqrt(eps(FT)) * (1 + abs(a) + abs(b))
            @test isapprox(clip_max(narrow, a, b, scale), max(a, b); atol = tolerance)
            @test isapprox(clip_min(narrow, a, b, scale), min(a, b); atol = tolerance)
        end

        # The smooth bound is differentiable at the bound, where the hard one has a kink:
        # the one-sided slopes of clip_max(0, x) at x = 0 are both 1/2, against 0 and 1
        if FT == Float64
            policy = SmoothClipping(FT(0.1))
            h = FT(1e-6)
            below = (clip_max(policy, 0, zero(FT), scale) - clip_max(policy, 0, -h, scale)) / h
            above = (clip_max(policy, 0, h, scale) - clip_max(policy, 0, zero(FT), scale)) / h
            @test isapprox(below, 0.5; atol = 1e-4)
            @test isapprox(above, 0.5; atol = 1e-4)
            @test (clip_max(hard, 0, zero(FT), scale) - clip_max(hard, 0, -h, scale)) / h == 0
        end
    end
end

# A convective boundary layer with moisture and wind, which reaches cloud within the test
function shoc_column_model(closure; FT = eltype(closure.constants.ggr), Nz = 24)
    grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
    θᵢ(z) = 300 + 3e-3 * z
    qᵢ(z) = 0.016 * exp(-z / 2000)
    reference_state = ReferenceState(grid, ThermodynamicConstants(FT); base_pressure=101500, potential_temperature=θᵢ)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), closure)
    set!(model, θ = (args...) -> θᵢ(last(args)), qᵗ = (args...) -> qᵢ(last(args)), u = 5)
    set!(model.tracers.ρe, reference_state.density * FT(0.1))
    return model
end

function run_shoc_column(closure; steps = 20, Δt = 60)
    model = shoc_column_model(closure)
    for _ in 1:steps
        time_step!(model, Δt)
    end
    ρ = Array(interior(model.dynamics.reference_state.density))
    diagnostics = model.closure_fields.diagnostics
    return (θ = Array(interior(model.formulation.potential_temperature_density)) ./ ρ,
            q = Array(interior(model.moisture_density)) ./ ρ,
            e = Array(interior(model.tracers.ρe)) ./ ρ,
            cloud_fraction = Array(interior(diagnostics.shoc_cldfrac)),
            cloud_liquid = Array(interior(diagnostics.shoc_ql)),
            w3 = Array(interior(diagnostics.w3)),
            tkh = Array(interior(diagnostics.tkh)))
end

all_finite(run) = all(field -> all(isfinite, field), run)

surface_forcing = (surface_temperature_flux = 0.1, surface_moisture_flux = 5e-5, friction_velocity = 0.3)

@testset "SHOC component swaps in a column [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    original = run_shoc_column(SHOC(OriginalSHOC(), FT; surface_forcing...))
    @test all_finite(original)

    @testset "SmoothClipping" begin
        closure = SHOC(OriginalSHOC(), FT; clipping = SmoothClipping(FT; width = 1e-2), surface_forcing...)
        @test isbits(closure)
        @test closure_parameters(closure) == closure_parameters(SHOC(OriginalSHOC(), FT))
        smooth = run_shoc_column(closure)
        @test all_finite(smooth)
        @test all(smooth.e .> closure.constants.mintke)
        @test all(x -> 0 ≤ x ≤ 1, smooth.cloud_fraction)
        @test all(≥(0), smooth.cloud_liquid)
        # Smoothing the bounds is a small change of the solution
        @test isapprox(smooth.θ, original.θ; rtol = 1e-4)
        @test isapprox(smooth.e, original.e; rtol = 0.2)

        # With a narrower smoothing the solution is closer to that of the hard bounds (in single
        # precision both differences are a few units of round-off of θ)
        if FT == Float64
            narrower = run_shoc_column(SHOC(OriginalSHOC(), FT; clipping = SmoothClipping(FT; width = 1e-4), surface_forcing...))
            @test maximum(abs, narrower.θ .- original.θ) < maximum(abs, smooth.θ .- original.θ)
        end
    end

    @testset "Bounds of the third moment" begin
        hard = HardClipping()
        reset = CanutoThirdMoment(FT)
        clamped = CanutoThirdMoment(FT; bound = ClampBound())
        smooth = CanutoThirdMoment(FT; bound = TanhBound())
        @test summary(clamped) == "CanutoThirdMoment{$FT} with ClampBound"
        @test closure_parameters(SHOC(OriginalSHOC(), FT; third_moment = smooth)).ℂʷ₂ == FT(7)

        w_sec = FT(0.5)
        bound = reset.w3clip * sqrt(2 * w_sec^3)
        @test third_moment_bound(reset, w_sec) ≈ bound

        for w3 in FT[-10, -1, -0.5, -1e-3, 0, 1e-3, 0.5, 1, 10] .* bound
            inside = abs(w3) ≤ bound
            # E3SM resets a third moment beyond the bound to 0.02
            @test bound_third_moment(reset, hard, w_sec, w3) == (inside ? w3 : FT(0.02))
            @test bound_third_moment(clamped, hard, w_sec, w3) == clamp(w3, -bound, bound)
            @test abs(bound_third_moment(clamped, SmoothClipping(FT(0.1)), w_sec, w3)) ≤ bound
            @test abs(bound_third_moment(smooth, hard, w_sec, w3)) ≤ bound
            @test sign(bound_third_moment(smooth, hard, w_sec, w3)) == sign(w3)
        end
        # Well inside the bound the smooth bounds change little
        @test isapprox(bound_third_moment(smooth, hard, w_sec, bound / 100), bound / 100; rtol = 1e-4)
        @test bound_third_moment(smooth, hard, w_sec, zero(FT)) == 0

        for third_moment in (clamped, smooth)
            run = run_shoc_column(SHOC(OriginalSHOC(), FT; third_moment, surface_forcing...))
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test isapprox(run.θ, original.θ; rtol = 1e-4)
        end
    end
end
