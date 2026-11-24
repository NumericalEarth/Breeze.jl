include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Grids: znodes
using Breeze.TurbulenceClosures: convert_eltype, column_stability_field, diffusivity_stability_functionsᶜᶜᶠ,
                                 dissipation_stability_functionᶜᶜᶜ, shoc_distribution
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: trainable_parameter_names, is_calibratable, component_scalar_names,
                                      clip_max, clip_min, clip_below, clip_between, clip_negative,
                                      stable_damping, hard_floor, third_moment_bound, limit_third_moment, third_moment,
                                      bound_stratification_number, assumed_pdf, plume_saturation, plume_liquid_water_temperature, plume_variance,
                                      smooth_scalar_plumes, isotropy_ratio, shoc_dissipation_coefficient
using Breeze.Thermodynamics: VaporEquivalentExponent, WarmPhaseEquilibrium
using Enzyme: Enzyme
using Test

# Bitwise equality, with NaN equal to NaN and the two zeros distinct
same_bits(a, b) = a === b


# Bitwise equality, with NaN equal to NaN and the two zeros distinct
same_bits(a, b) = a === b

# The limited third moment as a function of one scalar, for derivatives: of the diagnosed third
# moment, of the vertical velocity variance, of `maximum_skewness`, and of the coefficient `c` of the diagnosis
limited_by_third_moment(w3, limiter, w_sec) = limit_third_moment(limiter, w_sec, w3)
limited_by_variance(w_sec, limiter, w3) = limit_third_moment(limiter, w_sec, w3)
limited_by_maximum_skewness(maximum_skewness, w_sec, w3) =
    limit_third_moment(PolynomialSaturatedSkewness(maximum_skewness), w_sec, w3)
limited_by_coefficient(c, limiter, g, arguments) =
    limit_third_moment(limiter, arguments[3], third_moment(AlgebraicThirdMoment(Float64; ℂʷ₂ = c), g, arguments...))

# The derivative of `f(x, parameters...)` with respect to `x` by reverse-mode automatic differentiation
derivative(f, x, parameters...) =
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Active(x), map(Enzyme.Const, parameters)...)[1][1]

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

        # The smooth bound has a continuous slope at the bound, where the hard one has a kink:
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

@testset "SHOC components of the subgrid distribution [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    @testset "Third moment and skewness limiters" begin
        hard = HardClipping()
        g = FT(9.80616) # the gravitational acceleration of shoc.F90 (`ggr`)
        diagnosis = AlgebraicThirdMoment(FT)
        reset = ResetAboveThresholdSkewness(FT)
        clamped = ClampedSkewness(FT)
        polynomial = PolynomialSaturatedSkewness(FT)
        @test summary(polynomial) == "PolynomialSaturatedSkewness{$FT}"
        @test closure_parameters(SimplifiedHigherOrderClosure(FT; subgrid_distribution = shoc_distribution(FT; skewness = reset))).ℂʷ₂ == FT(7)
        @test SimplifiedHigherOrderClosure(FT).subgrid_distribution.skewness === polynomial
        @test_throws ArgumentError ResetAboveThresholdSkewness(FT; clip = 1)

        # The diagnosis and the limiter of shoc.F90 with its literals written out. The diagnosis with
        # its default fields returns the same bits. The limiter writes the bound as
        # maximum_skewness (w'w')^{3/2} where shoc.F90 writes 1.2 √(2 w'w'³): the two bounds agree to
        # round-off, and the limited third moments are equal but where the third moment is within
        # round-off of the bound
        function literal_third_moment(ggr, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi, thedz, thedz2,
                                      thl_sec_diff, wthl_sec_diff, wsec_diff, tke_diff, wthl)
            c = FT(7)
            iso = isotropy_zi
            isosqrd = iso * iso
            buoy_sgs2 = isosqrd * brunt_zi
            bet2 = ggr / thetal_zi
            f0 = thedz2 * (bet2 * bet2 * bet2) * ((iso * iso) * (iso * iso)) * wthl * thl_sec_diff
            f1 = thedz2 * (bet2 * bet2) * (iso * iso * iso) * (wthl * wthl_sec_diff + FT(0.5) * w_sec_zi * thl_sec_diff)
            f2 = thedz * bet2 * isosqrd * wthl * wsec_diff + 2 * thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff
            f3 = thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff + thedz * bet2 * isosqrd * (wthl * tke_diff)
            f4 = thedz * iso * w_sec_zi * (wsec_diff + tke_diff)
            f5 = thedz * iso * w_sec_zi * wsec_diff
            a4 = FT(2.4) / (3 * c + 5)
            a5 = FT(0.6) / (c * (3 + 5 * c))
            omega0 = a4 / (1 - a5 * buoy_sgs2)
            omega1 = omega0 / (2 * c)
            omega2 = omega1 * f3 + (FT(5) / 4) * omega0 * f4
            a0 = (FT(0.52) * (1 / (c * c))) / (c - 2)
            a1 = FT(0.87) / (c * c)
            a2 = FT(0.5) / c
            a3 = FT(0.6) / (c * (c - 2))
            x0 = (a2 * buoy_sgs2 * (1 - a3 * buoy_sgs2)) / (1 - (a1 + a3) * buoy_sgs2)
            y0 = (2 * a2 * buoy_sgs2 * x0) / (1 - a3 * buoy_sgs2)
            x1 = (a0 * f0 + a1 * f1 + a2 * (1 - a3 * buoy_sgs2) * f2) / (1 - (a1 + a3) * buoy_sgs2)
            y1 = (2 * a2 * (buoy_sgs2 * x1 + (a0 / a1) * f0 + f1)) / (1 - a3 * buoy_sgs2)
            aa0 = omega0 * x0 + omega1 * y0
            aa1 = omega0 * x1 + omega1 * y1 + omega2
            return (aa1 - FT(1.2) * x1 - FT(1.5) * f5) / (c - FT(1.2) * x0 + aa0)
        end

        function literal_limiter(theterm, w3)
            cond = FT(1.2) * sqrt(2 * (theterm * theterm * theterm))
            tsign = ifelse(w3 < 0, -one(FT), one(FT))
            return ifelse(tsign * w3 > cond, FT(0.02), w3)
        end

        # Deterministic pseudo-random inputs over the ranges the column visits, across the poles
        # of the diagnosis (b = τ² N² up to several hundred) and across the bound of the limiter
        fraction(n, m) = FT(mod(n * 0.6180339887498949 + m * 0.7548776662466927, 1))
        nreset = 0
        for n in 1:2000
            τ = FT(10)^(1 + FT(2.5) * fraction(n, 1))
            N² = (fraction(n, 2) - FT(0.4)) * FT(4e-4)
            w_sec = FT(4e-4) + 2 * fraction(n, 3)^2
            θˡ = 280 + 40 * fraction(n, 4)
            thedz = 1 / (10 + 90 * fraction(n, 5))
            thedz2 = thedz / 2
            arguments = (τ, N², w_sec, θˡ, thedz, thedz2, (fraction(n, 6) - FT(0.5)) * FT(0.1),
                         (fraction(n, 7) - FT(0.5)) * FT(0.02), (fraction(n, 8) - FT(0.5)) * FT(0.5),
                         (fraction(n, 9) - FT(0.5)) * FT(0.5), (fraction(n, 10) - FT(0.5)) * FT(0.1))
            w3 = third_moment(diagnosis, g, arguments...)
            @test same_bits(w3, literal_third_moment(g, arguments...))
            literal_bound = FT(1.2) * sqrt(2 * (w_sec * w_sec * w_sec))
            @test isapprox(third_moment_bound(reset, w_sec), literal_bound; rtol = 8 * eps(FT))
            if abs(abs(w3) - literal_bound) > 8 * eps(FT) * literal_bound
                @test same_bits(limit_third_moment(reset, w_sec, w3), literal_limiter(w_sec, w3))
            end
            nreset += limit_third_moment(reset, w_sec, w3) != w3
        end
        @test 0 < nreset < 2000 # the inputs are on both sides of the bound

        w_sec = FT(0.5)
        bound = third_moment_bound(reset, w_sec)
        @test reset.maximum_skewness ≈ FT(1.2) * sqrt(FT(2))
        @test bound ≈ reset.maximum_skewness * w_sec^FT(1.5)
        @test bound ≈ FT(1.2) * sqrt(2 * w_sec^3) # as shoc.F90 writes it
        for limiter in (reset, clamped, polynomial)
            @test third_moment_bound(limiter, w_sec) ≈ bound
        end

        for w3 in FT[-10, -1.875, -1, -0.5, -1e-3, 0, 1e-3, 0.5, 1, 1.875, 10] .* bound
            inside = abs(w3) ≤ bound
            # E3SM resets a third moment beyond the bound to 0.02
            @test limit_third_moment(reset, w_sec, w3) == (inside ? w3 : FT(0.02))
            @test limit_third_moment(clamped, w_sec, w3) == clamp(w3, -bound, bound)
            @test abs(limit_third_moment(polynomial, w_sec, w3)) ≤ bound
            @test sign(limit_third_moment(polynomial, w_sec, w3)) == sign(w3)
            @test limit_third_moment(polynomial, w_sec, -w3) == -limit_third_moment(polynomial, w_sec, w3)
        end

        # Well inside the bound the polynomial changes little: less than a hyperbolic tangent would
        @test isapprox(limit_third_moment(polynomial, w_sec, bound / 100), bound / 100; rtol = 1e-4)
        @test isapprox(limit_third_moment(polynomial, w_sec, bound / 2), bound / 2; rtol = 0.05)
        @test limit_third_moment(polynomial, w_sec, bound) > bound * tanh(one(FT))
        @test limit_third_moment(polynomial, w_sec, zero(FT)) == 0

        # The polynomial is the bound from 15/8 of it on, and increasing before
        @test limit_third_moment(polynomial, w_sec, FT(1.875) * bound) ≈ bound
        @test limit_third_moment(polynomial, w_sec, 100 * bound) ≈ bound
        samples = range(-2 * bound, 2 * bound, length = 401)
        limited = [limit_third_moment(polynomial, w_sec, FT(w3)) for w3 in samples]
        @test issorted(limited)

        # Without vertical velocity variance the bound is zero and so is the third moment
        @test limit_third_moment(polynomial, zero(FT), FT(1)) == 0
        @test limit_third_moment(polynomial, -FT(1), FT(1)) == 0

        if FT == Float64
            # The slope of the polynomial limiter against central differences, on both sides of the
            # bound at which shoc.F90 resets the third moment (ratio 1) and of the point where the
            # polynomial reaches the bound (ratio 15/8): the exact slope is (1 - u²)², u = 8/15 ratio
            h = 1e-6
            for ratio in (-2.5, -1.2, -1, -0.3, 0.0, 0.3, 0.9, 1 - 1e-9, 1.0, 1 + 1e-9, 1.1, 1.5, 1.8, 1.874, 1.876, 2.5)
                w3 = ratio * bound
                difference = (limit_third_moment(polynomial, w_sec, w3 + h * bound) -
                              limit_third_moment(polynomial, w_sec, w3 - h * bound)) / (2 * h * bound)
                u = clamp(8 * ratio / 15, -1, 1)
                @test isapprox(difference, (1 - u^2)^2; atol = 1e-6)

                gradient = derivative(limited_by_third_moment, w3, polynomial, w_sec)
                @test isapprox(gradient, difference; atol = 1e-6)

                # The limiter of shoc.F90 jumps at ratio 1; this one does not
                if abs(ratio) > 0.5
                    jump = limit_third_moment(polynomial, w_sec, w3 * (1 + h)) - limit_third_moment(polynomial, w_sec, w3 * (1 - h))
                    @test abs(jump) ≤ 2 * h * abs(w3)
                end
            end
            @test abs(limit_third_moment(reset, w_sec, bound * (1 + h)) - limit_third_moment(reset, w_sec, bound * (1 - h))) > bound / 2

            # The slope with respect to the variance and to maximum_skewness, through the bound
            for ratio in (0.3, 1.1, 1.8, 2.5)
                w3 = ratio * bound
                for (func, x, parameters) in ((limited_by_variance, w_sec, (polynomial, w3)),
                                              (limited_by_maximum_skewness, polynomial.maximum_skewness, (w_sec, w3)))
                    difference = (func(x * (1 + h), parameters...) - func(x * (1 - h), parameters...)) / (2 * h * x)
                    gradient = derivative(func, x, parameters...)
                    @test isapprox(gradient, difference; rtol = 1e-6, atol = 1e-9)
                end
            end

            # The diagnosis and the limiter together, with respect to the coefficient c, at a state
            # whose diagnosed third moment is beyond the bound of shoc.F90
            arguments = (300.0, 1e-4, 0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            raw = third_moment(diagnosis, g, arguments...)
            for ratio in (0.9, 1.2, 1.6)
                limiter = PolynomialSaturatedSkewness(abs(raw) / (ratio * 0.4^1.5))
                @test abs(raw) / third_moment_bound(limiter, 0.4) ≈ ratio
                parameters = (limiter, g, arguments)
                difference = (limited_by_coefficient(7 + 1e-5, parameters...) - limited_by_coefficient(7 - 1e-5, parameters...)) / 2e-5
                gradient = derivative(limited_by_coefficient, 7.0, parameters...)
                @test gradient != 0
                @test isapprox(gradient, difference; rtol = 1e-6)
            end
        end

        end
    end

    @testset "Stratification number bound" begin
        g = FT(9.80616) # the gravitational acceleration of shoc.F90 (`ggr`)
        diagnosis = AlgebraicThirdMoment(FT)
        unbounded = UnboundedStratificationNumber()
        rational = RationalSaturatedStratificationNumber(FT)
        @test summary(rational) == "RationalSaturatedStratificationNumber{$FT}"
        @test rational.pole_fraction == FT(1 / 2)
        @test SimplifiedHigherOrderClosure(FT).subgrid_distribution.stratification_number_bound === rational

        # The first pole of the diagnosis, 1 / (a₁ + a₃), from its coefficients
        c = FT(7)
        first_pole = 1 / (FT(0.87) / c^2 + FT(0.6) / (c * (c - 2)))
        @test isapprox(first_pole, FT(28.655); rtol = 1e-4)
        αᴺᵐᵃˣ = first_pole / 2

        # Unbounded: the identity, and the diagnosis with it is the diagnosis of shoc.F90
        fraction(n, m) = FT(mod(n * 0.6180339887498949 + m * 0.7548776662466927, 1))
        for n in 1:500
            αᴺ = (fraction(n, 1) - FT(0.3)) * FT(10)^(3 * fraction(n, 2))
            @test same_bits(bound_stratification_number(unbounded, αᴺ, first_pole), αᴺ)

            τ = FT(10)^(1 + FT(2.5) * fraction(n, 1))
            N² = (fraction(n, 2) - FT(0.4)) * FT(4e-4)
            arguments = (τ, N², FT(0.4), FT(300), FT(1 / 40), FT(1 / 80), FT(0.02), FT(-0.004), FT(0.1), FT(0.15), FT(0.03))
            @test same_bits(third_moment(diagnosis, unbounded, g, arguments...),
                            third_moment(diagnosis, g, arguments...))
        end

        # Rational: the identity where αᴺ ≤ 0; increasing, and below αᴺᵐᵃˣ = γ × the first pole
        # for every αᴺ > 0, so that 1 - (a₁ + a₃) αᴺ ≥ 1 - γ
        # (to round-off where αᴺ ≫ αᴺᵐᵃˣ, where the bounded value rounds to αᴺᵐᵃˣ)
        samples = sort(vcat(FT.(-50:0.5:0), FT(10) .^ range(FT(-6), FT(30); length = 400), first_pole))
        bounded = [bound_stratification_number(rational, αᴺ, first_pole) for αᴺ in samples]
        @test all(k -> bounded[k] == samples[k], findall(≤(0), samples))
        @test all(≤(αᴺᵐᵃˣ * (1 + eps(FT))), bounded)
        @test all(k -> bounded[k] < αᴺᵐᵃˣ, findall(αᴺ -> 0 < αᴺ < 1000 * αᴺᵐᵃˣ, samples))
        @test all(αᴺ -> 1 - αᴺ / first_pole ≥ FT(1 / 2) - 2 * eps(FT), bounded)
        @test issorted(bounded[samples .< 1000 * αᴺᵐᵃˣ])
        @test all(k -> bounded[k + 1] ≥ bounded[k] * (1 - 2 * eps(FT)), 1:length(bounded) - 1)
        @test isapprox(bounded[end], αᴺᵐᵃˣ; rtol = 4 * eps(FT))
        @test isapprox(bound_stratification_number(rational, αᴺᵐᵃˣ, first_pole), αᴺᵐᵃˣ / 2)

        # At and beyond the first pole the unbounded diagnosis diverges; the bounded one does not
        τ = FT(300)
        for N² in (first_pole / τ^2, (1 / (FT(0.6) / (c * (c - 2)))) / τ^2, FT(1e-2))
            arguments = (τ, N², FT(0.4), FT(300), FT(1 / 40), FT(1 / 80), FT(0.02), FT(-0.004), FT(0.1), FT(0.15), FT(0.03))
            @test isfinite(third_moment(diagnosis, rational, g, arguments...))
        end
        # The first singularity of the unbounded diagnosis is where its denominator vanishes, at
        # αᴺ ≈ 24.26 (0.85 of the first pole, where the poles of X₀ and X₁ cancel in the ratio):
        # there the unbounded third moment is orders of magnitude larger than the bounded one
        near_zero = (τ, FT(24.2589579) / τ^2, FT(0.4), FT(300), FT(1 / 40), FT(1 / 80), FT(0.02), FT(-0.004), FT(0.1), FT(0.15), FT(0.03))
        @test abs(third_moment(diagnosis, unbounded, g, near_zero...)) > 100 * abs(third_moment(diagnosis, rational, g, near_zero...))

        if FT == Float64
            # The slope of the bound, 1 / (1 + αᴺ / αᴺᵐᵃˣ)² for αᴺ > 0 and 1 for αᴺ ≤ 0, against
            # central differences and Enzyme; continuous through αᴺ = 0
            bounded_by_number(αᴺ, bound, first_pole) = bound_stratification_number(bound, αᴺ, first_pole)
            h = 1e-6
            for αᴺ in (-30.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 5.0, αᴺᵐᵃˣ, first_pole, 100.0, 1e4)
                exact = αᴺ > 0 ? 1 / (1 + αᴺ / αᴺᵐᵃˣ)^2 : 1.0
                step = h * max(abs(αᴺ), 1)
                difference = (bounded_by_number(αᴺ + step, rational, first_pole) - bounded_by_number(αᴺ - step, rational, first_pole)) / (2 * step)
                @test isapprox(difference, exact; rtol = 1e-6, atol = 1e-9)
                @test isapprox(derivative(bounded_by_number, αᴺ, rational, first_pole), exact; rtol = 1e-12)
                @test derivative(bounded_by_number, αᴺ, unbounded, first_pole) == 1
            end

            # The diagnosed third moment with respect to N², to the pole fraction γ and to the
            # coefficient c, which moves the first pole and so the bound: Enzyme against central
            # differences, in weak and in strong stratification (αᴺ beyond the first pole)
            moment_by_N²(N², bound, g, arguments) = third_moment(diagnosis, bound, g, arguments[1], N², arguments[2:end]...)
            moment_by_fraction(γ, g, arguments) = third_moment(diagnosis, RationalSaturatedStratificationNumber(Float64; pole_fraction = γ), g, arguments...)
            moment_by_c(c, g, arguments) = third_moment(AlgebraicThirdMoment(Float64; ℂʷ₂ = c), rational, g, arguments...)
            rest = (0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            for αᴺ in (-5.0, 0.5, 5.0, 20.0, 60.0)
                N² = αᴺ / τ^2
                arguments = (τ, N², rest...)
                difference = (moment_by_N²(N² * (1 + h), rational, g, (τ, rest...)) -
                              moment_by_N²(N² * (1 - h), rational, g, (τ, rest...))) / (2 * h * N²)
                gradient = derivative(moment_by_N², N², rational, g, (τ, rest...))
                @test isapprox(gradient, difference; rtol = 1e-5)
                for (func, x) in ((moment_by_fraction, 0.5), (moment_by_c, 7.0))
                    difference = (func(x * (1 + h), g, arguments) - func(x * (1 - h), g, arguments)) / (2 * h * x)
                    gradient = derivative(func, x, g, arguments)
                    @test isapprox(gradient, difference; rtol = 1e-5, atol = 1e-12)
                end
            end
        end

    end

    @testset "SmoothDoubleGaussianPDF" begin
        hard = HardClipping()
        original_pdf = DoubleGaussianPDF(FT)
        smooth_pdf = SmoothDoubleGaussianPDF(FT; clipping = hard)
        @test SimplifiedHigherOrderClosure(FT; subgrid_distribution = shoc_distribution(FT; assumed_pdf = smooth_pdf)) isa
              SimplifiedHigherOrderClosure

        # One level at 900 hPa and 285 K, close to saturation, between interfaces that carry the
        # same moments (w3, thl_sec, wthl_sec, qwthl_sec, wqw_sec, qw_sec)
        pressure, θˡ = FT(9e4), FT(285) * (FT(1e5) / FT(9e4))^(FT(287.042) / FT(1004.64))
        thermodynamics = (constants = ThermodynamicConstants(FT), standard_pressure = FT(1e5), exner_exponent = VaporEquivalentExponent(),
                          phase_equilibrium = WarmPhaseEquilibrium())
        # Saturation of the level's air with its total water as vapor: a fixed point in qᵗ
        qᵛ⁺ = zero(FT)
        for _ in 1:4
            Tˡ = plume_liquid_water_temperature(thermodynamics, θˡ, qᵛ⁺, pressure)
            qᵛ⁺ = first(plume_saturation(thermodynamics, Tˡ, qᵛ⁺, pressure))
        end
        pdf_of(pdf, qᵗ, w_sec, moments) =
            assumed_pdf(pdf, thermodynamics, FT(1020), FT(980), FT(1000), pressure, θˡ, zero(FT), qᵗ,
                        w_sec, moments, moments)

        # Away from the switches (resolved variances, plumes well apart) the smooth PDF is the
        # original one as its tolerances vanish, and within a few percent of it with the
        # tolerances of shoc.F90, which are a sixth of the standard deviation of total water here
        sharp_pdf = SmoothDoubleGaussianPDF(FT; clipping = hard, thl_tol = 1e-5, rt_tol = 1e-7, w_tol = 1e-4, minimum_saturation_spread = 1e-9)
        w_sec = FT(0.6)
        for w3 in FT[-0.2, 0.05, 0.4], qᵗ in (FT(0.97), FT(1.0), FT(1.03)) .* qᵛ⁺
            moments = (w3, FT(0.25), FT(-0.25), FT(-1.3e-4), FT(2.9e-4), FT(4e-7))
            a = pdf_of(original_pdf, qᵗ, w_sec, moments)
            b = pdf_of(smooth_pdf, qᵗ, w_sec, moments)
            sharp = pdf_of(sharp_pdf, qᵗ, w_sec, moments)
            @test isapprox(b[1], a[1]; atol = 0.03)        # cloud fraction
            @test isapprox(b[2], a[2]; rtol = 0.05)        # liquid water
            @test isapprox(b[4], a[4]; rtol = 0.05)        # liquid water flux
            if FT == Float64
                @test isapprox(sharp[1], a[1]; atol = 1e-4)
                @test isapprox(sharp[2], a[2]; rtol = 1e-3)
                @test isapprox(sharp[4], a[4]; rtol = 1e-3)
                @test isapprox(sharp[5], a[5]; atol = 1e-4)
            end
        end

        # Calm and uniform air: the plumes collapse onto the mean, saturated air is cloud and
        # unsaturated air is clear, as with the switches
        still = ntuple(_ -> zero(FT), 6)
        calm = FT(2e-4)
        for pdf in (original_pdf, smooth_pdf)
            cloud_fraction, liquid, liquid_variance, liquid_flux, buoyancy_flux = pdf_of(pdf, FT(1.02) * qᵛ⁺, calm, still)
            @test isapprox(cloud_fraction, 1; atol = 1e-3)
            @test liquid > 0
            @test liquid_flux == 0
            @test first(pdf_of(pdf, FT(0.98) * qᵛ⁺, calm, still)) < 1e-3
        end
        @test isapprox(pdf_of(smooth_pdf, FT(1.02) * qᵛ⁺, calm, still)[2], pdf_of(original_pdf, FT(1.02) * qᵛ⁺, calm, still)[2]; rtol = 1e-2)

        # No jump where the switches of the original are: the outputs are continuous in the
        # vertical velocity variance across w_tol² and in the scalar variance across its tolerance²
        variable = (FT(1e-4), FT(0.02), FT(-2e-3), FT(-2e-6), FT(3e-6), FT(4e-8))
        step(pdf, f) = maximum(abs, f(pdf, 1 + FT(1e-3)) .- f(pdf, 1 - FT(1e-3)))
        across_calm(pdf, x) = pdf_of(pdf, qᵛ⁺, original_pdf.w_tol^2 * x, variable)[1:2]
        across_uniform(pdf, x) = pdf_of(pdf, qᵛ⁺, FT(0.3), (variable[1], variable[2], variable[3], variable[4], variable[5],
                                                         original_pdf.rt_tol^2 * x))[1:2]
        @test step(smooth_pdf, across_calm)[1] < 1e-3
        @test step(smooth_pdf, across_uniform)[1] < 1e-3

        # The in-plume variance ratio tends to one as the correlation with w vanishes
        w̃₁, w̃₂ = sqrt(FT(0.6)), -sqrt(FT(0.6))
        _, _, ratio₁, ratio₂ = smooth_scalar_plumes(smooth_pdf, hard, zero(FT), w̃₁, w̃₂, zero(FT), FT(0.5), true)
        @test ratio₁ ≈ 1 && ratio₂ ≈ 1
        # and is that of the original where the original is defined
        corr = FT(0.3)
        x̃₁, x̃₂, ratio₁, ratio₂ = smooth_scalar_plumes(smooth_pdf, hard, corr, w̃₁, w̃₂, zero(FT), FT(0.5), false)
        @test ratio₁ ≈ plume_variance(hard, true, x̃₁, x̃₂, FT(0.5), zero(FT), one(FT), FT(100))
        @test ratio₂ ≈ plume_variance(hard, false, x̃₁, x̃₂, FT(0.5), zero(FT), one(FT), FT(100))

        # A spread of the saturation deficit that does not vanish: the cloud fraction passes
        # through one half at saturation instead of jumping
        @test isapprox(first(pdf_of(smooth_pdf, qᵛ⁺, calm, still)), 0.5; atol = 0.05)

    end
end
#####
##### SHOC's return-to-isotropy time scale and diffusivities as stability functions, against shoc.F90
#####

# `isotropic_ts` and `eddy_diffusivities` of shoc.F90 (without the cold-surface fallback), with the dissipation
# a_diss = Cee / ℓ e^{3/2} of `adv_sgs_tke`
function fortran_isotropy_and_diffusivities(tke, brunt, brunt_int, mix; ggr = 9.80616)
    Cs, Ck = 0.15, 0.1
    Ce = Ck^3 / Cs^4
    Cee = Ce / 0.7 * 0.19 + Ce / 0.7 * 0.51
    a_diss = Cee / mix * tke^1.5
    lambda = 0.001 + ((brunt_int / ggr) - 0.02) * 2.65
    lambda = max(0.001, min(0.08, lambda))
    lambda = brunt <= 0 ? 0.0 : lambda
    tscale = (2 * tke) / a_diss
    isotropy = min(20000.0, tscale / (1 + lambda * brunt * tscale^2))
    return isotropy, 0.1 * isotropy * tke, 0.1 * isotropy * tke, Cee, lambda
end

@testset "ColumnStabilityDamping is the time scale and diffusivities of shoc.F90 [$FT]" for FT in test_float_types()
    sf = ColumnStabilityDamping(FT)
    @test summary(sf) == "ColumnStabilityDamping{$FT}"
    @test convert_eltype(Float32, sf).ℂᶜ === 0.1f0
    @test shoc_dissipation_coefficient(sf) ≈ fortran_isotropy_and_diffusivities(0.1, 0.0, 0.0, 100.0)[4] rtol = 1e-6
    @test column_stability_field(RectilinearGrid(size = 4, z = (0, 1), topology = (Flat, Flat, Bounded)), sf) isa Field

    rtol_round = FT == Float32 ? 1e-5 : 1e-12
    for tke in (0.01, 0.1, 1.0), brunt in (-1e-4, 0.0, 1e-5, 3e-4), brunt_int in (0.0, 0.25, 0.5, 2.0), mix in (30.0, 300.0)
        iso, tkh, tk, _, λ = fortran_isotropy_and_diffusivities(tke, brunt, brunt_int, mix)
        # shoc.F90 does not damp where N² ≤ 0; the port damps there with ℂᶜ² / 4
        λ = brunt <= 0 ? 0.001 : λ
        τ₀ = fortran_isotropy_and_diffusivities(tke, 0.0, brunt_int, mix)[1]
        iso = brunt < 0 ? τ₀ / (1 + 0.1^2 / 4 * abs(brunt) * τ₀^2) : iso
        tkh = tk = 0.1 * iso * tke
        s = isotropy_ratio(sf, FT(λ), FT(mix), sqrt(FT(tke)), FT(brunt))
        τ = s * mix / sqrt(tke)
        bound = (iso / 20000)^2 # the sum of inverse squares in place of min(maxiso, ⋅)
        @test abs(τ - iso) / iso ≤ bound + rtol_round
        @test abs(sf.ℂᶜ * s * mix * sqrt(tke) - tkh) / tkh ≤ bound + rtol_round
        @test abs(sf.ℂᵘ * s * mix * sqrt(tke) - tk) / tk ≤ bound + rtol_round
    end

    # The stratification bounds the time scale whatever the dissipation ratio ℂᴰ₂ / ℂᴰ₁: τ |N| ≤ 1 / (2 √λ) in stable
    # air, and in unstable air τ |N| ≤ 1 / ℂᶜ, so that the buoyant growth rate of the TKE, ℂᶜ τ |N²|, is at most |N|
    for ratio in (0.5, 1, 1.5, 2.25, 3, 5), N² in (-1e-3, -1e-5, 1e-5, 1e-3), tke in (1e-4, 0.1, 10.0), λ in (0.001, 0.08)
        sfʳ = ColumnStabilityDamping(FT; ℂᴰ₂ = ratio * 0.1)
        mix = FT(500)
        τ = isotropy_ratio(sfʳ, FT(λ), mix, sqrt(FT(tke)), FT(N²)) * mix / sqrt(FT(tke))
        bound = N² > 0 ? 1 / (2 * sqrt(FT(λ))) : 1 / sfʳ.ℂᶜ
        @test τ * sqrt(abs(FT(N²))) ≤ (1 + rtol_round) * bound
        N² < 0 && @test sfʳ.ℂᶜ * τ * abs(FT(N²)) ≤ (1 + rtol_round) * sqrt(abs(FT(N²)))
    end
end
