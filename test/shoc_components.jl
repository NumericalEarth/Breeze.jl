include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Grids: znodes
using Breeze.TurbulenceClosures: convert_eltype, column_stability_field, diffusivity_stability_functionsᶜᶜᶠ,
                                 dissipation_stability_functionᶜᶜᶜ, shoc_distribution
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: trainable_parameter_names, is_calibratable, component_scalar_names,
                                      clipped_maximum, clip_between, bounded_condensate, skewness_share,
                                      third_moment_bound, limit_third_moment, third_moment,
                                      bound_stratification_number, assumed_pdf, component_saturation, liquid_water_temperature, component_variance,
                                      smooth_scalar_components, isotropy_ratio, shoc_dissipation_coefficient,
                                      column_stability_damping, lower_troposphere_weight
using Breeze.Thermodynamics: VaporEquivalentExponent, WarmPhaseEquilibrium
using Enzyme: Enzyme
using Test

# Bitwise equality, with NaN equal to NaN and the two zeros distinct
same_bits(a, b) = a === b

# The limited third moment as a function of one scalar, for derivatives: of the diagnosed third
# moment, of the vertical velocity variance, of `maximum_skewness`, and of the coefficient `c` of the diagnosis
limited_by_third_moment(w′³, limiter, w′²) = limit_third_moment(limiter, w′², w′³)
limited_by_variance(w′², limiter, w′³) = limit_third_moment(limiter, w′², w′³)
limited_by_maximum_skewness(maximum_skewness, w′², w′³) =
    limit_third_moment(PolynomialSaturatedSkewness(maximum_skewness), w′², w′³)
limited_by_coefficient(c, limiter, g, arguments) =
    limit_third_moment(limiter, arguments[3], third_moment(AlgebraicThirdMoment(Float64; ℂʷ₂ = c), UnboundedStratificationNumber(), g, arguments...))

# The derivative of `f(x, parameters...)` with respect to `x` by reverse-mode automatic differentiation
derivative(f, x, parameters...) =
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Active(x), map(Enzyme.Const, parameters)...)[1][1]

@testset "The smooth bounds of the assumed PDFs [$FT]" for FT in test_float_types()
    finite = FT[-1e6, -100, -1, -floatmin(FT), -0.0, 0.0, floatmin(FT), 4e-4, 0.5, 1, 100, 101, 1e6]
    scale = one(FT)

    @test_throws ArgumentError SmoothClipping(FT; width = 0)
    @test_throws ArgumentError SmoothClipping(-one(FT))
    @test SmoothClipping(FT(0.1)) === SmoothClipping(FT; width = 0.1)

    for width in FT[1e-1, 1e-2, 1e-3]
        clipping = SmoothClipping(width)
        δ = width * scale
        for a in finite, b in finite
            # One-sided, and within δ / 2 of the larger (to round-off)
            roundoff = 4eps(FT) * (abs(a) + abs(b))
            @test clipped_maximum(clipping, a, b, scale) ≥ max(a, b) - roundoff
            @test clipped_maximum(clipping, a, b, scale) - max(a, b) ≤ δ / 2 + roundoff
            @test clipped_maximum(clipping, a, b, scale) isa FT
        end
        for x in finite
            # Between the bounds by construction: a weighted mean of them with weights (1 ∓ t) / 2, |t| ≤ 1, for an
            # argument at any distance, with no difference of large numbers
            @test FT(0.01) ≤ clip_between(clipping, x, FT(0.01), FT(0.99), 1) ≤ FT(0.99)
            @test -1 ≤ clip_between(clipping, x, -1, 1, 1) ≤ 1
            @test 0 ≤ clip_between(clipping, x, 0, 100, 1) ≤ 100
            @test abs(clip_between(clipping, x, -1, 1, 1) - clamp(x, -1, 1)) ≤ δ
            @test clip_between(clipping, x, -1, 1, 1) isa FT
            # Odd about the middle of symmetric bounds
            @test clip_between(clipping, -x, -1, 1, 1) == -clip_between(clipping, x, -1, 1, 1)
        end
        # The bound is approached from inside, as (δ² / 4) (1 / (x - 1) - 1 / (x + 1)) beyond it
        for x in FT[3, 10, 100]
            @test isapprox(1 - clip_between(clipping, x, -1, 1, 1), δ^2 / (2 * (x^2 - 1)); rtol = 0.05, atol = 8eps(FT))
        end
        # Strictly inside the bounds where the argument is within a few bounds of them, and increasing
        @test FT(0.01) < clip_between(clipping, FT(-1), FT(0.01), FT(0.99), 1) < clip_between(clipping, FT(2), FT(0.01), FT(0.99), 1) < FT(0.99)
        samples = range(FT(-3), FT(3); length = 601)
        @test issorted([clip_between(clipping, x, -1, 1, 1) for x in samples])
        # Far inside the bounds the argument is unchanged, to δ² / (2 × the distance to the nearer bound)
        @test isapprox(clip_between(clipping, FT(0.3), -1, 1, 1), FT(0.3); atol = δ^2)
    end

    # The width → 0 limit is the larger of the two, and the bounded argument
    narrow = SmoothClipping(sqrt(eps(FT)))
    for a in finite, b in finite
        @test isapprox(clipped_maximum(narrow, a, b, scale), max(a, b); atol = sqrt(eps(FT)) * (1 + abs(a) + abs(b)))
    end
    for x in finite
        @test isapprox(clip_between(narrow, x, -1, 1, 1), clamp(x, -1, 1); atol = sqrt(eps(FT)))
    end

    # The smooth bounds have continuous slopes: the one-sided slopes of clipped_maximum(0, x) at x = 0 are both 1/2,
    # and the slope of the two-sided bound at a bound is 1/2 from either side
    if FT == Float64
        clipping = SmoothClipping(FT(0.1))
        h = FT(1e-6)
        below = (clipped_maximum(clipping, 0, zero(FT), scale) - clipped_maximum(clipping, 0, -h, scale)) / h
        above = (clipped_maximum(clipping, 0, h, scale) - clipped_maximum(clipping, 0, zero(FT), scale)) / h
        @test isapprox(below, 0.5; atol = 1e-4)
        @test isapprox(above, 0.5; atol = 1e-4)
        below = (clip_between(clipping, one(FT), -1, 1, 1) - clip_between(clipping, 1 - h, -1, 1, 1)) / h
        above = (clip_between(clipping, 1 + h, -1, 1, 1) - clip_between(clipping, one(FT), -1, 1, 1)) / h
        @test isapprox(below, 0.5; atol = 1e-3)
        @test isapprox(above, 0.5; atol = 1e-3)
        @test isapprox(derivative((x, clipping) -> clip_between(clipping, x, -1, 1, 1), 1.0, clipping), 0.5; atol = 1e-3)
    end

    @testset "The liquid of a component is no more than its total water" begin
        for qᶜ in FT[0, 1e-9, 1e-5, 1e-3, 1e-2, 1], qᵗ in FT[-1e-3, 0, 1e-9, 1e-5, 1e-3, 1e-2]
            qˡ = bounded_condensate(qᶜ, qᵗ)
            @test qˡ isa FT
            @test 0 ≤ qˡ ≤ qᶜ
            @test qˡ ≤ max(qᵗ, 0) * (1 + 2eps(FT))
        end
        # A condensate well below the total water is changed by the fraction (qᶜ / qᵗ)⁴ / 4
        @test isapprox(1 - bounded_condensate(FT(2e-3), FT(1e-2)) / FT(2e-3), FT(0.2)^4 / 4; rtol = 1e-2, atol = 4eps(FT))
        @test bounded_condensate(FT(1e-4), FT(1e-2)) ≈ FT(1e-4)
        # and one far above it is the total water
        @test bounded_condensate(FT(1), FT(1e-3)) ≈ FT(1e-3)
        @test bounded_condensate(zero(FT), zero(FT)) == 0
    end

    @testset "The share of the skewness that total water takes" begin
        pdf = DoubleGaussianPDF(FT)
        middle = (pdf.ℂᵍ₃ + pdf.ℂᵍ₄) / 2
        @test skewness_share(pdf, zero(FT)) == 0
        @test skewness_share(pdf, middle) ≈ FT(0.5)
        separations = range(zero(FT), FT(3); length = 301)
        shares = [skewness_share(pdf, d) for d in separations]
        @test all(share -> 0 ≤ share < 1, shares)
        @test issorted(shares)
        # The limits of the linear ramp of shoc.F90, zero and one, and its slope 1 / (ℂᵍ₄ - ℂᵍ₃) at the middle
        @test skewness_share(pdf, pdf.ℂᵍ₃ / 4) < 1e-3
        @test skewness_share(pdf, 4 * pdf.ℂᵍ₄) > 1 - 1e-3
        h = FT == Float32 ? FT(1e-2) : FT(1e-5)
        slope = (skewness_share(pdf, middle + h * middle) - skewness_share(pdf, middle - h * middle)) / (2 * h * middle)
        @test isapprox(slope, 1 / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃); rtol = 1e-2)
        # It vanishes faster than the separation, so that the skewness over the separation vanishes with it
        @test skewness_share(pdf, FT(1e-3)) / FT(1e-3) < 1e-6
    end
end

@testset "SHOC components of the subgrid distribution [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    @testset "Third moment and skewness limiter" begin
        g = FT(9.80616) # the gravitational acceleration of shoc.F90 (`ggr`)
        diagnosis = AlgebraicThirdMoment(FT)
        polynomial = PolynomialSaturatedSkewness(FT)
        @test summary(polynomial) == "PolynomialSaturatedSkewness{$FT}"
        @test closure_parameters(SimplifiedHigherOrderClosure(FT; subgrid_distribution = shoc_distribution(FT; skewness = TanhSaturatedSkewness(FT)))).ℂʷ₂ == FT(7)
        @test SimplifiedHigherOrderClosure(FT).subgrid_distribution.skewness === polynomial
        @test_throws ArgumentError PolynomialSaturatedSkewness(FT; clip = 1)

        # The diagnosis of shoc.F90 (`compute_diag_third_shoc_moment`) with its literals written out: the diagnosis with
        # its default fields returns the same bits
        function literal_third_moment(g, τ, N², w′², θˡⁱ, compact_inverse_spacing, wide_inverse_spacing,
                                      Δθˡⁱ′², Δw′θˡⁱ′, Δw′², Δe, w′θˡⁱ′)
            c = FT(7)
            τ² = τ * τ
            αᴺ = τ² * N²
            β = g / θˡⁱ
            f₀ = wide_inverse_spacing * (β * β * β) * ((τ * τ) * (τ * τ)) * w′θˡⁱ′ * Δθˡⁱ′²
            f₁ = wide_inverse_spacing * (β * β) * (τ * τ * τ) * (w′θˡⁱ′ * Δw′θˡⁱ′ + FT(0.5) * w′² * Δθˡⁱ′²)
            f₂ = compact_inverse_spacing * β * τ² * w′θˡⁱ′ * Δw′² + 2 * wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′
            f₃ = wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′ + compact_inverse_spacing * β * τ² * (w′θˡⁱ′ * Δe)
            f₄ = compact_inverse_spacing * τ * w′² * (Δw′² + Δe)
            f₅ = compact_inverse_spacing * τ * w′² * Δw′²
            a₄ = FT(2.4) / (3 * c + 5)
            a₅ = FT(0.6) / (c * (3 + 5 * c))
            ω₀ = a₄ / (1 - a₅ * αᴺ)
            ω₁ = ω₀ / (2 * c)
            ω₂ = ω₁ * f₃ + (FT(5) / 4) * ω₀ * f₄
            a₀ = (FT(0.52) * (1 / (c * c))) / (c - 2)
            a₁ = FT(0.87) / (c * c)
            a₂ = FT(0.5) / c
            a₃ = FT(0.6) / (c * (c - 2))
            X₀ = (a₂ * αᴺ * (1 - a₃ * αᴺ)) / (1 - (a₁ + a₃) * αᴺ)
            Y₀ = (2 * a₂ * αᴺ * X₀) / (1 - a₃ * αᴺ)
            X₁ = (a₀ * f₀ + a₁ * f₁ + a₂ * (1 - a₃ * αᴺ) * f₂) / (1 - (a₁ + a₃) * αᴺ)
            Y₁ = (2 * a₂ * (αᴺ * X₁ + (a₀ / a₁) * f₀ + f₁)) / (1 - a₃ * αᴺ)
            A₀ = ω₀ * X₀ + ω₁ * Y₀
            A₁ = ω₀ * X₁ + ω₁ * Y₁ + ω₂
            return (A₁ - FT(1.2) * X₁ - FT(1.5) * f₅) / (c - FT(1.2) * X₀ + A₀)
        end

        # Deterministic pseudo-random inputs over the ranges the column visits, across the poles
        # of the diagnosis (αᴺ = τ² N² up to several hundred) and across the bound of the limiter. The limiter writes
        # the bound as maximum_skewness (w′²)^{3/2} where shoc.F90 writes 1.2 √(2 (w′²)³): the two agree to round-off.
        # Where shoc.F90 resets a third moment beyond the bound to a constant, the limited third moment is within
        # the bound, of the sign of the diagnosed one, and no larger than it
        fraction(n, m) = FT(mod(n * 0.6180339887498949 + m * 0.7548776662466927, 1))
        limited_count = 0
        for n in 1:2000
            τ = FT(10)^(1 + FT(2.5) * fraction(n, 1))
            N² = (fraction(n, 2) - FT(0.4)) * FT(4e-4)
            w′² = FT(4e-4) + 2 * fraction(n, 3)^2
            θˡⁱ = 280 + 40 * fraction(n, 4)
            compact_inverse_spacing = 1 / (10 + 90 * fraction(n, 5))
            wide_inverse_spacing = compact_inverse_spacing / 2
            arguments = (τ, N², w′², θˡⁱ, compact_inverse_spacing, wide_inverse_spacing, (fraction(n, 6) - FT(0.5)) * FT(0.1),
                         (fraction(n, 7) - FT(0.5)) * FT(0.02), (fraction(n, 8) - FT(0.5)) * FT(0.5),
                         (fraction(n, 9) - FT(0.5)) * FT(0.5), (fraction(n, 10) - FT(0.5)) * FT(0.1))
            w′³ = third_moment(diagnosis, UnboundedStratificationNumber(), g, arguments...)
            @test same_bits(w′³, literal_third_moment(g, arguments...))
            literal_bound = FT(1.2) * sqrt(2 * (w′² * w′² * w′²))
            bound = third_moment_bound(polynomial, w′²)
            @test isapprox(bound, literal_bound; rtol = 8 * eps(FT))
            limited = limit_third_moment(polynomial, w′², w′³)
            @test abs(limited) ≤ bound
            @test abs(limited) ≤ abs(w′³) * (1 + 4eps(FT))
            @test sign(limited) == sign(w′³)
            limited_count += abs(w′³) > bound
        end
        @test 0 < limited_count < 2000 # the inputs are on both sides of the bound

        w′² = FT(0.5)
        bound = third_moment_bound(polynomial, w′²)
        @test polynomial.maximum_skewness ≈ FT(1.2) * sqrt(FT(2))
        @test bound ≈ polynomial.maximum_skewness * w′²^FT(1.5)
        @test bound ≈ FT(1.2) * sqrt(2 * w′²^3) # as shoc.F90 writes it
        @test third_moment_bound(TanhSaturatedSkewness(FT; maximum_skewness = polynomial.maximum_skewness), w′²) ≈ bound

        # Bounded by construction for an argument at any distance: the limited third moment is the bound times a
        # polynomial that is one minus a product of factors that are not negative
        for w′³ in FT[-1e30, -1e6, -10, -1.875, -1.8749, -1, -0.5, -1e-3, -1e-30, 0, 1e-30, 1e-3, 0.5, 1, 1.8749, 1.875, 10, 1e6, 1e30] .* bound
            @test abs(limit_third_moment(polynomial, w′², w′³)) ≤ bound
            @test sign(limit_third_moment(polynomial, w′², w′³)) == sign(w′³)
            @test limit_third_moment(polynomial, w′², -w′³) == -limit_third_moment(polynomial, w′², w′³)
        end
        # A third moment many orders of magnitude inside the bound keeps its relative precision
        @test isapprox(limit_third_moment(polynomial, w′², FT(1e-20) * bound), FT(1e-20) * bound; rtol = 8eps(FT))

        # Well inside the bound the polynomial changes little: less than a hyperbolic tangent would
        @test isapprox(limit_third_moment(polynomial, w′², bound / 100), bound / 100; rtol = 1e-4)
        @test isapprox(limit_third_moment(polynomial, w′², bound / 2), bound / 2; rtol = 0.05)
        @test limit_third_moment(polynomial, w′², bound) > bound * tanh(one(FT))
        @test limit_third_moment(polynomial, w′², zero(FT)) == 0

        # The polynomial is the bound from 15/8 of it on, and increasing before; its two forms meet at half of that
        @test limit_third_moment(polynomial, w′², FT(1.875) * bound) == bound
        @test limit_third_moment(polynomial, w′², 100 * bound) == bound
        samples = range(-2 * bound, 2 * bound, length = 401)
        limited = [limit_third_moment(polynomial, w′², FT(w′³)) for w′³ in samples]
        @test issorted(limited)
        meeting = FT(1.875) * bound / 2
        @test isapprox(limit_third_moment(polynomial, w′², meeting * (1 + 4eps(FT))), limit_third_moment(polynomial, w′², meeting); rtol = 16eps(FT))

        # Without vertical velocity variance the bound is zero and so is the third moment
        @test limit_third_moment(polynomial, zero(FT), FT(1)) == 0
        @test limit_third_moment(polynomial, -FT(1), FT(1)) == 0
        @test limit_third_moment(TanhSaturatedSkewness(FT), zero(FT), FT(1)) == 0
        @test limit_third_moment(TanhSaturatedSkewness(FT), -FT(1), FT(1)) == 0

        if FT == Float64
            # The slope of the polynomial limiter against central differences, on both sides of the
            # bound (ratio 1), of the point where the two forms of the polynomial meet (ratio 15/16) and of the point
            # where the polynomial reaches the bound (ratio 15/8): the exact slope is (1 - u²)², u = 8/15 ratio
            h = 1e-6
            for ratio in (-2.5, -1.2, -1, -0.3, 0.0, 0.3, 0.9, 15/16 - 1e-9, 15/16, 15/16 + 1e-9, 1 - 1e-9, 1.0, 1 + 1e-9, 1.1, 1.5, 1.8, 1.874, 1.876, 2.5)
                w′³ = ratio * bound
                difference = (limit_third_moment(polynomial, w′², w′³ + h * bound) -
                              limit_third_moment(polynomial, w′², w′³ - h * bound)) / (2 * h * bound)
                u = clamp(8 * ratio / 15, -1, 1)
                @test isapprox(difference, (1 - u^2)^2; atol = 1e-6)

                gradient = derivative(limited_by_third_moment, w′³, polynomial, w′²)
                @test isapprox(gradient, difference; atol = 1e-6)

                # No jump anywhere
                if abs(ratio) > 0.5
                    jump = limit_third_moment(polynomial, w′², w′³ * (1 + h)) - limit_third_moment(polynomial, w′², w′³ * (1 - h))
                    @test abs(jump) ≤ 2 * h * abs(w′³)
                end
            end

            # The slope with respect to the variance and to maximum_skewness, through the bound
            for ratio in (0.3, 1.1, 1.8, 2.5)
                w′³ = ratio * bound
                for (func, x, parameters) in ((limited_by_variance, w′², (polynomial, w′³)),
                                              (limited_by_maximum_skewness, polynomial.maximum_skewness, (w′², w′³)))
                    difference = (func(x * (1 + h), parameters...) - func(x * (1 - h), parameters...)) / (2 * h * x)
                    gradient = derivative(func, x, parameters...)
                    @test isapprox(gradient, difference; rtol = 1e-6, atol = 1e-9)
                end
            end

            # The diagnosis and the limiter together, with respect to the coefficient c, at a state
            # whose diagnosed third moment is beyond the bound
            arguments = (300.0, 1e-4, 0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            unlimited = third_moment(diagnosis, UnboundedStratificationNumber(), g, arguments...)
            for ratio in (0.9, 1.2, 1.6)
                limiter = PolynomialSaturatedSkewness(abs(unlimited) / (ratio * 0.4^1.5))
                @test abs(unlimited) / third_moment_bound(limiter, 0.4) ≈ ratio
                parameters = (limiter, g, arguments)
                difference = (limited_by_coefficient(7 + 1e-5, parameters...) - limited_by_coefficient(7 - 1e-5, parameters...)) / 2e-5
                gradient = derivative(limited_by_coefficient, 7.0, parameters...)
                @test gradient != 0
                @test isapprox(gradient, difference; rtol = 1e-6)
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

    @testset "DoubleGaussianPDF and SmoothDoubleGaussianPDF" begin
        # A narrow smoothing of the bounds, where the bounds are not what is tested
        narrow = SmoothClipping(FT(1e-4))
        shoc_pdf = DoubleGaussianPDF(FT; clipping = narrow)
        smooth_pdf = SmoothDoubleGaussianPDF(FT; clipping = narrow)
        @test SimplifiedHigherOrderClosure(FT; subgrid_distribution = shoc_distribution(FT; assumed_pdf = smooth_pdf)) isa
              SimplifiedHigherOrderClosure
        @test DoubleGaussianPDF(FT).clipping === SmoothClipping(FT)
        @test_throws ArgumentError DoubleGaussianPDF(FT; minimum_liquid_water_temperature = 100)

        # One level at 900 hPa and 285 K, close to saturation, between interfaces that carry the
        # same moments (w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
        pressure, θˡⁱ = FT(9e4), FT(285) * (FT(1e5) / FT(9e4))^(FT(287.042) / FT(1004.64))
        thermodynamics = (constants = ThermodynamicConstants(FT), standard_pressure = FT(1e5), exner_exponent = VaporEquivalentExponent(),
                          phase_equilibrium = WarmPhaseEquilibrium())
        # Saturation of the level's air with its total water as vapor: a fixed point in qᵗ
        qᵛ⁺ = zero(FT)
        for _ in 1:4
            Tˡ = liquid_water_temperature(thermodynamics, θˡⁱ, qᵛ⁺, pressure)
            qᵛ⁺ = first(component_saturation(thermodynamics, Tˡ, qᵛ⁺, pressure))
        end
        pdf_of(pdf, qᵗ, w′², moments) =
            assumed_pdf(pdf, thermodynamics, FT(1020), FT(980), FT(1000), pressure, θˡⁱ, zero(FT), qᵗ,
                        w′², moments, moments)

        # With resolved variances the smooth PDF is the PDF of shoc.F90 as its tolerances vanish, and within a few
        # percent of it with the tolerances of shoc.F90, which are a sixth of the standard deviation of total water here
        sharp_pdf = SmoothDoubleGaussianPDF(FT; clipping = narrow, potential_temperature_tolerance = 1e-5, total_water_tolerance = 1e-7, vertical_velocity_tolerance = 1e-4, minimum_saturation_spread = 1e-9)
        w′² = FT(0.6)
        for w′³ in FT[-0.2, 0.05, 0.4], qᵗ in (FT(0.97), FT(1.0), FT(1.03)) .* qᵛ⁺
            moments = (w′³, FT(0.25), FT(-0.25), FT(-1.3e-4), FT(2.9e-4), FT(4e-7))
            reference = pdf_of(shoc_pdf, qᵗ, w′², moments)
            smooth = pdf_of(smooth_pdf, qᵗ, w′², moments)
            sharp = pdf_of(sharp_pdf, qᵗ, w′², moments)
            @test isapprox(smooth[1], reference[1]; atol = 0.03)        # cloud fraction
            @test isapprox(smooth[2], reference[2]; rtol = 0.05)        # liquid water
            @test isapprox(smooth[4], reference[4]; rtol = 0.05)        # liquid water flux
            if FT == Float64
                @test isapprox(sharp[1], reference[1]; atol = 1e-4)
                @test isapprox(sharp[2], reference[2]; rtol = 1e-3)
                @test isapprox(sharp[4], reference[4]; rtol = 1e-3)
                @test isapprox(sharp[5], reference[5]; atol = 1e-4)
            end
        end

        # Calm and uniform air: the components collapse onto the mean, saturated air is cloud and
        # unsaturated air is clear
        still = ntuple(_ -> zero(FT), 6)
        calm = FT(2e-4)
        for pdf in (shoc_pdf, smooth_pdf)
            cloud_fraction, liquid, liquid_variance, liquid_flux, buoyancy_flux = pdf_of(pdf, FT(1.02) * qᵛ⁺, calm, still)
            @test isapprox(cloud_fraction, 1; atol = 1e-3)
            @test liquid > 0
            @test abs(liquid_flux) ≤ 4eps(FT) * liquid
            @test first(pdf_of(pdf, FT(0.98) * qᵛ⁺, calm, still)) < 1e-3
            # and without vertical velocity variance at all
            @test all(isfinite, pdf_of(pdf, FT(1.02) * qᵛ⁺, zero(FT), still))
        end
        @test isapprox(pdf_of(smooth_pdf, FT(1.02) * qᵛ⁺, calm, still)[2], pdf_of(shoc_pdf, FT(1.02) * qᵛ⁺, calm, still)[2]; rtol = 1e-2)

        # No jump where shoc.F90 switches: the outputs are continuous in the vertical velocity variance across
        # vertical_velocity_tolerance² and in the scalar variance across its tolerance², for both PDFs
        variable = (FT(1e-4), FT(0.02), FT(-2e-3), FT(-2e-6), FT(3e-6), FT(4e-8))
        step(pdf, f) = maximum(abs, f(pdf, 1 + FT(1e-3)) .- f(pdf, 1 - FT(1e-3)))
        across_calm(pdf, x) = pdf_of(pdf, qᵛ⁺, shoc_pdf.vertical_velocity_tolerance^2 * x, variable)[1:2]
        across_uniform(pdf, x) = pdf_of(pdf, qᵛ⁺, FT(0.3), (variable[1], variable[2], variable[3], variable[4], variable[5],
                                                         shoc_pdf.total_water_tolerance^2 * x))[1:2]
        for pdf in (smooth_pdf, DoubleGaussianPDF(FT), EqualWidthDoubleGaussianPDF(FT))
            @test step(pdf, across_calm) < 1e-3
            @test step(pdf, across_uniform) < 1e-3
        end
        # and the components collapse onto the mean as the variances vanish: the cloud of the mean state
        vanishing = (zero(FT), FT(1e-12), zero(FT), zero(FT), zero(FT), FT(1e-16))
        for pdf in (DoubleGaussianPDF(FT), EqualWidthDoubleGaussianPDF(FT))
            @test first(pdf_of(pdf, FT(1.02) * qᵛ⁺, FT(0.3), vanishing)) ≈ 1
            @test first(pdf_of(pdf, FT(0.98) * qᵛ⁺, FT(0.3), vanishing)) < 1e-6
        end

        # The within-component variance ratio tends to one as the correlation with w vanishes, in both PDFs
        ŵ₁, ŵ₂ = sqrt(FT(0.6)), -sqrt(FT(0.6))
        _, _, ratio₁, ratio₂ = smooth_scalar_components(smooth_pdf, narrow, zero(FT), ŵ₁, ŵ₂, zero(FT), FT(0.5), true)
        @test ratio₁ ≈ 1 && ratio₂ ≈ 1
        @test component_variance(narrow, true, zero(FT), zero(FT), FT(0.5), zero(FT), one(FT), FT(100)) ≈ 1
        @test component_variance(narrow, true, FT(1e-6) / ŵ₁, FT(1e-6) / ŵ₂, FT(0.5), zero(FT), one(FT), FT(100)) ≈ 1
        # and the two forms of the ratio agree
        r = FT(0.3)
        x̂₁, x̂₂, ratio₁, ratio₂ = smooth_scalar_components(smooth_pdf, narrow, r, ŵ₁, ŵ₂, zero(FT), FT(0.5), false)
        @test ratio₁ ≈ component_variance(narrow, true, x̂₁, x̂₂, FT(0.5), zero(FT), one(FT), FT(100))
        @test ratio₂ ≈ component_variance(narrow, false, x̂₁, x̂₂, FT(0.5), zero(FT), one(FT), FT(100))

        # A spread of the saturation deficit that does not vanish: the cloud fraction passes
        # through one half at saturation instead of jumping
        @test isapprox(first(pdf_of(smooth_pdf, qᵛ⁺, calm, still)), 0.5; atol = 0.05)
    end
end

#####
##### SHOC's return-to-isotropy time scale and diffusivities as stability functions
#####

# `isotropic_ts` and `eddy_diffusivities` of shoc.F90 (without the cold-surface fallback), with the dissipation
# 𝕊ᴰ e^{3/2} / ℓ of `adv_sgs_tke`, for turbulent kinetic energy `e`, static stability `N²`, the integral `I` of the
# static stability over the lower troposphere divided by g, and mixing length `ℓ`: the time scale, the two
# diffusivities, the dissipation coefficient and the stability damping
function shoc_time_scale_and_diffusivities(e, N², I, ℓ)
    𝕊ᴰ = 0.1^3 / 0.15^4 / 0.7 * 0.19 + 0.1^3 / 0.15^4 / 0.7 * 0.51
    ε = 𝕊ᴰ / ℓ * e^1.5
    λ = max(0.001, min(0.08, 0.001 + (I - 0.02) * 2.65))
    λ = N² <= 0 ? 0.0 : λ
    τ₀ = (2 * e) / ε
    τ = min(20000.0, τ₀ / (1 + λ * N² * τ₀^2))
    return τ, 0.1 * τ * e, 0.1 * τ * e, 𝕊ᴰ, λ
end

@testset "ColumnStabilityDamping: the time scale and diffusivities of shoc.F90, regular through N² = 0 [$FT]" for FT in test_float_types()
    damping = ColumnStabilityDamping(FT)
    @test summary(damping) == "ColumnStabilityDamping{$FT}"
    @test convert_eltype(Float32, damping).ℂᶜ === 0.1f0
    @test shoc_dissipation_coefficient(damping) ≈ shoc_time_scale_and_diffusivities(0.1, 0.0, 0.0, 100.0)[4] rtol = 1e-6
    @test column_stability_field(RectilinearGrid(size = 4, z = (0, 1), topology = (Flat, Flat, Bounded)), damping) isa Field

    roundoff = FT == Float32 ? 1e-5 : 1e-12
    λ⁻ = 0.1^2 / 4 # the damping of unstable air, ℂᶜ² / 4
    for e in (0.01, 0.1, 1.0), N² in (-1e-4, 0.0, 1e-5, 3e-4), I in (0.0, 0.25, 0.5, 2.0), ℓ in (30.0, 300.0)
        τˢ, Kᶜˢ, Kᵘˢ, _, λ = shoc_time_scale_and_diffusivities(e, N², I, ℓ)
        # shoc.F90 does not damp where N² ≤ 0; the closure damps there with ℂᶜ² / 4
        λ = N² <= 0 ? 0.001 : λ
        τ₀ = shoc_time_scale_and_diffusivities(e, 0.0, I, ℓ)[1]
        τˢ = N² < 0 ? τ₀ / (1 + λ⁻ * abs(N²) * τ₀^2) : τˢ
        Kˢ = 0.1 * τˢ * e
        s = isotropy_ratio(damping, FT(λ), FT(ℓ), sqrt(FT(e)), FT(N²))
        τ = s * ℓ / sqrt(e)
        # The time scale is no longer than with the switch at N² = 0, and shorter by at most the damping the regular
        # stable and unstable parts add to it, (λ + ℂᶜ² / 4) (√(αᴺ² + 1) - |αᴺ|) / 2 with αᴺ = N² τ₀²: half the sum of
        # the two dampings in neutral air, and (λ + ℂᶜ² / 4) / (4 |αᴺ|) far from it. The sum of inverse squares in
        # place of min(maxiso, ⋅) shortens it by (τ / maxiso)² / 2 more
        αᴺ = N² * τ₀^2
        regular = (λ + λ⁻) * (sqrt(αᴺ^2 + 1) - abs(αᴺ)) / 2
        bound = (τˢ / 20000)^2
        @test τ ≤ τˢ * (1 + roundoff)
        @test (τˢ - τ) / τˢ ≤ regular + bound + roundoff
        @test abs(damping.ℂᶜ * s * ℓ * sqrt(e) - Kˢ) / Kˢ ≤ regular + bound + roundoff
        @test abs(damping.ℂᵘ * s * ℓ * sqrt(e) - Kˢ) / Kˢ ≤ regular + bound + roundoff
    end

    # Far from N² = 0 on the scale τ₀⁻² the time scale is that of the switch
    let e = 0.5, ℓ = 400.0, λ = 0.04
        τ₀ = shoc_time_scale_and_diffusivities(e, 0.0, 0.0, ℓ)[1]
        for αᴺ in (-1e4, -1e2, 1e2, 1e4)
            N² = αᴺ / τ₀^2
            τˢ = N² < 0 ? τ₀ / (1 + λ⁻ * abs(N²) * τ₀^2) : τ₀ / (1 + λ * N² * τ₀^2)
            τ = isotropy_ratio(damping, FT(λ), FT(ℓ), sqrt(FT(e)), FT(N²)) * ℓ / sqrt(e)
            @test isapprox(τ, τˢ; rtol = (λ + λ⁻) / (4 * abs(αᴺ)) + (τˢ / 20000)^2 + roundoff)
        end
    end

    # Regular through N² = 0: the one-sided slopes of the time scale in N² agree there, where those of the switch
    # differ by (λ + ℂᶜ² / 4) τ₀³
    if FT == Float64
        e, ℓ, λ = 0.5, 400.0, 0.04
        τ₀ = shoc_time_scale_and_diffusivities(e, 0.0, 0.0, ℓ)[1]
        time_scale(N²) = isotropy_ratio(damping, λ, ℓ, sqrt(e), N²) * ℓ / sqrt(e)
        h = 1e-3 / τ₀^2
        below = (time_scale(0.0) - time_scale(-h)) / h
        above = (time_scale(h) - time_scale(0.0)) / h
        kink = (λ + λ⁻) * τ₀^3
        @test abs(above - below) < 1e-2 * kink
        @test isapprox(derivative((N², damping, λ, ℓ, u) -> isotropy_ratio(damping, λ, ℓ, u, N²), 0.0, damping, λ, ℓ, sqrt(e)) * ℓ / sqrt(e),
                       (above + below) / 2; rtol = 1e-4)
        # In neutral air the damping is half the sum of the stable and the unstable ones
        @test time_scale(0.0) ≈ τ₀ / (1 + (λ + λ⁻) / 2) rtol = (τ₀ / 20000)^2
    end

    # The stratification bounds the time scale whatever the dissipation ratio ℂᴰ₂ / ℂᴰ₁: τ |N| ≤ 1 / (2 √λ) in stable
    # air, and in unstable air τ |N| ≤ 1 / ℂᶜ, so that the buoyant growth rate of the TKE, ℂᶜ τ |N²|, is at most |N|
    for ratio in (0.5, 1, 1.5, 2.25, 3, 5), N² in (-1e-3, -1e-5, 1e-5, 1e-3), e in (1e-4, 0.1, 10.0), λ in (0.001, 0.08)
        scaled = ColumnStabilityDamping(FT; ℂᴰ₂ = ratio * 0.1)
        ℓ = FT(500)
        τ = isotropy_ratio(scaled, FT(λ), ℓ, sqrt(FT(e)), FT(N²)) * ℓ / sqrt(FT(e))
        bound = N² > 0 ? 1 / (2 * sqrt(FT(λ))) : 1 / scaled.ℂᶜ
        @test τ * sqrt(abs(FT(N²))) ≤ (1 + roundoff) * bound
        N² < 0 && @test scaled.ℂᶜ * τ * abs(FT(N²)) ≤ (1 + roundoff) * sqrt(abs(FT(N²)))
    end
end

@testset "The stability damping of the column [$FT]" for FT in test_float_types()
    damping = ColumnStabilityDamping(FT)
    ℂᵝ₁, ℂᵝ₂, ℂᵝ₃, ℂᵝ₄ = damping.ℂᵝ₁, damping.ℂᵝ₂, damping.ℂᵝ₃, damping.ℂᵝ₄
    clamped(I) = clamp(ℂᵝ₁ + (I - ℂᵝ₄) * ℂᵝ₃, ℂᵝ₁, ℂᵝ₂) # shoc.F90
    middle = ℂᵝ₄ + (ℂᵝ₂ - ℂᵝ₁) / (2 * ℂᵝ₃)             # where the line is midway between its bounds

    # Between its bounds by construction, for an integrated stability of any size and sign, and increasing
    integrals = vcat(-FT(10) .^ range(FT(6), FT(-6); length = 49), zero(FT), FT(10) .^ range(FT(-6), FT(6); length = 49), middle)
    sort!(integrals)
    dampings = [column_stability_damping(damping, I) for I in integrals]
    @test all(λ -> λ isa FT, dampings)
    @test all(λ -> ℂᵝ₁ * (1 - 4eps(FT)) ≤ λ ≤ ℂᵝ₂ * (1 + 4eps(FT)), dampings)
    @test all(k -> dampings[k+1] ≥ dampings[k] * (1 - 4eps(FT)), 1:length(dampings)-1)
    @test issorted([column_stability_damping(damping, I) for I in range(FT(-0.05), FT(0.15); length = 201)])
    @test ℂᵝ₁ < column_stability_damping(damping, FT(-0.05)) < column_stability_damping(damping, FT(0.15)) < ℂᵝ₂

    # The limits are the bounds, and the line of shoc.F90 where it is midway between them
    @test column_stability_damping(damping, FT(-1e6)) ≈ ℂᵝ₁
    @test column_stability_damping(damping, FT(1e6)) ≈ ℂᵝ₂
    @test column_stability_damping(damping, middle) ≈ (ℂᵝ₁ + ℂᵝ₂) / 2
    @test column_stability_damping(damping, middle) ≈ clamped(middle)
    # The line meets ℂᵝ₁ at I = ℂᵝ₄; there λ - ℂᵝ₁ = (1 - 2^(-1/4)) / 2 of ℂᵝ₂ - ℂᵝ₁
    @test (column_stability_damping(damping, ℂᵝ₄) - ℂᵝ₁) / (ℂᵝ₂ - ℂᵝ₁) ≈ (1 - FT(2)^(-FT(1) / 4)) / 2 rtol = FT == Float32 ? 1e-4 : 1e-10
    # Never farther from the clamped line than that, and close to it far from its two kinks
    @test all(I -> abs(column_stability_damping(damping, I) - clamped(I)) ≤ (1 - FT(2)^(-FT(1) / 4)) / 2 * (ℂᵝ₂ - ℂᵝ₁) * (1 + 16eps(FT)), integrals)
    @test isapprox(column_stability_damping(damping, FT(-0.2)), clamped(FT(-0.2)); atol = 1e-4 * (ℂᵝ₂ - ℂᵝ₁))
    @test isapprox(column_stability_damping(damping, FT(0.3)), clamped(FT(0.3)); atol = 1e-4 * (ℂᵝ₂ - ℂᵝ₁))

    if FT == Float64
        # Its slope at the middle is ℂᵝ₃, and continuous everywhere: at the kinks of the clamped line too
        h = 1e-7
        slope(I) = (column_stability_damping(damping, I + h) - column_stability_damping(damping, I - h)) / 2h
        @test slope(middle) ≈ ℂᵝ₃ rtol = 1e-6
        for I in (ℂᵝ₄, 2 * middle - ℂᵝ₄)
            @test isapprox(slope(I - h), slope(I + h); rtol = 1e-4)
            @test 0 < slope(I) < ℂᵝ₃
        end
        @test derivative((I, damping) -> column_stability_damping(damping, I), middle, damping) ≈ ℂᵝ₃
    end

    # The weight of a level in the integral over the lower troposphere: in (0, 1), increasing with the pressure,
    # one half at the pressure of the coefficient, with the limits of the switch of shoc.F90
    pᴵ = damping.stability_integration_pressure
    pressures = FT(10) .^ range(FT(0), FT(7); length = 141)
    weights = [lower_troposphere_weight(p, pᴵ) for p in pressures]
    @test all(weight -> weight isa FT && 0 ≤ weight ≤ 1, weights)
    @test issorted(weights)
    @test lower_troposphere_weight(pᴵ, pᴵ) == FT(0.5)
    @test lower_troposphere_weight(FT(1e5), pᴵ) > 0.97
    @test lower_troposphere_weight(FT(6e4), pᴵ) < 0.01
    @test lower_troposphere_weight(FT(1), pᴵ) < 1e-30 # the top of a deep model: no overflow to a NaN
    @test lower_troposphere_weight(FT(1e7), pᴵ) == 1
end
