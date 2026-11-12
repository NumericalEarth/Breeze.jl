include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Breeze.AtmosphereModels: materialize_closure, closure_scalar_names
using Breeze.TurbulenceClosures: convert_eltype
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: trainable_parameter_names, is_calibratable, component_scalar_names
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: clip_max, clip_min, clip_below, clip_between, clip_negative,
                                      stable_damping, hard_floor, third_moment_bound, limit_third_moment, third_moment,
                                      bound_stratification_number,
                                      vd_shoc_decomp!, vd_shoc_solve!, tke_step, buoyancy_production,
                                      assumed_pdf, plume_saturation, plume_variance, smooth_scalar_plumes,
                                      time_scale_column, dissipation_coefficient, time_scale_and_diffusivities, limit_time_scale, cold_surface_fallback,
                                      mixing_length_column!, mixing_length
using Enzyme: Enzyme
using Test

# Bitwise equality, with NaN equal to NaN and the two zeros distinct
same_bits(a, b) = a === b

# The limited third moment as a function of one scalar, for derivatives: of the diagnosed third
# moment, of the vertical velocity variance, of `maximum_skewness`, and of the coefficient `c` of the diagnosis
limited_by_third_moment(w3, limiter, policy, w_sec) = limit_third_moment(limiter, policy, w_sec, w3)
limited_by_variance(w_sec, limiter, policy, w3) = limit_third_moment(limiter, policy, w_sec, w3)
limited_by_maximum_skewness(maximum_skewness, policy, w_sec, w3) =
    limit_third_moment(PolynomialSaturatedSkewness(maximum_skewness), policy, w_sec, w3)
limited_by_coefficient(c, limiter, policy, constants, arguments) =
    limit_third_moment(limiter, policy, arguments[3], third_moment(AlgebraicThirdMoment(Float64; ℂʷ₂ = c), constants, arguments...))

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
    original = run_shoc_column(SimplifiedHigherOrderClosure(FT; surface_forcing...))
    @test all_finite(original)

    @testset "SmoothClipping" begin
        closure = SimplifiedHigherOrderClosure(FT; clipping = SmoothClipping(FT; width = 1e-2), surface_forcing...)
        @test isbits(closure)
        @test closure_parameters(closure) == closure_parameters(SimplifiedHigherOrderClosure(FT))
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
            narrower = run_shoc_column(SimplifiedHigherOrderClosure(FT; clipping = SmoothClipping(FT; width = 1e-4), surface_forcing...))
            @test maximum(abs, narrower.θ .- original.θ) < maximum(abs, smooth.θ .- original.θ)
        end
    end

    @testset "Third moment and skewness limiters" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        diagnosis = AlgebraicThirdMoment(FT)
        reset = ResetAboveThresholdSkewness(FT)
        clamped = ClampedSkewness(FT)
        polynomial = PolynomialSaturatedSkewness(FT)
        @test summary(polynomial) == "PolynomialSaturatedSkewness{$FT}"
        @test closure_parameters(SimplifiedHigherOrderClosure(FT; skewness = polynomial)).ℂʷ₂ == FT(7)
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
            w3 = third_moment(diagnosis, constants, arguments...)
            @test same_bits(w3, literal_third_moment(constants.ggr, arguments...))
            literal_bound = FT(1.2) * sqrt(2 * (w_sec * w_sec * w_sec))
            @test isapprox(third_moment_bound(reset, w_sec), literal_bound; rtol = 8 * eps(FT))
            if abs(abs(w3) - literal_bound) > 8 * eps(FT) * literal_bound
                @test same_bits(limit_third_moment(reset, hard, w_sec, w3), literal_limiter(w_sec, w3))
            end
            nreset += limit_third_moment(reset, hard, w_sec, w3) != w3
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
            @test limit_third_moment(reset, hard, w_sec, w3) == (inside ? w3 : FT(0.02))
            @test limit_third_moment(clamped, hard, w_sec, w3) == clamp(w3, -bound, bound)
            @test abs(limit_third_moment(clamped, SmoothClipping(FT(0.1)), w_sec, w3)) ≤ bound
            @test abs(limit_third_moment(polynomial, hard, w_sec, w3)) ≤ bound
            @test sign(limit_third_moment(polynomial, hard, w_sec, w3)) == sign(w3)
            @test limit_third_moment(polynomial, hard, w_sec, -w3) == -limit_third_moment(polynomial, hard, w_sec, w3)
        end

        # Well inside the bound the polynomial changes little: less than a hyperbolic tangent would
        @test isapprox(limit_third_moment(polynomial, hard, w_sec, bound / 100), bound / 100; rtol = 1e-4)
        @test isapprox(limit_third_moment(polynomial, hard, w_sec, bound / 2), bound / 2; rtol = 0.05)
        @test limit_third_moment(polynomial, hard, w_sec, bound) > bound * tanh(one(FT))
        @test limit_third_moment(polynomial, hard, w_sec, zero(FT)) == 0

        # The polynomial is the bound from 15/8 of it on, and increasing before
        @test limit_third_moment(polynomial, hard, w_sec, FT(1.875) * bound) ≈ bound
        @test limit_third_moment(polynomial, hard, w_sec, 100 * bound) ≈ bound
        samples = range(-2 * bound, 2 * bound, length = 401)
        limited = [limit_third_moment(polynomial, hard, w_sec, FT(w3)) for w3 in samples]
        @test issorted(limited)

        # Without vertical velocity variance the bound is zero and so is the third moment
        @test limit_third_moment(polynomial, hard, zero(FT), FT(1)) == 0
        @test limit_third_moment(polynomial, hard, -FT(1), FT(1)) == 0

        if FT == Float64
            # The slope of the polynomial limiter against central differences, on both sides of the
            # bound at which shoc.F90 resets the third moment (ratio 1) and of the point where the
            # polynomial reaches the bound (ratio 15/8): the exact slope is (1 - u²)², u = 8/15 ratio
            h = 1e-6
            for ratio in (-2.5, -1.2, -1, -0.3, 0.0, 0.3, 0.9, 1 - 1e-9, 1.0, 1 + 1e-9, 1.1, 1.5, 1.8, 1.874, 1.876, 2.5)
                w3 = ratio * bound
                difference = (limit_third_moment(polynomial, hard, w_sec, w3 + h * bound) -
                              limit_third_moment(polynomial, hard, w_sec, w3 - h * bound)) / (2 * h * bound)
                u = clamp(8 * ratio / 15, -1, 1)
                @test isapprox(difference, (1 - u^2)^2; atol = 1e-6)

                gradient = derivative(limited_by_third_moment, w3, polynomial, hard, w_sec)
                @test isapprox(gradient, difference; atol = 1e-6)

                # The limiter of shoc.F90 jumps at ratio 1; this one does not
                if abs(ratio) > 0.5
                    jump = limit_third_moment(polynomial, hard, w_sec, w3 * (1 + h)) - limit_third_moment(polynomial, hard, w_sec, w3 * (1 - h))
                    @test abs(jump) ≤ 2 * h * abs(w3)
                end
            end
            @test abs(limit_third_moment(reset, hard, w_sec, bound * (1 + h)) - limit_third_moment(reset, hard, w_sec, bound * (1 - h))) > bound / 2

            # The slope with respect to the variance and to maximum_skewness, through the bound
            for ratio in (0.3, 1.1, 1.8, 2.5)
                w3 = ratio * bound
                for (func, x, parameters) in ((limited_by_variance, w_sec, (polynomial, hard, w3)),
                                              (limited_by_maximum_skewness, polynomial.maximum_skewness, (hard, w_sec, w3)))
                    difference = (func(x * (1 + h), parameters...) - func(x * (1 - h), parameters...)) / (2 * h * x)
                    gradient = derivative(func, x, parameters...)
                    @test isapprox(gradient, difference; rtol = 1e-6, atol = 1e-9)
                end
            end

            # The diagnosis and the limiter together, with respect to the coefficient c, at a state
            # whose diagnosed third moment is beyond the bound of shoc.F90
            arguments = (300.0, 1e-4, 0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            raw = third_moment(diagnosis, constants, arguments...)
            for ratio in (0.9, 1.2, 1.6)
                limiter = PolynomialSaturatedSkewness(abs(raw) / (ratio * 0.4^1.5))
                @test abs(raw) / third_moment_bound(limiter, 0.4) ≈ ratio
                parameters = (limiter, hard, constants, arguments)
                difference = (limited_by_coefficient(7 + 1e-5, parameters...) - limited_by_coefficient(7 - 1e-5, parameters...)) / 2e-5
                gradient = derivative(limited_by_coefficient, 7.0, parameters...)
                @test gradient != 0
                @test isapprox(gradient, difference; rtol = 1e-6)
            end
        end

        for skewness in (clamped, polynomial)
            run = run_shoc_column(SimplifiedHigherOrderClosure(FT; skewness, surface_forcing...))
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test isapprox(run.θ, original.θ; rtol = 1e-4)
        end
    end

    @testset "Stratification number bound" begin
        constants = SHOCConstants(FT)
        diagnosis = AlgebraicThirdMoment(FT)
        unbounded = UnboundedStratificationNumber()
        rational = RationalSaturatedStratificationNumber(FT)
        @test summary(rational) == "RationalSaturatedStratificationNumber{$FT}"
        @test rational.pole_fraction == FT(1 / 2)
        @test SimplifiedHigherOrderClosure(FT).stratification_number_bound === unbounded
        @test smooth_shoc_components(FT).stratification_number_bound === rational
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; stratification_number_bound = ClampedSkewness())
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; skewness = rational)

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
            @test same_bits(third_moment(diagnosis, unbounded, constants, arguments...),
                            third_moment(diagnosis, constants, arguments...))
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
            @test isfinite(third_moment(diagnosis, rational, constants, arguments...))
        end
        # The first singularity of the unbounded diagnosis is where its denominator vanishes, at
        # αᴺ ≈ 24.26 (0.85 of the first pole, where the poles of X₀ and X₁ cancel in the ratio):
        # there the unbounded third moment is orders of magnitude larger than the bounded one
        near_zero = (τ, FT(24.2589579) / τ^2, FT(0.4), FT(300), FT(1 / 40), FT(1 / 80), FT(0.02), FT(-0.004), FT(0.1), FT(0.15), FT(0.03))
        @test abs(third_moment(diagnosis, unbounded, constants, near_zero...)) > 100 * abs(third_moment(diagnosis, rational, constants, near_zero...))

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
            moment_by_N²(N², bound, constants, arguments) = third_moment(diagnosis, bound, constants, arguments[1], N², arguments[2:end]...)
            moment_by_fraction(γ, constants, arguments) = third_moment(diagnosis, RationalSaturatedStratificationNumber(Float64; pole_fraction = γ), constants, arguments...)
            moment_by_c(c, constants, arguments) = third_moment(AlgebraicThirdMoment(Float64; ℂʷ₂ = c), rational, constants, arguments...)
            rest = (0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            for αᴺ in (-5.0, 0.5, 5.0, 20.0, 60.0)
                N² = αᴺ / τ^2
                arguments = (τ, N², rest...)
                difference = (moment_by_N²(N² * (1 + h), rational, constants, (τ, rest...)) -
                              moment_by_N²(N² * (1 - h), rational, constants, (τ, rest...))) / (2 * h * N²)
                gradient = derivative(moment_by_N², N², rational, constants, (τ, rest...))
                @test isapprox(gradient, difference; rtol = 1e-5)
                for (func, x) in ((moment_by_fraction, 0.5), (moment_by_c, 7.0))
                    difference = (func(x * (1 + h), constants, arguments) - func(x * (1 - h), constants, arguments)) / (2 * h * x)
                    gradient = derivative(func, x, constants, arguments)
                    @test isapprox(gradient, difference; rtol = 1e-5, atol = 1e-12)
                end
            end
        end

        run = run_shoc_column(SimplifiedHigherOrderClosure(FT; stratification_number_bound = rational, surface_forcing...))
        @test all_finite(run)
        @test isapprox(run.θ, original.θ; rtol = 1e-4)
    end

    @testset "ImplicitSinkTKE" begin
        # Without a sink the factorization is that of the diffusion alone; with one, the solve
        # damps a uniform profile by 1 / (1 + Δt ω) where the diffusivity vanishes
        Nz = 8
        Δt = FT(100)
        constants = SHOCConstants(FT)
        level() = zeros(FT, 1, 1, Nz)
        zi = reshape(FT.(0:100:100Nz), 1, 1, Nz + 1)
        zt = (zi[:, :, 1:Nz] .+ zi[:, :, 2:Nz+1]) ./ 2
        pdel = fill(FT(1000), 1, 1, Nz)
        kv = reshape(FT.(range(5, 50, length = Nz)), 1, 1, Nz)
        factorize(sink...) = (f = (level(), level(), level());
                              vd_shoc_decomp!(1, 1, Nz, Δt, constants, zt, zi, pdel, kv, zero(FT), f..., sink...); f)
        @test factorize() == factorize(nothing) == factorize(level())

        ω = FT(1e-3)
        f = (level(), level(), level())
        vd_shoc_decomp!(1, 1, Nz, Δt, constants, zt, zi, pdel, level(), zero(FT), f..., fill(ω, 1, 1, Nz))
        e = ones(FT, 1, 1, Nz)
        vd_shoc_solve!(1, 1, Nz, f..., e)
        @test all(e .≈ 1 / (1 + Δt * ω))

        closure = SimplifiedHigherOrderClosure(FT; tke_update = ImplicitSinkTKE(), surface_forcing...)
        @test isbits(closure)
        @test closure_parameters(closure).ℂᴾ == FT(0.1)
        @test haskey(shoc_column_model(closure).closure_fields.scratch, :tke_sink_rate)
        @test !haskey(shoc_column_model(SimplifiedHigherOrderClosure(FT)).closure_fields.scratch, :tke_sink_rate)

        # The update is positive without a floor for any time step, production and stratification,
        # where forward Euler leaves a negative value for the floor to catch
        hard = HardClipping()
        implicit_update = ImplicitSinkTKE(FT)
        euler = ForwardEulerTKE(FT)
        unfloored = SHOCConstants(FT; mintke = 0, maxtke = floatmax(FT))
        ℓ, Sᴰ = FT(50), FT(2)
        for Δt in FT[60, 300, 900, 3600], e in FT[1e-6, 4e-4, 0.1, 5], P in FT[0, 1e-4, 1e-2], B in FT[-1e-2, -1e-4, 0, 1e-3]
            estimate, ε, ω, stored = tke_step(implicit_update, hard, constants, Δt, e, P, B, ℓ, Sᴰ)
            @test estimate > 0 && stored > 0 && ω ≥ 0 && ε ≥ 0
            @test estimate ≤ stored
            # The same sources and sinks as forward Euler, to first order in the time step
            small = FT(1e-5)
            estimate, _, _, _ = tke_step(implicit_update, hard, unfloored, small, e, P, B, ℓ, Sᴰ)
            explicit, _, _, _ = tke_step(euler, hard, unfloored, small, e, P, B, ℓ, Sᴰ)
            P + B ≥ 0 && e ≥ constants.mintke && @test isapprox(estimate, explicit; rtol = FT == Float64 ? 1e-4 : 1e-3)
        end
        @test first(tke_step(euler, hard, unfloored, FT(900), FT(0.1), zero(FT), zero(FT), ℓ, Sᴰ)) == 0

        implicit = run_shoc_column(closure)
        @test all_finite(implicit)
        @test maximum(implicit.e) > FT(0.01)
        @test all(implicit.e .> 0)
        # The turbulence the column starts with decays over several steps in the stable air aloft
        # rather than in one, which mixes it a little more
        @test isapprox(implicit.θ, original.θ; rtol = 2e-3)

        for Δt in (300, 900)
            run = run_shoc_column(closure; steps = 10, Δt)
            @test all_finite(run)
            @test all(run.e .> 0)
        end
    end

    @testset "DowngradientBuoyancyFlux" begin
        constants = SHOCConstants(FT)
        wthv, tkh, N² = FT(0.02), FT(30), FT(1e-4)
        @test buoyancy_production(PDFBuoyancyFlux(), constants, wthv, tkh, N²) == (constants.ggr / constants.basetemp) * wthv
        @test buoyancy_production(DowngradientBuoyancyFlux(), constants, wthv, tkh, N²) == -tkh * N²
        @test buoyancy_production(DowngradientBuoyancyFlux(), constants, wthv, tkh, -N²) > 0

        downgradient = run_shoc_column(SimplifiedHigherOrderClosure(FT; buoyancy_source = DowngradientBuoyancyFlux(), surface_forcing...))
        @test all_finite(downgradient)
        @test maximum(downgradient.e) > FT(0.01)
        @test isapprox(downgradient.θ, original.θ; rtol = 2e-3)
    end

    @testset "SmoothDoubleGaussianPDF" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        original_pdf = DoubleGaussianPDF(FT)
        smooth_pdf = SmoothDoubleGaussianPDF(FT)
        @test keys(closure_parameters(SimplifiedHigherOrderClosure(FT; assumed_pdf = smooth_pdf))) ==
              keys(closure_parameters(SimplifiedHigherOrderClosure(FT)))

        # One level at 900 hPa and 285 K, close to saturation, between interfaces that carry the
        # same moments (w3, thl_sec, wthl_sec, qwthl_sec, wqw_sec, qw_sec)
        pressure, θˡ = FT(9e4), FT(285) * (FT(1e5) / FT(9e4))^(constants.rgas / constants.cp)
        qˢ = first(plume_saturation(FT(285), pressure, constants))
        pdf_of(pdf, qᵗ, w_sec, moments; policy = hard) =
            assumed_pdf(pdf, policy, constants, FT(1020), FT(980), FT(1000), pressure, θˡ, zero(FT), qᵗ, w_sec, moments, moments)

        # Away from the switches (resolved variances, plumes well apart) the smooth PDF is the
        # original one as its tolerances vanish, and within a few percent of it with the
        # tolerances of shoc.F90, which are a sixth of the standard deviation of total water here
        sharp_pdf = SmoothDoubleGaussianPDF(FT; thl_tol = 1e-5, rt_tol = 1e-7, w_tol = 1e-4, minimum_saturation_spread = 1e-9)
        w_sec = FT(0.6)
        for w3 in FT[-0.2, 0.05, 0.4], qᵗ in (FT(0.97), FT(1.0), FT(1.03)) .* qˢ
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
            cloud_fraction, liquid, liquid_variance, liquid_flux, buoyancy_flux = pdf_of(pdf, FT(1.02) * qˢ, calm, still)
            @test isapprox(cloud_fraction, 1; atol = 1e-3)
            @test liquid > 0
            @test liquid_flux == 0
            @test first(pdf_of(pdf, FT(0.98) * qˢ, calm, still)) < 1e-3
        end
        @test isapprox(pdf_of(smooth_pdf, FT(1.02) * qˢ, calm, still)[2], pdf_of(original_pdf, FT(1.02) * qˢ, calm, still)[2]; rtol = 1e-2)

        # No jump where the switches of the original are: the outputs are continuous in the
        # vertical velocity variance across w_tol² and in the scalar variance across its tolerance²
        variable = (FT(1e-4), FT(0.02), FT(-2e-3), FT(-2e-6), FT(3e-6), FT(4e-8))
        step(pdf, f) = maximum(abs, f(pdf, 1 + FT(1e-3)) .- f(pdf, 1 - FT(1e-3)))
        across_calm(pdf, x) = pdf_of(pdf, qˢ, original_pdf.w_tol^2 * x, variable)[1:2]
        across_uniform(pdf, x) = pdf_of(pdf, qˢ, FT(0.3), (variable[1], variable[2], variable[3], variable[4], variable[5],
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
        @test isapprox(first(pdf_of(smooth_pdf, qˢ, calm, still)), 0.5; atol = 0.05)

        for clipping in (HardClipping(), SmoothClipping(FT; width = 1e-2))
            run = run_shoc_column(SimplifiedHigherOrderClosure(FT; assumed_pdf = SmoothDoubleGaussianPDF(), clipping, surface_forcing...))
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test all(≥(0), run.cloud_liquid)
            @test isapprox(run.θ, original.θ; rtol = 2e-3)
        end
    end

    @testset "RichardsonStabilityFunctions" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        richardson = RichardsonStabilityFunctions(FT)
        @test richardson.functions isa RationalStabilityFunction{FT}
        @test isbits(richardson)
        @test summary(richardson) == "RichardsonStabilityFunctions with RationalStabilityFunction{$FT}"

        # The shear term of the TKE update is its ℂᴾ times the squared shear
        tke_update = ForwardEulerTKE(FT; ℂᴾ = 0.5)
        N², S², e, ℓ = FT(1e-4), FT(4e-4), FT(0.3), FT(120)
        Ri = N² / (S² + richardson.minimum_shear_squared)
        functions = richardson.functions
        @test time_scale_column(richardson, hard, constants, FT(0.01)) === nothing
        @test dissipation_coefficient(richardson, nothing, N², tke_update.ℂᴾ * S², tke_update) ≈
              Breeze.TurbulenceClosures.dissipation_stability_function(functions, Ri)
        τ, Kᶜ, Kᵘ = time_scale_and_diffusivities(richardson, hard, constants, nothing, e, FT(1e-3), ℓ, N², tke_update.ℂᴾ * S²,
                                                 tke_update, false)
        @test Kᶜ ≈ Breeze.TurbulenceClosures.tracer_stability_function(functions, Ri) * ℓ * sqrt(e)
        @test Kᵘ ≈ Breeze.TurbulenceClosures.momentum_stability_function(functions, Ri) * ℓ * sqrt(e)
        τ₀ = richardson.ℂᵗ * ℓ / sqrt(e)
        @test τ ≈ τ₀ / sqrt(1 + τ₀^2 * N²)
        @test τ^2 * N² < 1
        # In unstable air the time scale is the turnover time
        τ, _, _ = time_scale_and_diffusivities(richardson, hard, constants, nothing, e, FT(1e-3), ℓ, -N², tke_update.ℂᴾ * S²,
                                               tke_update, false)
        @test τ ≈ τ₀

        # The coefficients of the stability functions are coefficients of the closure
        closure = SimplifiedHigherOrderClosure(FT; turbulence_time_scale = RichardsonStabilityFunctions(),
                                               tke_update = ForwardEulerTKE(ℂᴾ = 1), surface_forcing...)

        # The functions may be given first, under their own name; without any, they are the
        # rational functions, and the closure holds its own copy of their coefficients
        @test RichardsonStabilityFunctions(RationalStabilityFunction(), FT) === RichardsonStabilityFunctions(FT)
        @test RichardsonStabilityFunctions(RationalStabilityFunction(ℂᶜ⁰ = 0.3), FT; ℂᵗ = 2) ===
              RichardsonStabilityFunctions(FT; functions = RationalStabilityFunction(ℂᶜ⁰ = 0.3), ℂᵗ = 2)
        @test SimplifiedHigherOrderClosure(FT; turbulence_time_scale = RichardsonStabilityFunctions(RationalStabilityFunction()),
                                           tke_update = ForwardEulerTKE(ℂᴾ = 1), surface_forcing...) === closure
        @test isbits(closure)
        ℂ = closure_parameters(closure)
        @test ℂ.ℂᶜ⁰ == functions.ℂᶜ⁰ && ℂ.Ri⁺ == functions.Ri⁺ && ℂ.ℂᵗ == 1
        @test !haskey(ℂ, :ℂᵝ₁)
        changed = with_parameters(closure; ℂᶜ⁰ = FT(0.3), ℂᵗ = FT(2))
        @test changed.turbulence_time_scale.functions.ℂᶜ⁰ == FT(0.3)
        @test changed.turbulence_time_scale.ℂᵗ == FT(2)
        @test typeof(changed) === typeof(closure)

        for tke_update in (ForwardEulerTKE(ℂᴾ = 1), ImplicitSinkTKE(ℂᴾ = 1))
            closure = SimplifiedHigherOrderClosure(FT; turbulence_time_scale = RichardsonStabilityFunctions(), tke_update,
                                                   surface_forcing...)
            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test all(≥(0), run.cloud_liquid)
            @test isapprox(run.θ, original.θ; rtol = 2e-3)
        end
    end

    @testset "StratificationLimitedTimeScale" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        damping = ColumnStabilityDamping(FT)
        limited = StratificationLimitedTimeScale(FT)
        @test limited.time_scale === damping
        @test limited.ℂᴺ == FT(3.5)
        @test isbits(limited)
        @test summary(limited) == "StratificationLimitedTimeScale of ColumnStabilityDamping{$FT}"
        @test StratificationLimitedTimeScale(ColumnStabilityDamping(), FT; ℂᴺ = 2) === StratificationLimitedTimeScale(FT; ℂᴺ = 2)
        @test StratificationLimitedTimeScale(RichardsonStabilityFunctions(), FT).time_scale === RichardsonStabilityFunctions(FT)

        # The default is the original configuration, whose time scale is not limited
        @test SimplifiedHigherOrderClosure(FT).turbulence_time_scale === damping
        for τ in FT[1, 100, 2e4], N² in FT[-1e-3, 0, 1e-4, 1]
            @test same_bits(limit_time_scale(damping, τ, N²), τ)
            @test same_bits(limit_time_scale(RichardsonStabilityFunctions(FT), τ, N²), τ)
        end

        # ℂᴺ is a trainable coefficient of the closure, after those of the component it wraps
        closure = SimplifiedHigherOrderClosure(FT; turbulence_time_scale = limited, surface_forcing...)
        @test isbits(closure)
        ℂ = closure_parameters(closure)
        original_keys = keys(closure_parameters(SimplifiedHigherOrderClosure(FT)))
        @test keys(ℂ) == (original_keys[1:findfirst(==(:ℂᴰ₅), original_keys)]..., :ℂᴺ,
                          original_keys[findfirst(==(:ℂᴰ₅), original_keys)+1:end]...)
        @test :ℂᴺ ∈ trainable_parameter_names(closure)
        changed = with_parameters(closure; ℂᴺ = FT(5), ℂᶜ = FT(0.2))
        @test changed.turbulence_time_scale.ℂᴺ == FT(5)
        @test changed.turbulence_time_scale.time_scale.ℂᶜ == FT(0.2)
        @test typeof(changed) === typeof(closure)

        # The column quantities, dissipation, fallback and diffusivities are those of the wrapped component
        column = time_scale_column(damping, hard, constants, FT(0.05))
        @test time_scale_column(limited, hard, constants, FT(0.05)) === column
        @test dissipation_coefficient(limited, column, FT(1e-4), FT(1e-5), ForwardEulerTKE(FT)) ===
              dissipation_coefficient(damping, column, FT(1e-4), FT(1e-5), ForwardEulerTKE(FT))
        @test cold_surface_fallback(limited, FT(170), FT(10), FT(100)) && !cold_surface_fallback(limited, FT(290), FT(10), FT(100))

        # The limit: 1/τ = 1/τ₀ + N⁺/ℂᴺ, so τ ≤ τ₀ and τ N⁺ < ℂᴺ for every τ₀ and N²; the identity where N² ≤ 0
        Nₘᵢₙ = limited.minimum_buoyancy_frequency
        fraction(n, m) = FT(mod(n * 0.6180339887498949 + m * 0.7548776662466927, 1))
        for n in 1:500
            e = FT(10)^(-3 + 3 * fraction(n, 1))
            mix = FT(10)^(3 * fraction(n, 2))
            N² = (fraction(n, 3) - FT(0.3)) * FT(10)^(-8 + 7 * fraction(n, 4))
            runaway = false
            a_diss = column.Cee * e^(FT(3) / 2) / mix
            τ₀, Kᶜ₀, Kᵘ₀ = time_scale_and_diffusivities(damping, hard, constants, column, e, a_diss, mix, N², FT(1e-5), ForwardEulerTKE(FT), runaway)
            # The levels carry the time scale and diffusivities of the wrapped component; the limit acts at the interfaces
            level = time_scale_and_diffusivities(limited, hard, constants, column, e, a_diss, mix, N², FT(1e-5), ForwardEulerTKE(FT), runaway)
            @test all(map(same_bits, level, (τ₀, Kᶜ₀, Kᵘ₀)))
            τ = limit_time_scale(limited, τ₀, N²)
            @test τ ≤ τ₀
            if N² ≤ 0
                @test same_bits(τ, τ₀)
            else
                N = sqrt(N²)
                N⁺ = N² / (sqrt(N² + Nₘᵢₙ^2) + Nₘᵢₙ)
                @test isapprox(τ, τ₀ / (1 + τ₀ * N⁺ / limited.ℂᴺ); rtol = 8eps(FT))
                @test τ * N⁺ ≤ limited.ℂᴺ * (1 + 4eps(FT))
                # αᴺ = τ² N² < (ℂᴺ)² where N ≫ Nₘᵢₙ, and below (ℂᴺ N / (N - Nₘᵢₙ))² everywhere
                @test τ^2 * N² < limited.ℂᴺ^2 * (N > Nₘᵢₙ ? (N / (N - Nₘᵢₙ))^2 : FT(Inf)) * (1 + 8eps(FT))
                N > 100Nₘᵢₙ && @test τ^2 * N² < limited.ℂᴺ^2 * (1 + FT(0.0203))
            end
        end
        # αᴺ < (ℂᴺ)² once N ≥ 0.01: at τ₀ = maxiso and the strongest stratification
        for N² in FT[1e-4, 4e-4, 1e-2, 1]
            τ = limit_time_scale(limited, FT(2e4), N²)
            @test τ^2 * N² < limited.ℂᴺ^2
        end
        # With ℂᴺ = Inf it is the identity
        unlimited = StratificationLimitedTimeScale(FT; ℂᴺ = Inf)
        for τ in FT[1, 100, 2e4], N² in FT[-1e-3, 0, 1e-8, 1e-4, 1]
            @test same_bits(limit_time_scale(unlimited, τ, N²), τ)
        end

        if FT == Float64
            # Slopes against the exact ones and central differences: in N² (continuous value at N² = 0,
            # slope -τ₀² / (2 ℂᴺ Nₘᵢₙ) from above and 0 from below), in ℂᴺ and in τ₀
            τ₀ = 400.0
            limited_by_N²(N², ts, τ₀) = limit_time_scale(ts, τ₀, N²)
            limited_by_ℂᴺ(ℂᴺ, τ₀, N²) = limit_time_scale(StratificationLimitedTimeScale(Float64; ℂᴺ), τ₀, N²)
            limited_by_τ₀(τ₀, ts, N²) = limit_time_scale(ts, τ₀, N²)
            ℂᴺ = limited.ℂᴺ
            h = 1e-6
            for N² in (-1e-4, 0.0, 1e-8, 1e-6, 1e-4, 1e-2)
                N²⁺ = max(N², 0.0)
                N⁺ = sqrt(N²⁺ + Nₘᵢₙ^2) - Nₘᵢₙ
                dN⁺ = N² > 0 ? 1 / (2 * sqrt(N²⁺ + Nₘᵢₙ^2)) : (N² == 0 ? 1 / (2Nₘᵢₙ) : 0.0)
                τ = τ₀ / (1 + τ₀ * N⁺ / ℂᴺ)
                exact_N² = -τ^2 / ℂᴺ * dN⁺
                N² == 0 || @test isapprox(derivative(limited_by_N², N², limited, τ₀), exact_N²; rtol = 1e-10)
                N² == 0 && @test any(slope -> isapprox(derivative(limited_by_N², N², limited, τ₀), slope; rtol = 1e-10), (0.0, exact_N²)) # a one-sided slope
                exact_ℂᴺ = τ^2 * N⁺ / ℂᴺ^2
                @test isapprox(derivative(limited_by_ℂᴺ, ℂᴺ, τ₀, N²), exact_ℂᴺ; rtol = 1e-10, atol = 1e-14)
                @test isapprox((limited_by_ℂᴺ(ℂᴺ + h, τ₀, N²) - limited_by_ℂᴺ(ℂᴺ - h, τ₀, N²)) / 2h, exact_ℂᴺ; rtol = 1e-6, atol = 1e-9)
                exact_τ₀ = (τ / τ₀)^2
                @test isapprox(derivative(limited_by_τ₀, τ₀, limited, N²), exact_τ₀; rtol = 1e-10)
                if N² ≠ 0
                    step = h * abs(N²)
                    difference = (limited_by_N²(N² + step, limited, τ₀) - limited_by_N²(N² - step, limited, τ₀)) / 2step
                    @test isapprox(difference, exact_N²; rtol = 1e-5, atol = 1e-6)
                end
                @test all(isfinite, (exact_N², exact_ℂᴺ, exact_τ₀))
            end
            # The value is continuous at N² = 0 and the slope is bounded there
            @test limited_by_N²(1e-14, limited, τ₀) ≈ τ₀ rtol = 1e-6
            @test abs(derivative(limited_by_N², 1e-14, limited, τ₀)) < τ₀^2 / (2 * ℂᴺ * Nₘᵢₙ) * (1 + 1e-6)

            # The diagnosed third moment through the limited time scale, in ℂᴺ and in N²: Enzyme against central
            # differences, from weak stratification to beyond the singularity of the unlimited diagnosis
            diagnosis = AlgebraicThirdMoment(Float64)
            rest = (0.4, 300.0, 1 / 40, 1 / 80, 0.02, -0.004, 0.1, 0.15, 0.03)
            moment_by_ℂᴺ(ℂᴺ, τ₀, N², rest) = third_moment(diagnosis, UnboundedStratificationNumber(), constants,
                                                          limit_time_scale(StratificationLimitedTimeScale(Float64; ℂᴺ), τ₀, N²), N², rest...)
            moment_by_N²(N², ℂᴺ, τ₀, rest) = moment_by_ℂᴺ(ℂᴺ, τ₀, N², rest)
            for αᴺ₀ in (-5.0, 0.5, 5.0, 24.26, 60.0, 250.0)
                N² = αᴺ₀ / τ₀^2
                for (func, x, parameters) in ((moment_by_ℂᴺ, 3.5, (τ₀, N², rest)), (moment_by_ℂᴺ, 2.5, (τ₀, N², rest)),
                                              (moment_by_N², N², (3.5, τ₀, rest)))
                    step = h * abs(x)
                    difference = (func(x + step, parameters...) - func(x - step, parameters...)) / 2step
                    gradient = derivative(func, x, parameters...)
                    @test isfinite(gradient)
                    @test isapprox(gradient, difference; rtol = 1e-5, atol = 1e-12)
                end
                # Beyond the singularity of the unlimited diagnosis the limited one is finite and moderate
                τ = limit_time_scale(limited, τ₀, N²)
                @test τ^2 * N² < ℂᴺ^2
            end
        end

        # Steps in a column, of both wrapped components; with ℂᴺ = Inf the same bits as the component itself
        for (wrapped, tke_update, rtol) in ((ColumnStabilityDamping(), ForwardEulerTKE(), 1e-4),
                                            (RichardsonStabilityFunctions(), ForwardEulerTKE(ℂᴾ = 1), 2e-3))
            closure = SimplifiedHigherOrderClosure(FT; turbulence_time_scale = StratificationLimitedTimeScale(wrapped), tke_update,
                                                   surface_forcing...)
            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test isapprox(run.θ, original.θ; rtol)

            model = shoc_column_model(closure)
            for _ in 1:20
                time_step!(model, 60)
            end
            # At the interfaces, where the moments use it, the limited time scale keeps τ N⁺ < ℂᴺ, however
            # large the stratification number of the interpolated τ₀ and N² would be without it
            τ₀ = Array(interior(model.closure_fields.diagnostics.isotropy))[1, 1, :]
            N² = Array(interior(model.closure_fields.diagnostics.brunt))[1, 1, :]
            τ₀ᶠ, N²ᶠ = (τ₀[1:end-1] .+ τ₀[2:end]) ./ 2, (N²[1:end-1] .+ N²[2:end]) ./ 2
            τᶠ = limit_time_scale.(Ref(closure.turbulence_time_scale), τ₀ᶠ, N²ᶠ)
            N⁺ = @. max(N²ᶠ, 0) / (sqrt(max(N²ᶠ, 0) + Nₘᵢₙ^2) + Nₘᵢₙ)
            @test all(τᶠ .* N⁺ .≤ closure.turbulence_time_scale.ℂᴺ * (1 + 4eps(FT)))
            @test all(τᶠ .≤ τ₀ᶠ)

            itself = run_shoc_column(SimplifiedHigherOrderClosure(FT; turbulence_time_scale = wrapped, tke_update, surface_forcing...))
            infinite = run_shoc_column(SimplifiedHigherOrderClosure(FT; turbulence_time_scale = StratificationLimitedTimeScale(wrapped; ℂᴺ = Inf),
                                                                    tke_update, surface_forcing...))
            # the same bits on the CPU; on the GPU the two kernels may contract floating-point operations differently
            if default_arch isa CPU
                @test all(name -> isequal(infinite[name], itself[name]), keys(itself))
            else
                @test all(name -> isapprox(infinite[name], itself[name]; rtol = sqrt(eps(FT))), keys(itself))
            end
        end
    end

    @testset "GradientLimitedMixingLength" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        ml = GradientLimitedMixingLength(FT(0.8), FT(1.8))
        closure = SimplifiedHigherOrderClosure(FT; mixing_length = ml, surface_forcing...)
        @test isbits(closure)
        @test closure.mixing_length === ml
        ℂ = closure_parameters(closure)
        @test ℂ.ℂˢ == FT(0.8) && ℂ.ℂⁱ == FT(1.8)
        @test !haskey(ℂ, :ℂˡ₅)      # no dimensional coefficient is left
        @test with_parameters(closure; ℂⁱ = FT(1.5)).mixing_length.ℂⁱ == FT(1.5)

        # The adapter computes the lengths of the TKE closure for the same e and N²: a stable
        # layer between two neutral ones, with turbulence that decays upward
        model = shoc_column_model(closure)
        grid = model.grid
        Nz = size(grid, 3)
        fields = model.closure_fields
        state, host, scratch = fields.state, fields.host, fields.scratch
        @test keys(scratch) == (:du, :dl, :d, :gradient_limited_length, :stratification_envelope, :interface_stratification)

        z = znodes(grid, Center())
        set!(state.tke, z -> 0.02 + 0.5 * exp(-z / 800))
        set!(host.thv, z -> 300 + 5e-3 * clamp(z - 1000, 0, 800))
        set!(host.zt_grid, z -> z)

        context = @allowscalar mixing_length_column!(ml, hard, constants, 1, 1, grid, Nz, zero(FT), state, host,
                                                     fields.diagnostics, scratch)
        @test context.minimum_tke == constants.mintke
        lengths = @allowscalar [mixing_length(ml, hard, constants, context, 1, 1, k, grid, zero(FT), zero(FT), zero(FT),
                                              FT(1e5), state, scratch) for k in 1:Nz]

        # The same through the TKE closure and its own stratification field
        tke_closure = TKEBasedTurbulenceClosure(; mixing_length = ml, minimum_tke = constants.mintke)
        ℓ, ℓˢ = ZFaceField(grid), ZFaceField(grid)
        N² = scratch.interface_stratification
        @allowscalar Breeze.TurbulenceClosures.fill_mixing_length!(ℓ, ℓˢ, 1, 1, grid, ml, tke_closure, state.tke, N²)
        expected = @allowscalar [Breeze.TurbulenceClosures.mixing_lengthᶜᶜᶜ(1, 1, k, grid, tke_closure, state.tke, (; ℓˢ, N²))
                                 for k in 1:Nz]
        @test Array(interior(scratch.gradient_limited_length)) == Array(interior(ℓ))
        @test Array(interior(scratch.stratification_envelope)) == Array(interior(ℓˢ))
        @test lengths == expected
        @test all(isfinite, lengths) && all(>(0), lengths)

        # N² is that of the virtual potential temperature across each interior interface
        N²_column = vec(Array(interior(N²)))
        θᵥ = vec(Array(interior(host.thv)))
        @test N²_column[1] == 0 && N²_column[end] == 0
        @test isapprox(maximum(N²_column), constants.ggr / 302 * 5e-3; rtol = 2e-2)
        @test (N²_column[2:Nz] .> 0) == (diff(θᵥ) .> 0)
        @test any(>(0), N²_column) && any(==(0), N²_column[2:Nz])

        # In the neutral layer below, the length is the wall length ℂˢ z until the stable layer
        # above limits it; a host mesh finer than the length caps it
        @test lengths[1] ≈ ml.ℂˢ * @allowscalar(z[1])
        @test maximum(lengths) < ml.ℂˢ * 3000
        capped = @allowscalar mixing_length(ml, hard, constants, context, 1, 1, 2, grid, zero(FT), zero(FT), zero(FT),
                                            FT(1), state, scratch)
        @test capped == 1

        for (label, closure) in ("SHOC diffusivities" => closure,
                                 "Richardson diffusivities" =>
                                     SimplifiedHigherOrderClosure(FT; mixing_length = GradientLimitedMixingLength(FT(0.8), FT(1.8)),
                                                                  turbulence_time_scale = RichardsonStabilityFunctions(),
                                                                  tke_update = ImplicitSinkTKE(ℂᴾ = 1), surface_forcing...))
            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test maximum(run.e) > FT(0.01)
        end
    end

    @testset "smooth_shoc_components" begin
        components = smooth_shoc_components(FT)
        @test keys(components) == (:mixing_length, :turbulence_time_scale, :tke_update, :stratification_number_bound, :skewness, :assumed_pdf, :clipping)
        @test smooth_shoc_components() === components
        @test smooth_shoc_components(Float32).clipping === SmoothClipping(Float32(1e-2))

        # The set, and the set with SHOC's own mixing length and time scale named after it
        smooth_closure = SimplifiedHigherOrderClosure(FT; components..., surface_forcing...)
        own = SimplifiedHigherOrderClosure(FT; components...,
                                           mixing_length = BlendedMixingLength(),
                                           turbulence_time_scale = ColumnStabilityDamping(),
                                           tke_update = ImplicitSinkTKE(),
                                           surface_forcing...)

        # The switches are smooth in both
        for closure in (smooth_closure, own)
            @test isbits(closure)
            @test closure.clipping === SmoothClipping(FT(1e-2))
            @test closure.assumed_pdf isa SmoothDoubleGaussianPDF{FT}
            @test closure.third_moment isa AlgebraicThirdMoment{FT}
            @test closure.stratification_number_bound isa RationalSaturatedStratificationNumber{FT}
            @test closure.skewness isa PolynomialSaturatedSkewness{FT}
            @test closure.tke_update isa ImplicitSinkTKE{FT}
            @test closure.buoyancy_source isa PDFBuoyancyFlux
            @test all(value -> value isa FT, values(closure_parameters(closure)))

            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(run.e .> 0)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test all(≥(0), run.cloud_liquid)
            @test maximum(run.e) > FT(0.01)
            @test isapprox(run.θ, original.θ; rtol = 5e-3)
        end

        # The set has the mixing length and the stability functions that the TKE closure also
        # uses, starting from its published coefficients, and no dimensional coefficient
        published = TKEBasedTurbulenceClosure()
        @test smooth_closure.mixing_length == published.mixing_length
        @test smooth_closure.turbulence_time_scale.functions == published.stability_functions
        @test smooth_closure.tke_update.ℂᴾ == 1
        ℂ = closure_parameters(smooth_closure)
        @test !haskey(ℂ, :ℂˡ₅)

        # The closure holds its own copies of the coefficients of these components: they are
        # coefficients of the closure, all trainable, and are replaced like the others
        reused = (fieldnames(GradientLimitedMixingLength)..., fieldnames(RationalStabilityFunction)...)
        @test length(reused) == 16
        @test all(name -> haskey(ℂ, name), reused)
        @test all(name -> name ∈ trainable_parameter_names(smooth_closure), reused)
        recalibrated = with_parameters(smooth_closure; NamedTuple{reused}(map(name -> 2 * ℂ[name], reused))...)
        @test recalibrated.mixing_length.ℂˢ == 2 * published.mixing_length.ℂˢ
        @test recalibrated.turbulence_time_scale.functions.Ri⁺ == 2 * published.stability_functions.Ri⁺
        @test typeof(recalibrated) === typeof(smooth_closure)

        # Only the formulations are shared with the TKE closure: every coefficient of the reused
        # components is a coefficient of the SHOC closure, replaced independently of the TKE closure
        ℂ = closure_parameters(smooth_closure)
        shared = (:ℂˢ, :ℂⁱ, :ℂᵘ⁻, :ℂᵘ⁰, :ℂᵘ⁺, :ℂᶜ⁻, :ℂᶜ⁰, :ℂᶜ⁺, :ℂᵉ⁻, :ℂᵉ⁰, :ℂᵉ⁺, :ℂᴰ⁻, :ℂᴰ⁰, :ℂᴰ⁺, :Ri⁻, :Ri⁺, :ℂᵗ, :ℂᴾ)
        @test all(name -> haskey(ℂ, name), shared)
        doubled = with_parameters(smooth_closure; NamedTuple{shared}(map(name -> 2 * ℂ[name], shared))...)
        @test all(name -> closure_parameters(doubled)[name] == 2 * ℂ[name], shared)
        @test doubled.mixing_length.ℂˢ == 2 * published.mixing_length.ℂˢ
        @test doubled.turbulence_time_scale.functions.Ri⁺ == 2 * published.stability_functions.Ri⁺
        @test TKEBasedTurbulenceClosure().mixing_length == published.mixing_length
        # and the column responds to each (but to ℂⁱ: in this column the envelope of the stable
        # layer is nowhere below the wall length)
        unchanged = run_shoc_column(SimplifiedHigherOrderClosure(FT; components..., surface_forcing...))
        for name in (:ℂˢ, :ℂᵘ⁰, :ℂᶜ⁺, :ℂᴰ⁰, :Ri⁺, :ℂᴾ)
            changed = with_parameters(SimplifiedHigherOrderClosure(FT; components..., surface_forcing...); NamedTuple{(name,)}((FT(1.5) * ℂ[name],))...)
            @testset "response to $name" begin
                @test run_shoc_column(changed).θ != unchanged.θ
            end
        end

        # SHOC's own mixing length and time scale, with the published coefficients of shoc.F90
        @test own.mixing_length === SimplifiedHigherOrderClosure(FT).mixing_length
        @test own.turbulence_time_scale === SimplifiedHigherOrderClosure(FT).turbulence_time_scale
        @test own.tke_update.ℂᴾ == FT(0.1)
        @test keys(closure_parameters(own)) == keys(closure_parameters(SimplifiedHigherOrderClosure(FT)))

        # A keyword after the set replaces one of its components
        @test SimplifiedHigherOrderClosure(FT; components..., clipping = HardClipping()).clipping isa HardClipping
    end

    @testset "Every combination of components" begin
        choices = (mixing_length = (BlendedMixingLength(), GradientLimitedMixingLength()),
                   turbulence_time_scale = (ColumnStabilityDamping(), RichardsonStabilityFunctions(RationalStabilityFunction()),
                                            StratificationLimitedTimeScale()),
                   tke_update = (ForwardEulerTKE(), ImplicitSinkTKE()),
                   buoyancy_source = (PDFBuoyancyFlux(), DowngradientBuoyancyFlux()),
                   stratification_number_bound = (UnboundedStratificationNumber(), RationalSaturatedStratificationNumber()),
                   skewness = (ResetAboveThresholdSkewness(), ClampedSkewness(), PolynomialSaturatedSkewness()),
                   assumed_pdf = (DoubleGaussianPDF(), SmoothDoubleGaussianPDF()),
                   clipping = (HardClipping(), SmoothClipping()))

        combinations = vec([NamedTuple{keys(choices)}(combination) for combination in Iterators.product(values(choices)...)])
        @test length(combinations) == 576

        # Every combination constructs, and its coefficients are collected and replaced
        constructed = map(combinations) do components
            closure = SimplifiedHigherOrderClosure(FT; components..., surface_forcing...)
            ℂ = closure_parameters(closure)
            held = filter(is_calibratable, mapreduce(component_scalar_names, (a, b) -> (a..., b...), values(components)))
            scaled = map(value -> value * FT(1.25), ℂ)
            changed = with_parameters(closure; scaled...)
            promoted = with_parameters(closure; ℂᴾ = big(0.5))

            isbits(closure) &&
                all(slot -> getfield(closure, slot) === convert_eltype(FT, components[slot]), keys(components)) &&
                all(value -> value isa FT, values(ℂ)) &&
                allunique(keys(ℂ)) &&
                all(name -> haskey(ℂ, name), held) &&
                all(name -> name ∈ keys(ℂ), trainable_parameter_names(closure)) &&
                typeof(changed) === typeof(closure) &&
                closure_parameters(changed) == scaled &&
                changed.constants === closure.constants &&
                changed.surface_moisture_flux === closure.surface_moisture_flux &&
                all(value -> value isa BigFloat, values(closure_parameters(promoted))) &&
                promoted.tke_update.ℂᴾ == big(0.5)
        end
        @test all(constructed)

        # A component in a slot that is not its own is refused by the constructor
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; mixing_length = ColumnStabilityDamping())
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; mixing_length = IntegralMixingLength())
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; clipping = ClampedSkewness())
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; mixing_length = :gradient_limited)

        # A sample of the combinations in which every component occurs takes steps in a column,
        # and again after its closure is rebuilt with other coefficients and materialized
        sample = combinations[1:19:end]
        @test all(slot -> all(choice -> any(components -> components[slot] === choice, sample), choices[slot]), keys(choices))

        for components in sample
            closure = SimplifiedHigherOrderClosure(FT; components..., surface_forcing...)
            model = shoc_column_model(closure)
            for _ in 1:10
                time_step!(model, 60)
            end

            ℂ = closure_parameters(model.closure)
            rebuilt = with_parameters(model.closure; ℂᴾ = ℂ.ℂᴾ * FT(1.1), ℂʷ₂ = ℂ.ℂʷ₂ * FT(0.9))
            model.closure = materialize_closure(rebuilt, closure_scalar_names(model), model.grid.architecture)
            @test model.closure.tke_update.ℂᴾ == ℂ.ℂᴾ * FT(1.1)
            for _ in 1:10
                time_step!(model, 60)
            end

            diagnostics = model.closure_fields.diagnostics
            @testset "$(join(map(summary, values(components)), ", "))" begin
                @test all(isfinite, Array(interior(model.formulation.potential_temperature_density)))
                @test all(isfinite, Array(interior(model.moisture_density)))
                @test all(isfinite, Array(interior(model.tracers.ρe)))
                @test all(isfinite, Array(interior(diagnostics.w3)))
                @test all(x -> 0 ≤ x ≤ 1, Array(interior(diagnostics.shoc_cldfrac)))
                @test all(≥(0), Array(interior(diagnostics.shoc_ql)))
            end
        end
    end
end
