include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: clip_max, clip_min, clip_below, clip_between, clip_negative,
                                      stable_damping, hard_floor, third_moment_bound, bound_third_moment,
                                      vd_shoc_decomp!, vd_shoc_solve!, tke_step, buoyancy_production,
                                      assumed_pdf, plume_saturation, plume_variance, smooth_scalar_plumes,
                                      time_scale_column, dissipation_coefficient, time_scale_and_diffusivities,
                                      mixing_length_column!, mixing_length
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

        closure = SHOC(OriginalSHOC(), FT; tke_update = ImplicitSinkTKE(), surface_forcing...)
        @test isbits(closure)
        @test closure_parameters(closure).ℂᴾ == FT(0.1)
        @test haskey(shoc_column_model(closure).closure_fields.scratch, :tke_sink_rate)
        @test !haskey(shoc_column_model(SHOC(OriginalSHOC(), FT)).closure_fields.scratch, :tke_sink_rate)

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

        downgradient = run_shoc_column(SHOC(OriginalSHOC(), FT; buoyancy_source = DowngradientBuoyancyFlux(), surface_forcing...))
        @test all_finite(downgradient)
        @test maximum(downgradient.e) > FT(0.01)
        @test isapprox(downgradient.θ, original.θ; rtol = 2e-3)
    end

    @testset "SmoothDoubleGaussianPDF" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        original_pdf = DoubleGaussianPDF(FT)
        smooth_pdf = SmoothDoubleGaussianPDF(FT)
        @test keys(closure_parameters(SHOC(OriginalSHOC(), FT; assumed_pdf = smooth_pdf))) ==
              keys(closure_parameters(SHOC(OriginalSHOC(), FT)))

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
        @test ratio₁ ≈ plume_variance(hard, true, x̃₁, x̃₂, FT(0.5), zero(FT), one(FT))
        @test ratio₂ ≈ plume_variance(hard, false, x̃₁, x̃₂, FT(0.5), zero(FT), one(FT))

        # A spread of the saturation deficit that does not vanish: the cloud fraction passes
        # through one half at saturation instead of jumping
        @test isapprox(first(pdf_of(smooth_pdf, qˢ, calm, still)), 0.5; atol = 0.05)

        for clipping in (HardClipping(), SmoothClipping(FT; width = 1e-2))
            run = run_shoc_column(SHOC(OriginalSHOC(), FT; assumed_pdf = SmoothDoubleGaussianPDF(), clipping, surface_forcing...))
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
        closure = SHOC(OriginalSHOC(), FT; turbulence_time_scale = RichardsonStabilityFunctions(),
                       tke_update = ForwardEulerTKE(ℂᴾ = 1), surface_forcing...)
        @test isbits(closure)
        ℂ = closure_parameters(closure)
        @test ℂ.ℂᶜ⁰ == functions.ℂᶜ⁰ && ℂ.Ri⁺ == functions.Ri⁺ && ℂ.ℂᵗ == 1
        @test !haskey(ℂ, :ℂᵝ₁)
        changed = with_parameters(closure; ℂᶜ⁰ = FT(0.3), ℂᵗ = FT(2))
        @test changed.turbulence_time_scale.functions.ℂᶜ⁰ == FT(0.3)
        @test changed.turbulence_time_scale.ℂᵗ == FT(2)
        @test typeof(changed) === typeof(closure)

        for tke_update in (ForwardEulerTKE(ℂᴾ = 1), ImplicitSinkTKE(ℂᴾ = 1))
            closure = SHOC(OriginalSHOC(), FT; turbulence_time_scale = RichardsonStabilityFunctions(), tke_update,
                           surface_forcing...)
            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test all(≥(0), run.cloud_liquid)
            @test isapprox(run.θ, original.θ; rtol = 2e-3)
        end
    end

    @testset "GradientLimitedMixingLength" begin
        hard = HardClipping()
        constants = SHOCConstants(FT)
        ml = GradientLimitedMixingLength(FT(0.8), FT(1.8))
        closure = SHOC(OriginalSHOC(), FT; mixing_length = ml, surface_forcing...)
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
                                     SHOC(OriginalSHOC(), FT; mixing_length = GradientLimitedMixingLength(FT(0.8), FT(1.8)),
                                          turbulence_time_scale = RichardsonStabilityFunctions(),
                                          tke_update = ImplicitSinkTKE(ℂᴾ = 1), surface_forcing...))
            run = run_shoc_column(closure)
            @test all_finite(run)
            @test all(x -> 0 ≤ x ≤ 1, run.cloud_fraction)
            @test maximum(run.e) > FT(0.01)
        end
    end

    @testset "DifferentiableSHOC" begin
        @test DifferentiableSHOC() === DifferentiableSHOC(core = :tke_closure)
        @test_throws ArgumentError DifferentiableSHOC(core = :mass_flux)

        # The smoothed switches are those of both cores
        for core in (:tke_closure, :shoc)
            closure = SHOC(DifferentiableSHOC(; core), FT; surface_forcing...)
            @test isbits(closure)
            @test closure.clipping === SmoothClipping(FT(1e-2))
            @test closure.assumed_pdf isa SmoothDoubleGaussianPDF{FT}
            @test closure.third_moment isa CanutoThirdMoment{TanhBound, FT}
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

        # The default core is the eddy diffusivity of the TKE closure, with its published
        # coefficients and no dimensional coefficient
        closure = SHOC(DifferentiableSHOC(), FT)
        published = TKEBasedTurbulenceClosure()
        @test closure.mixing_length == published.mixing_length
        @test closure.turbulence_time_scale.functions == published.stability_functions
        @test closure.tke_update.ℂᴾ == 1
        @test !haskey(closure_parameters(closure), :ℂˡ₅)

        # SHOC's own mixing length and time scale, with the published coefficients of shoc.F90
        own = SHOC(DifferentiableSHOC(core = :shoc), FT)
        @test own.mixing_length === SHOC(OriginalSHOC(), FT).mixing_length
        @test own.turbulence_time_scale === SHOC(OriginalSHOC(), FT).turbulence_time_scale
        @test own.tke_update.ℂᴾ == FT(0.1)

        # A keyword replaces a component of the preset
        @test SHOC(DifferentiableSHOC(), FT; clipping = HardClipping()).clipping isa HardClipping
    end
end
