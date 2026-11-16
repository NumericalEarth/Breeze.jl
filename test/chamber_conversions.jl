include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: Thermodynamics, PlanarLiquidSurface, ThermodynamicConstants, CondensedPhase, saturation_vapor_pressure,
                             dry_air_gas_constant, vapor_gas_constant
using Test

# Conversions used to map the ICMW Pi-chamber protocol (mixing ratio Qv, temperature T; Chen et al. 2025, case description
# doi:10.7910/DVN/JWRBE7) onto Breeze's prognostic variables (specific humidity qᵛ, potential temperature θ; status notes 2026-11-14).

# Five-point Gauss–Hermite quadrature for a standard normal variable (probabilists' rule, exact for polynomials of degree ≤ 9), with
# nodes and weights in closed form, evaluated in BigFloat: nodes 0, ±√(5 − √10), ±√(5 + √10); weights 8/15, (7 + 2√10)/60 (inner pair),
# (7 − 2√10)/60 (outer pair), summing to 1. This is a local moment/asymptotic check of the conversion around the mean state; it is not
# a bound over the full real line, where q = r/(1 + r) has a pole at r = −1 that the physical mixing ratio never approaches.
const gauss_hermite = let s10 = sqrt(big"10"), inner = sqrt(5 - s10), outer = sqrt(5 + s10)
    ((big"0", big"8" / 15), (inner, (7 + 2s10) / 60), (-inner, (7 + 2s10) / 60), (outer, (7 - 2s10) / 60), (-outer, (7 - 2s10) / 60))
end
normal_expectation(f) = sum(w * f(z) for (z, w) in gauss_hermite)

@testset "Mixing ratio to specific humidity: σ_q = σ_r / (1 + r̄)² to second order" begin
    # The rule reproduces the standard-normal moments through degree 9 exactly; the tolerance is a few BigFloat eps of the largest moment
    ε = 8 * eps(big"15.0")
    @test normal_expectation(z -> one(z)) ≈ 1 atol = ε
    @test abs(normal_expectation(z -> z)) < ε
    @test normal_expectation(z -> z^2) ≈ 1 atol = ε
    @test normal_expectation(z -> z^4) ≈ 3 atol = ε
    @test normal_expectation(z -> z^6) ≈ 15 atol = ε

    r̄ = big"0.0104126"     # ICMW case description §1.1, Q̄v₀
    σ_r = big"0.4775e-3"    # ICMW σ_Qv
    a = 1 + r̄
    q(x) = 1 - 1 / (a + x) # = r/(1 + r) with r = r̄ + x
    q̄ = normal_expectation(z -> q(σ_r * z))
    σ_q = sqrt(normal_expectation(z -> (q(σ_r * z) - q̄)^2))
    linear = σ_r / a^2

    # Expansion: q = const + x/a² − x²/a³ + x³/a⁴ − …, so with Gaussian moments (E x⁴ = 3σ⁴, E x⁶ = 15σ⁶)
    #   Var q = (σ²/a⁴) (1 + 8s + 69s² + 696s³ + …),  s = σ²/a²,
    # (the 8s collects Var x²/a⁶ = 2σ⁴/a⁶ and 2 Cov(x, x³)/a⁶ = 6σ⁴/a⁶), hence
    #   σ_q / (σ/a²) = 1 + 4s + 26.5 s² + O(s³).
    # The test checks the leading correction 4s with the next term as the explicit remainder bound (30 s² ≥ 26.5 s² + the s³ terms,
    # which are below 1e-3 s² for s ≈ 2.2e-7); the coefficients come from the series, not from the observed residual.
    s = (σ_r / a)^2
    @test abs(σ_q / linear - 1 - 4s) ≤ 30 * s^2
    @test σ_q / linear - 1 > 0 # the nonlinearity increases the standard deviation (4s ≈ 8.9e-7 here)
    @test abs(q̄ - q(big"0")) ≈ σ_r^2 / a^3 rtol = 10 * s # the mean shifts by E[x²]/a³ to leading order
    @test Float64(linear / σ_r) ≈ 1 / Float64(a)^2 # the 2.05% factor between the two standard deviations
end

@testset "Mean state from the pinned mixing ratio and the two saturation conventions" begin
    T₀, p₀, r₀ = 287.25, 1e5, 0.0104126 # ICMW §1.1
    q₀ = r₀ / (1 + r₀)
    @test r₀ / (1 + r₀) ≈ 0.010305 atol = 1e-6

    # NCAR convention: S = Qv/Qv_sat − 1 with Qv_sat = ε e_s/(p − e_s), e_s = 2.53e11 exp(−5420/T), ε = Rᵈ/Rᵛ with Rᵈ = 287, Rᵛ = 467
    e_s_ncar = 2.53e11 * exp(-5.42e3 / T₀)
    ε_ncar = 287 / 467
    S_ncar = r₀ / (ε_ncar * e_s_ncar / (p₀ - e_s_ncar)) - 1
    @test 0.0295 < S_ncar < 0.0315 # the protocol quotes S̄ = 3.11 %

    # Breeze convention: its own saturation vapor pressure and gas constants; the same Qv₀ gives a different supersaturation
    constants = ThermodynamicConstants(Float64)
    e_s_breeze = saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ε_breeze = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    S_breeze = r₀ / (ε_breeze * e_s_breeze / (p₀ - e_s_breeze)) - 1
    @info "Supersaturation of Q̄v₀ = 0.0104126 at 287.25 K, 1000 hPa" S_ncar S_breeze e_s_ncar e_s_breeze ε_ncar ε_breeze
    # Measured (gate 16493, 2026-11-14): S_NCAR = 0.0311368, S_Breeze = 0.0244435, a 0.669 percentage-point mismatch between the
    # reference's and Breeze's saturation thermodynamics for the same mixing ratio; it is a model-thermodynamics difference, not an
    # exact-reference match, and is tracked as unresolved in the campaign status
    @test S_ncar ≈ 0.0311368 atol = 1e-6
    @test S_breeze ≈ 0.0244435 atol = 1e-6
    @test S_ncar - S_breeze ≈ 0.00669 atol = 1e-4
end
