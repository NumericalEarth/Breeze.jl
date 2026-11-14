include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: Thermodynamics, PlanarLiquidSurface, ThermodynamicConstants, CondensedPhase, saturation_vapor_pressure,
                             dry_air_gas_constant, vapor_gas_constant
using Test

# Conversions used to map the ICMW Pi-chamber protocol (mixing ratio Qv, temperature T; Chen et al. 2025, case description
# doi:10.7910/DVN/JWRBE7) onto Breeze's prognostic variables (specific humidity qᵛ, potential temperature θ; status notes 2026-11-14).

# Five-point Gauss–Hermite quadrature of a standard normal variable: exact moments through order 9
const gauss_hermite = ((0.0, 0.9453087204829419), (1.3556261799742675, 0.3936193231522412), (-1.3556261799742675, 0.3936193231522412),
                       (2.8569700138728056, 0.019953242059045913), (-2.8569700138728056, 0.019953242059045913))
normal_moments(f, μ, σ) = begin
    mean = sum(w * f(μ + σ * sqrt(2) * x) for (x, w) in gauss_hermite) / sqrt(π)
    var = sum(w * (f(μ + σ * sqrt(2) * x) - mean)^2 for (x, w) in gauss_hermite) / sqrt(π)
    return mean, sqrt(var)
end

@testset "Mixing ratio to specific humidity: σ_q = σ_r / (1 + r̄)² to second order" begin
    r̄ = 0.0104126     # ICMW case description §1.1, Q̄v₀
    σ_r = 0.4775e-3    # ICMW σ_Qv
    q(r) = r / (1 + r)
    q̄, σ_q = normal_moments(q, r̄, σ_r)
    linear = σ_r / (1 + r̄)^2
    # q = r/(1+r) is smooth with q′ = r′/(1+r̄)² − r′²/(1+r̄)³ + …; for a symmetric r′ the odd term does not change the variance and
    # the relative correction to σ_q is of order (σ_r/(1+r̄))², about 2e-7 here — the linear formula is not exact, but its error is
    # bounded by a few times (σ_r/(1+r̄))²
    bound = 3 * (σ_r / (1 + r̄))^2
    @test abs(σ_q / linear - 1) ≤ bound
    @test abs(σ_q / linear - 1) > 0 # the nonlinearity is real, if tiny
    @test abs(q̄ - q(r̄)) ≤ 3 * σ_r^2 / (1 + r̄)^3 # the mean shifts by the second-order term
    @test linear / σ_r ≈ 1 / (1 + r̄)^2 # the 2.05% factor between the two standard deviations
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
    @test abs(S_breeze - S_ncar) < 0.02 # the two conventions differ by a fraction of a percent-point; the driver reports both
end
