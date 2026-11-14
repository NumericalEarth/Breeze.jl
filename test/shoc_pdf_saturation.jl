include(joinpath(@__DIR__, "setup.jl"))

#####
##### The saturation thermodynamics of the assumed PDF of SHOC (the `pdf_saturation` slot), and
##### the condensation of SHOC-owned cloud with P3
#####

using Breeze
using Breeze.Thermodynamics: CondensedPhase, WarmPhaseEquilibrium, MixedPhaseEquilibrium, DryExponent,
                             VaporEquivalentExponent, saturation_vapor_pressure, PlanarLiquidSurface, PlanarIceSurface,
                             liquid_latent_heat, ice_latent_heat
using Breeze.TurbulenceClosures: convert_eltype
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, plume_saturation, plume_liquid_water_temperature,
                                                               buoyancy_flux_coefficients, murphy_koop_liquid_svp
using Breeze.Microphysics.PredictedParticleProperties: PrognosticVaporExcess
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Test


# Thermodynamic constants of Breeze with the constants of SHOCConstants: the gas constants, one
# heat capacity for dry air, vapor and liquid (so the latent heat is constant) and SHOC's latent heat
function matching_constants(FT, c)
    ℛ = 8.314462618
    liquid = CondensedPhase(FT; reference_latent_heat = c.lcond, heat_capacity = c.cp, density = 1000)
    return ThermodynamicConstants(FT; molar_gas_constant = ℛ, dry_air_molar_mass = ℛ / c.rgas, vapor_molar_mass = ℛ / c.rv,
                                  dry_air_heat_capacity = c.cp, vapor_heat_capacity = c.cp, liquid)
end

matching_thermodynamics(FT, c) = (constants = matching_constants(FT, c), standard_pressure = c.basepres,
                                  exner_exponent = DryExponent())

model_thermodynamics(FT) = (constants = ThermodynamicConstants(FT), standard_pressure = FT(1e5),
                            exner_exponent = VaporEquivalentExponent())

@testset "pdf_saturation slot [$FT]" for FT in test_float_types()
    closure = SimplifiedHigherOrderClosure(FT)
    @test closure.pdf_saturation isa FixedConstantsLiquidSaturation
    native = SimplifiedHigherOrderClosure(FT; pdf_saturation = ModelThermodynamicsSaturation())
    @test native.pdf_saturation isa ModelThermodynamicsSaturation{WarmPhaseEquilibrium}
    @test keys(closure_parameters(native)) == keys(closure_parameters(closure))
    @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; pdf_saturation = DoubleGaussianPDF())

    mixed = SimplifiedHigherOrderClosure(FT; pdf_saturation = ModelThermodynamicsSaturation(phase_equilibrium = MixedPhaseEquilibrium()))
    @test mixed.pdf_saturation.phase_equilibrium.freezing_temperature isa FT
    @test convert_eltype(Float32, mixed.pdf_saturation).phase_equilibrium.freezing_temperature === 273.15f0
    @test occursin("pdf_saturation: ModelThermodynamicsSaturation{MixedPhaseEquilibrium}", sprint(show, mixed))
    # The temperatures of the equilibrium are not coefficients and survive with_parameters
    @test with_parameters(mixed; ℂᶜ = 0.2).pdf_saturation == mixed.pdf_saturation
end

@testset "FixedConstantsLiquidSaturation is the saturation of shoc.F90 [$FT]" for FT in test_float_types()
    c = SHOCConstants(FT)
    sat = FixedConstantsLiquidSaturation()
    p, T = FT(9e4), FT(285)
    qs, β, cp_over_L, Π = plume_saturation(sat, nothing, c, T, FT(0.01), p)
    es = murphy_koop_liquid_svp(T)
    @test qs == FT(0.622) * es / max(es, p - es)
    @test β == (c.rgas / c.rv) * (c.lcond / (c.rgas * T)) * (c.lcond / (c.cp * T))
    @test cp_over_L == c.cp / c.lcond
    @test Π == (p / c.basepres)^(c.rgas / c.cp)
    @test plume_saturation(T, p, c) == (qs, β)

    # The assumed PDF without a saturation component is the one with FixedConstantsLiquidSaturation
    θˡ = FT(285) / Π
    moments = (FT(0.05), FT(0.25), FT(-0.25), FT(-1.3e-4), FT(2.9e-4), FT(4e-7))
    for pdf in (DoubleGaussianPDF(FT), SmoothDoubleGaussianPDF(FT)), qᵗ in (FT(0.97), FT(1.03)) .* qs
        default = assumed_pdf(pdf, HardClipping(), c, FT(1020), FT(980), FT(1000), p, θˡ, zero(FT), qᵗ, FT(0.6), moments, moments)
        given = assumed_pdf(pdf, sat, nothing, HardClipping(), c, FT(1020), FT(980), FT(1000), p, θˡ, zero(FT), qᵗ, FT(0.6),
                            moments, moments)
        @test all(map(===, default, given))
    end
end

@testset "Warm agreement of the two saturations with matching constants [$FT]" for FT in test_float_types()
    c = SHOCConstants(FT)
    fixed = FixedConstantsLiquidSaturation()
    native = ModelThermodynamicsSaturation()
    thermo = matching_thermodynamics(FT, c)
    rtol_round = FT == Float32 ? 1e-5 : 1e-12

    for p in FT[1e5, 9e4, 7e4], T in FT[275, 285, 295, 303]
        qw = FT(0.01)
        θˡ = T * (c.basepres / p)^(c.rgas / c.cp)
        @test plume_liquid_water_temperature(native, thermo, c, θˡ, qw, p) ≈
              plume_liquid_water_temperature(fixed, nothing, c, θˡ, qw, p) rtol = rtol_round

        qs_f, β_f, r_f, Π_f = plume_saturation(fixed, nothing, c, T, qw, p)
        qs_n, β_n, r_n, Π_n = plume_saturation(native, thermo, c, T, qw, p)
        es = saturation_vapor_pressure(T, thermo.constants, PlanarLiquidSurface())
        # Same constants: β differs only by the factor p / (p - e_s) of the derivative of
        # q_s = ε e_s / (p - e_s), which the fixed form drops
        @test β_n ≈ β_f * p / (p - es) rtol = rtol_round
        @test r_n ≈ r_f rtol = rtol_round
        @test Π_n ≈ Π_f rtol = rtol_round
        # The vapor pressure of the model (temperature-dependent latent heat) is within 4 % of
        # that of Murphy and Koop in warm conditions (2.6 % at 285 K); q_s also carries the factor (1 - q_w)
        @test qs_n ≈ qs_f * (1 - qw) rtol = 0.04

        # The buoyancy flux coefficients agree where the potential temperature of the level is basetemp
        ql = FT(2e-4)
        θˡ₀ = c.basetemp - (c.lcond / c.cp) * (c.basepres / p)^(c.rgas / c.cp) * ql
        A_f = buoyancy_flux_coefficients(fixed, nothing, c, p, θˡ₀, qw, ql)
        A_n = buoyancy_flux_coefficients(native, thermo, c, p, θˡ₀, qw, ql)
        @test A_n[1] ≈ A_f[1] rtol = rtol_round
        @test A_n[2] ≈ A_f[2] rtol = rtol_round
    end

    # The assumed PDF of a level at 900 hPa and 285 K near saturation: the two agree within the
    # difference of the vapor pressures
    p = FT(9e4)
    θˡ = FT(285) * (c.basepres / p)^(c.rgas / c.cp)
    qs = first(plume_saturation(fixed, nothing, c, FT(285), zero(FT), p))
    moments = (FT(0.05), FT(0.25), FT(-0.25), FT(-1.3e-4), FT(2.9e-4), FT(4e-7))
    for pdf in (DoubleGaussianPDF(FT), SmoothDoubleGaussianPDF(FT)), qᵗ in (FT(0.99), FT(1.02), FT(1.05)) .* qs
        a = assumed_pdf(pdf, fixed, nothing, HardClipping(), c, FT(1020), FT(980), FT(1000), p, θˡ, zero(FT), qᵗ, FT(0.6), moments, moments)
        b = assumed_pdf(pdf, native, thermo, HardClipping(), c, FT(1020), FT(980), FT(1000), p, θˡ, zero(FT), qᵗ, FT(0.6), moments, moments)
        @test isapprox(b[1], a[1]; atol = 0.1)                 # cloud fraction
        @test isapprox(b[2], a[2]; atol = 0.15 * a[2] + 2e-5)  # liquid water
        # The buoyancy flux is a small residual of larger terms near saturation; the 2.6 % difference
        # of q_s moves the cloud fraction and changes it by up to 0.015 here
        @test isapprox(b[5], a[5]; atol = 0.02)  # buoyancy flux
    end
end

# The derivative of the saturation specific humidity with respect to temperature by centered
# differences, at fixed total water and pressure
function centered_slope(sat, thermo, c, T, qw, p; δ = 1e-3)
    q₊ = first(plume_saturation(sat, thermo, c, T + δ, qw, p))
    q₋ = first(plume_saturation(sat, thermo, c, T - δ, qw, p))
    return (q₊ - q₋) / 2δ
end

@testset "Linearization about the plume and mixed-phase saturation" begin
    FT = Float64
    c = SHOCConstants(FT)
    thermo = model_thermodynamics(FT)
    constants = thermo.constants
    warm = ModelThermodynamicsSaturation()
    mixed = ModelThermodynamicsSaturation(phase_equilibrium = MixedPhaseEquilibrium())
    p, qw = FT(6e4), FT(2e-3)

    # β q_s = (ℒ / ĉᵖ) ∂q_s/∂T, inside and outside the ramp of the liquid fraction
    for (sat, T) in ((warm, 285.0), (warm, 250.0), (mixed, 285.0), (mixed, 255.0), (mixed, 238.0), (mixed, 225.0))
        qs, β, cp_over_L, _ = plume_saturation(sat, thermo, c, FT(T), qw, p)
        @test β * qs * cp_over_L ≈ centered_slope(sat, thermo, c, FT(T), qw, p) rtol = 1e-5
    end

    # Liquid above freezing, ice below homogeneous nucleation, in between in between
    ε = Breeze.Thermodynamics.dry_air_gas_constant(constants) / Breeze.Thermodynamics.vapor_gas_constant(constants)
    q_over(surface, T) = (e = saturation_vapor_pressure(T, constants, surface); ε * (1 - qw) * e / (p - e))
    @test first(plume_saturation(mixed, thermo, c, 285.0, qw, p)) ≈ q_over(PlanarLiquidSurface(), 285.0) rtol = 1e-12
    @test first(plume_saturation(mixed, thermo, c, 225.0, qw, p)) ≈ q_over(PlanarIceSurface(), 225.0) rtol = 1e-12
    qs_mid = first(plume_saturation(mixed, thermo, c, 253.15, qw, p))
    @test q_over(PlanarIceSurface(), 253.15) < qs_mid < q_over(PlanarLiquidSurface(), 253.15)
    @test first(plume_saturation(warm, thermo, c, 253.15, qw, p)) ≈ q_over(PlanarLiquidSurface(), 253.15) rtol = 1e-12

    # The latent heat of the condensate: sublimation below homogeneous nucleation
    _, _, cp_over_L_ice, _ = plume_saturation(mixed, thermo, c, 225.0, qw, p)
    _, _, cp_over_L_liq, _ = plume_saturation(warm, thermo, c, 225.0, qw, p)
    @test cp_over_L_liq / cp_over_L_ice ≈ ice_latent_heat(225.0, constants) / liquid_latent_heat(225.0, constants) rtol = 1e-12

    # In a supercooled level the mixed-phase plumes condense more, and sooner, than the liquid ones
    T = 250.0
    Π = first((p / 1e5)^Breeze.Thermodynamics.exner_exponent(VaporEquivalentExponent(),
                                                             Breeze.Thermodynamics.MoistureMassFractions(qw), constants))
    θˡ = T / Π
    qs_liquid = first(plume_saturation(warm, thermo, c, T, qw, p))
    moments = (0.05, 0.25, -0.25, -1.3e-5, 2.9e-5, 4e-9)
    for qᵗ in (0.95, 1.0) .* qs_liquid
        lw = assumed_pdf(DoubleGaussianPDF(FT), warm, thermo, HardClipping(), c, 1020.0, 980.0, 1000.0, p, θˡ, 0.0, qᵗ, 0.6, moments, moments)
        mx = assumed_pdf(DoubleGaussianPDF(FT), mixed, thermo, HardClipping(), c, 1020.0, 980.0, 1000.0, p, θˡ, 0.0, qᵗ, 0.6, moments, moments)
        @test mx[1] ≥ lw[1]
        @test mx[2] > lw[2]
    end
end
