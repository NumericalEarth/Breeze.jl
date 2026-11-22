include(joinpath(@__DIR__, "setup.jl"))

#####
##### The saturation thermodynamics of the assumed PDF of SHOC (`DoubleGaussian.saturation`), compared
##### with the fixed-constant saturation of shoc.F90, written out below
#####

using Breeze
using Breeze.Thermodynamics: CondensedPhase, WarmPhaseEquilibrium, MixedPhaseEquilibrium, DryExponent,
                             VaporEquivalentExponent, saturation_vapor_pressure, PlanarLiquidSurface, PlanarIceSurface,
                             liquid_latent_heat, ice_latent_heat
using Breeze.TurbulenceClosures: convert_eltype
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: SimplifiedHigherOrderClosures, assumed_pdf, plume_saturation,
                                                               plume_liquid_water_temperature, buoyancy_flux_coefficients
using Oceananigans
using Test

#####
##### The saturation of shoc.F90 (`shoc_assumed_pdf_compute_temperature`, `shoc_assumed_pdf_compute_qs` and the
##### buoyancy flux of `shoc_assumed_pdf`), with its literals: liquid only, the vapor pressure of Murphy and Koop
##### (2005), the fixed constants of SHOCConstants. A saturation component of the test, so that the assumed PDF of
##### the closure can be evaluated with the Fortran saturation.
#####

struct FortranSaturation end

# wv_sat_scream.F90, MurphyKoop_svp over liquid
function murphy_koop_liquid_svp(t::FT) where FT
    logt = log(t)
    tmp = FT(54.842763) - (FT(6763.22) / t) - (FT(4.210) * logt) + (FT(0.000367) * t) +
          (tanh(FT(0.0415) * (t - FT(218.8))) * (FT(53.878) - (FT(1331.22) / t) - (FT(9.44523) * logt) + FT(0.014025) * t))
    return exp(tmp)
end

SimplifiedHigherOrderClosures.plume_liquid_water_temperature(::FortranSaturation, thermodynamics, c, thl, qw, pval) =
    thl / (c.basepres / pval)^(c.rgas / c.cp)

function SimplifiedHigherOrderClosures.plume_saturation(::FortranSaturation, thermodynamics, c, Tl, qw, pval)
    FT = typeof(Tl)
    esval = murphy_koop_liquid_svp(Tl)
    qs = FT(0.622) * esval / max(esval, pval - esval)
    beta = (c.rgas / c.rv) * (c.lcond / (c.rgas * Tl)) * (c.lcond / (c.cp * Tl))
    return qs, beta, c.cp / c.lcond, (pval / c.basepres)^(c.rgas / c.cp)
end

function SimplifiedHigherOrderClosures.buoyancy_flux_coefficients(::FortranSaturation, thermodynamics, c, pval, thl, qw, ql)
    epsterm = c.rgas / c.rv
    A_q = ((1 - epsterm) / epsterm) * c.basetemp
    A_c = (c.lcond / c.cp) * (c.basepres / pval)^(c.rgas / c.cp) - (1 / epsterm) * c.basetemp
    return A_q, A_c
end


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

@testset "The saturation of the distribution of SimplifiedHigherOrderClosure [$FT]" for FT in test_float_types()
    closure = SimplifiedHigherOrderClosure(FT)
    @test closure.subgrid_distribution.saturation isa ModelThermodynamicsSaturation{WarmPhaseEquilibrium}

    mixed = SimplifiedHigherOrderClosure(FT; subgrid_distribution = Breeze.TurbulenceClosures.shoc_distribution(FT;
                                             saturation = ModelThermodynamicsSaturation(phase_equilibrium = MixedPhaseEquilibrium())))
    @test mixed isa SimplifiedHigherOrderClosure
    saturation = mixed.subgrid_distribution.saturation
    @test saturation.phase_equilibrium.freezing_temperature isa FT
    @test convert_eltype(Float32, saturation).phase_equilibrium.freezing_temperature === 273.15f0
    @test keys(closure_parameters(mixed)) == keys(closure_parameters(closure))
    # The temperatures of the equilibrium are not coefficients and survive with_parameters
    @test with_parameters(mixed; ℂᶜ = 0.2).subgrid_distribution.saturation == saturation
end

@testset "The model's saturation agrees with that of shoc.F90 with matching constants [$FT]" for FT in test_float_types()
    c = SHOCConstants(FT)
    fixed = FortranSaturation()
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
        # Same constants: β = ℒ² / (Rᵛ cᵖ T²) in both (the model's ∂ ln qᵛ⁺ / ∂T at fixed density of the
        # component's air is ∂ ln e_s / ∂T)
        @test β_n ≈ β_f rtol = rtol_round
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
    # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) at the density ρ = p / (Rᵐ T) of the air with its total water as vapor
    q_over(surface, T) = (e = saturation_vapor_pressure(T, constants, surface); (ε * (1 - qw) + qw) * e / p)
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
