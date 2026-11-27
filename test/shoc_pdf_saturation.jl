include(joinpath(@__DIR__, "setup.jl"))

#####
##### The saturation thermodynamics of the assumed PDF of the subgrid distribution (the model's, with the phase of the
##### condensate of its microphysics), compared with the fixed-constant saturation of shoc.F90, written out below
#####

using Breeze
using Breeze.Thermodynamics: CondensedPhase, WarmPhaseEquilibrium, MixedPhaseEquilibrium, DryExponent,
                             VaporEquivalentExponent, saturation_vapor_pressure, saturation_vapor_pressure_log_slope,
                             PlanarLiquidSurface, PlanarIceSurface, liquid_latent_heat, ice_latent_heat,
                             MoistureMassFractions, LiquidIcePotentialTemperatureState, temperature,
                             mixture_gas_constant, mixture_heat_capacity, vapor_equivalent_heat_capacity,
                             dry_air_gas_constant, vapor_gas_constant, exner_exponent
using Breeze.AtmosphereModels.Diagnostics: condensate_phase_equilibrium
using Breeze.Microphysics: NonEquilibriumCloudFormation
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: SimplifiedHigherOrderClosures, assumed_pdf, component_saturation,
                                                               liquid_water_temperature, buoyancy_flux_coefficients,
                                                               distribution_thermodynamics, component_vapor_excess
using Oceananigans
using Test

#####
##### The saturation of shoc.F90 (`shoc_assumed_pdf_compute_temperature`, `shoc_assumed_pdf_compute_qs` and the
##### buoyancy flux of `shoc_assumed_pdf`), with its literals: liquid only, the vapor pressure of Murphy and Koop
##### (2005), and the constants E3SM hands SHOC (`shr_const`) and the reference values of shoc.F90. The
##### thermodynamics of the test, so that the assumed PDF of the closure can be evaluated with the saturation of shoc.F90.
#####

# `rgas`, `rv`, `cp`, `lcond`, `basetemp` and `basepres` of shoc.F90
shoc_constants(FT) = (dry_air_gas_constant = FT(287.042), vapor_gas_constant = FT(461.505), heat_capacity = FT(1004.64),
                      latent_heat = FT(2.501e6), reference_potential_temperature = FT(300), standard_pressure = FT(100000))

struct ShocThermodynamics{C}
    constants :: C
end

# wv_sat_scream.F90, MurphyKoop_svp over liquid
function murphy_koop_saturation_vapor_pressure(T::FT) where FT
    ln_T = log(T)
    exponent = FT(54.842763) - (FT(6763.22) / T) - (FT(4.210) * ln_T) + (FT(0.000367) * T) +
               (tanh(FT(0.0415) * (T - FT(218.8))) * (FT(53.878) - (FT(1331.22) / T) - (FT(9.44523) * ln_T) + FT(0.014025) * T))
    return exp(exponent)
end

shoc_exner_function(c, p) = (p / c.standard_pressure)^(c.dry_air_gas_constant / c.heat_capacity)

SimplifiedHigherOrderClosures.liquid_water_temperature(shoc::ShocThermodynamics, θˡⁱ, qᵗ, p) =
    shoc_exner_function(shoc.constants, p) * θˡⁱ

function SimplifiedHigherOrderClosures.component_saturation(shoc::ShocThermodynamics, Tˡ, qᵗ, p)
    FT = typeof(Tˡ)
    c = shoc.constants
    pᵛ⁺ = murphy_koop_saturation_vapor_pressure(Tˡ)
    qᵛ⁺ = FT(0.622) * pᵛ⁺ / max(pᵛ⁺, p - pᵛ⁺)
    β = (c.dry_air_gas_constant / c.vapor_gas_constant) * (c.latent_heat / (c.dry_air_gas_constant * Tˡ)) * (c.latent_heat / (c.heat_capacity * Tˡ))
    return qᵛ⁺, β, c.heat_capacity / c.latent_heat, shoc_exner_function(c, p)
end

function SimplifiedHigherOrderClosures.buoyancy_flux_coefficients(shoc::ShocThermodynamics, p, θˡⁱ, qᵗ, qˡ)
    c = shoc.constants
    ε = c.dry_air_gas_constant / c.vapor_gas_constant
    Aᵗ = ((1 - ε) / ε) * c.reference_potential_temperature
    Aᶜ = (c.latent_heat / c.heat_capacity) / shoc_exner_function(c, p) - (1 / ε) * c.reference_potential_temperature
    return Aᵗ, Aᶜ
end

# Thermodynamic constants of Breeze with the constants of shoc.F90: the gas constants, one
# heat capacity for dry air, vapor and liquid (so the latent heat is constant) and SHOC's latent heat
function matching_constants(FT, c)
    ℛ = 8.314462618
    liquid = CondensedPhase(FT; reference_latent_heat = c.latent_heat, heat_capacity = c.heat_capacity, density = 1000)
    return ThermodynamicConstants(FT; molar_gas_constant = ℛ, dry_air_molar_mass = ℛ / c.dry_air_gas_constant,
                                  vapor_molar_mass = ℛ / c.vapor_gas_constant,
                                  dry_air_heat_capacity = c.heat_capacity, vapor_heat_capacity = c.heat_capacity, liquid)
end

matching_thermodynamics(FT, c) = (constants = matching_constants(FT, c), standard_pressure = c.standard_pressure,
                                  exner_exponent = DryExponent(), phase_equilibrium = WarmPhaseEquilibrium())

model_thermodynamics(FT; phase_equilibrium = WarmPhaseEquilibrium()) =
    (constants = ThermodynamicConstants(FT), standard_pressure = FT(1e5), exner_exponent = VaporEquivalentExponent(),
     phase_equilibrium)

@testset "The distribution condenses on the phase of the condensate of the microphysics [$FT]" for FT in test_float_types()
    mixed = MixedPhaseEquilibrium(FT)
    @test condensate_phase_equilibrium(nothing) isa WarmPhaseEquilibrium
    @test condensate_phase_equilibrium(SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium())) isa WarmPhaseEquilibrium
    @test condensate_phase_equilibrium(SaturationAdjustment(FT; equilibrium = mixed)) === mixed
    @test condensate_phase_equilibrium(InstantaneousPrecipitation(FT; equilibrium = mixed)) === mixed
    @test condensate_phase_equilibrium(Breeze.DCMIP2016KesslerMicrophysics(FT)) isa WarmPhaseEquilibrium
    @test condensate_phase_equilibrium(NonEquilibriumCloudFormation(:liquid, nothing)) isa WarmPhaseEquilibrium
    # Schemes with prognostic cloud liquid distribute vapor and cloud liquid: their condensate is the cloud liquid,
    # relaxed toward saturation over liquid; their ice is not distributed
    @test condensate_phase_equilibrium(NonEquilibriumCloudFormation(:liquid, :ice)) isa WarmPhaseEquilibrium
    @test condensate_phase_equilibrium(PredictedParticlePropertiesMicrophysics(FT)) isa WarmPhaseEquilibrium

    # The thermodynamics of the distribution are those of the model, with the phase of its microphysics
    grid = RectilinearGrid(default_arch, FT; size = 4, z = (0, 1000), topology = (Flat, Flat, Bounded))
    for microphysics in (SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium()), SaturationAdjustment(FT; equilibrium = mixed))
        model = AtmosphereModel(grid; microphysics, closure = SimplifiedHigherOrderClosure(FT))
        thermodynamics = distribution_thermodynamics(model)
        @test thermodynamics.constants === model.thermodynamic_constants
        @test thermodynamics.phase_equilibrium === microphysics.equilibrium
    end
end

@testset "The model's saturation agrees with that of shoc.F90 with matching constants [$FT]" for FT in test_float_types()
    c = shoc_constants(FT)
    shoc = ShocThermodynamics(c)
    thermodynamics = matching_thermodynamics(FT, c)
    constants = thermodynamics.constants
    roundoff = FT == Float32 ? 1e-5 : 1e-12

    for p in FT[1e5, 9e4, 7e4], T in FT[275, 285, 295, 303]
        qᵗ = FT(0.01)
        θˡⁱ = T / shoc_exner_function(c, p)
        @test liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p) ≈
              liquid_water_temperature(shoc, θˡⁱ, qᵗ, p) rtol = roundoff

        qᵛ⁺ˢ, βˢ, cᵖℒ⁻¹ˢ, Πˢ = component_saturation(shoc, T, qᵗ, p)
        qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(thermodynamics, T, qᵗ, p)
        # Same constants: β = ℒ² / (Rᵛ cᵖ T²) in both (the model's ∂ ln qᵛ⁺ / ∂T at fixed pressure and
        # composition of the component's air is ∂ ln pᵛ⁺ / ∂T)
        @test β ≈ βˢ rtol = roundoff
        @test cᵖᵐℒ⁻¹ ≈ cᵖℒ⁻¹ˢ rtol = roundoff
        @test ∂Tˡ∂θ ≈ Πˢ rtol = roundoff
        # The vapor pressure of the model (temperature-dependent latent heat) is within 4 % of
        # that of Murphy and Koop in warm conditions (2.6 % at 285 K); qᵛ⁺ also carries the factor (1 - qᵗ)
        @test qᵛ⁺ ≈ qᵛ⁺ˢ * (1 - qᵗ) rtol = 0.04

        # The buoyancy flux coefficients are those of the model's buoyancy, a function of Rᵐ T: with these constants
        # (one heat capacity, a dry Exner exponent) and the level at the reference potential temperature of shoc.F90,
        # Aᵗ = θ (Rᵛ - Rᵈ) / Rᵐ and Aᶜ = ℒ / (cᵖ Π) - θ Rᵛ / Rᵐ, which are those of shoc.F90 with the gas constant
        # Rᵐ of the level's air in place of that of dry air
        qˡ = FT(2e-4)
        θ = c.reference_potential_temperature
        θˡⁱ₀ = θ - (c.latent_heat / c.heat_capacity) / shoc_exner_function(c, p) * qˡ
        Aᵗˢ, Aᶜˢ = buoyancy_flux_coefficients(shoc, p, θˡⁱ₀, qᵗ, qˡ)
        Aᵗ, Aᶜ = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ₀, qᵗ, qˡ)
        Rᵈ = dry_air_gas_constant(constants)
        Rᵛ = vapor_gas_constant(constants)
        Rᵐ = mixture_gas_constant(MoistureMassFractions(qᵗ - qˡ, qˡ, zero(FT)), constants)
        @test Aᵗ * Rᵐ / Rᵈ ≈ Aᵗˢ rtol = 10 * roundoff
        @test Aᶜ + θ * (Rᵛ / Rᵐ - Rᵛ / Rᵈ) ≈ Aᶜˢ rtol = 10 * roundoff
    end

    # The assumed PDF of a level at 900 hPa and 285 K near saturation: the two agree within the
    # difference of the vapor pressures
    p = FT(9e4)
    θˡⁱ = FT(285) / shoc_exner_function(c, p)
    qᵛ⁺ˢ = first(component_saturation(shoc, FT(285), zero(FT), p))
    moments = (FT(0.05), FT(0.25), FT(-0.25), FT(-1.3e-4), FT(2.9e-4), FT(4e-7))
    for pdf in (DoubleGaussianPDF(FT), SmoothDoubleGaussianPDF(FT)), qᵗ in (FT(0.99), FT(1.02), FT(1.05)) .* qᵛ⁺ˢ
        with_shoc = assumed_pdf(pdf, shoc, FT(1020), FT(980), FT(1000), p, θˡⁱ, zero(FT), qᵗ, FT(0.6), moments, moments)
        with_model = assumed_pdf(pdf, thermodynamics, FT(1020), FT(980), FT(1000), p, θˡⁱ, zero(FT), qᵗ, FT(0.6), moments, moments)
        @test isapprox(with_model[1], with_shoc[1]; atol = 0.1)                         # cloud fraction
        @test isapprox(with_model[2], with_shoc[2]; atol = 0.15 * with_shoc[2] + 2e-5)  # liquid water
        # The buoyancy flux is a small residual of larger terms near saturation; the 2.6 % difference
        # of qᵛ⁺ moves the cloud fraction and changes it by up to 0.015 here
        @test isapprox(with_model[5], with_shoc[5]; atol = 0.02)  # buoyancy flux
    end
end

# The buoyancy of the model at fixed pressure, as a function of (θˡⁱ, qᵗ, qˡ): the logarithm of the specific volume
# Rᵐ T / p, with the model's functions
function log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    constants = thermodynamics.constants
    q = MoistureMassFractions(qᵗ - qˡ, qˡ, zero(qˡ))
    𝒰 = LiquidIcePotentialTemperatureState(θˡⁱ, q, thermodynamics.standard_pressure, p, thermodynamics.exner_exponent)
    return log(mixture_gas_constant(q, constants) * temperature(𝒰, constants) / p)
end

@testset "The buoyancy flux coefficients are the derivatives of the model's buoyancy" begin
    FT = Float64
    for thermodynamics in (model_thermodynamics(FT), matching_thermodynamics(FT, shoc_constants(FT))),
        (p, θˡⁱ, qᵗ, qˡ) in ((9e4, 290.0, 0.012, 0.0), (9e4, 290.0, 0.012, 4e-4), (7e4, 300.0, 0.006, 1e-3), (1e5, 285.0, 0.0, 0.0))

        Aᵗ, Aᶜ = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
        δθ, δq = 1e-4, 1e-8
        ∂θ = (log_specific_volume(thermodynamics, p, θˡⁱ + δθ, qᵗ, qˡ) - log_specific_volume(thermodynamics, p, θˡⁱ - δθ, qᵗ, qˡ)) / 2δθ
        ∂qᵗ = (log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ + δq, qˡ) - log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ - δq, qˡ)) / 2δq
        ∂qˡ = (log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ, qˡ + δq) - log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ, qˡ - δq)) / 2δq
        # one-sided where there is no water or no liquid to take away
        qᵗ == 0 && (∂qᵗ = (log_specific_volume(thermodynamics, p, θˡⁱ, 2δq, qˡ) - log_specific_volume(thermodynamics, p, θˡⁱ, 0.0, qˡ)) / 2δq)
        qˡ == 0 && qᵗ > 0 && (∂qˡ = (log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ, 2δq) - log_specific_volume(thermodynamics, p, θˡⁱ, qᵗ, 0.0)) / 2δq)
        @test Aᵗ ≈ ∂qᵗ / ∂θ rtol = 1e-5
        qᵗ > 0 && @test Aᶜ ≈ ∂qˡ / ∂θ rtol = 1e-5
    end
end

# The derivative of the saturation specific humidity with respect to temperature by centered
# differences, at fixed total water and pressure
function centered_slope(thermodynamics, T, qᵗ, p; δ = 1e-3)
    qᵛ⁺₊ = first(component_saturation(thermodynamics, T + δ, qᵗ, p))
    qᵛ⁺₋ = first(component_saturation(thermodynamics, T - δ, qᵗ, p))
    return (qᵛ⁺₊ - qᵛ⁺₋) / 2δ
end

@testset "Linearization about the component and mixed-phase saturation" begin
    FT = Float64
    warm = model_thermodynamics(FT)
    mixed = model_thermodynamics(FT; phase_equilibrium = MixedPhaseEquilibrium())
    constants = warm.constants
    p, qᵗ = FT(6e4), FT(2e-3)

    # β qᵛ⁺ = (ℒ / cᵖᵐ) ∂qᵛ⁺/∂T, inside and outside the ramp of the liquid fraction
    for (thermodynamics, T) in ((warm, 285.0), (warm, 250.0), (mixed, 285.0), (mixed, 255.0), (mixed, 238.0), (mixed, 225.0))
        qᵛ⁺, β, cᵖᵐℒ⁻¹, _ = component_saturation(thermodynamics, FT(T), qᵗ, p)
        @test β * qᵛ⁺ * cᵖᵐℒ⁻¹ ≈ centered_slope(thermodynamics, FT(T), qᵗ, p) rtol = 1e-5
    end
    # with the model's slope of the saturation vapor pressure and its vapor-equivalent heat capacity of the component
    qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(warm, 285.0, qᵗ, p)
    @test β * cᵖᵐℒ⁻¹ ≈ saturation_vapor_pressure_log_slope(285.0, constants, PlanarLiquidSurface()) rtol = 1e-12
    @test cᵖᵐℒ⁻¹ ≈ vapor_equivalent_heat_capacity(MoistureMassFractions(qᵗ), constants) / liquid_latent_heat(285.0, constants) rtol = 1e-12
    # and the liquid-water temperature is the model's temperature of the component with its water all vapor
    θˡⁱ = 300.0
    𝒰 = LiquidIcePotentialTemperatureState(θˡⁱ, MoistureMassFractions(qᵗ), 1e5, p)
    @test liquid_water_temperature(warm, θˡⁱ, qᵗ, p) == temperature(𝒰, constants)
    @test ∂Tˡ∂θ * θˡⁱ == temperature(𝒰, constants)

    # With condensate outside the distribution held (the rain and ice of a microphysics with prognostic cloud
    # liquid), the liquid-water temperature is the model's temperature of that state, cᵖᵐ T = ĉᵖ Π θ where θ already
    # holds the latent heat of the held condensate, and the heat capacity of the linearization is its cᵖᵐ
    holding = merge(warm, (; held = (liquid = 1e-3, ice = 5e-4)))
    q = MoistureMassFractions(qᵗ, 1e-3, 5e-4)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Π = (p / 1e5)^exner_exponent(VaporEquivalentExponent(), q, constants)
    Tˡ = liquid_water_temperature(holding, θˡⁱ, qᵗ, p)
    @test Tˡ ≈ ĉᵖ * Π * θˡⁱ / cᵖᵐ rtol = 1e-14
    @test Tˡ < Π * θˡⁱ
    qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(holding, Tˡ, qᵗ, p)
    @test cᵖᵐℒ⁻¹ ≈ cᵖᵐ / liquid_latent_heat(Tˡ, constants) rtol = 1e-12
    @test ∂Tˡ∂θ ≈ ĉᵖ * Π / cᵖᵐ rtol = 1e-14

    # Liquid above freezing, ice below homogeneous nucleation, in between in between
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) at the density ρ = p / (Rᵐ T) of the air with its total water as vapor
    saturation_over(surface, T) = (ε * (1 - qᵗ) + qᵗ) * saturation_vapor_pressure(T, constants, surface) / p
    @test first(component_saturation(mixed, 285.0, qᵗ, p)) ≈ saturation_over(PlanarLiquidSurface(), 285.0) rtol = 1e-12
    @test first(component_saturation(mixed, 225.0, qᵗ, p)) ≈ saturation_over(PlanarIceSurface(), 225.0) rtol = 1e-12
    qᵛ⁺ᵐ = first(component_saturation(mixed, 253.15, qᵗ, p))
    @test saturation_over(PlanarIceSurface(), 253.15) < qᵛ⁺ᵐ < saturation_over(PlanarLiquidSurface(), 253.15)
    @test first(component_saturation(warm, 253.15, qᵗ, p)) ≈ saturation_over(PlanarLiquidSurface(), 253.15) rtol = 1e-12

    # The latent heat of the condensate: sublimation below homogeneous nucleation
    _, _, cᵖᵐℒ⁻¹ⁱ, _ = component_saturation(mixed, 225.0, qᵗ, p)
    _, _, cᵖᵐℒ⁻¹ˡ, _ = component_saturation(warm, 225.0, qᵗ, p)
    @test cᵖᵐℒ⁻¹ˡ / cᵖᵐℒ⁻¹ⁱ ≈ ice_latent_heat(225.0, constants) / liquid_latent_heat(225.0, constants) rtol = 1e-12

    # In a supercooled level the mixed-phase components condense more, and sooner, than the liquid ones
    T = 250.0
    Π = (p / 1e5)^exner_exponent(VaporEquivalentExponent(), MoistureMassFractions(qᵗ), constants)
    θˡⁱ = T / Π
    qᵛ⁺ˡ = first(component_saturation(warm, T, qᵗ, p))
    moments = (0.05, 0.25, -0.25, -1.3e-5, 2.9e-5, 4e-9)
    for qᵗ in (0.95, 1.0) .* qᵛ⁺ˡ
        liquid = assumed_pdf(DoubleGaussianPDF(FT), warm, 1020.0, 980.0, 1000.0, p, θˡⁱ, 0.0, qᵗ, 0.6, moments, moments)
        mixed_phase = assumed_pdf(DoubleGaussianPDF(FT), mixed, 1020.0, 980.0, 1000.0, p, θˡⁱ, 0.0, qᵗ, 0.6, moments, moments)
        @test mixed_phase[1] ≥ liquid[1]
        @test mixed_phase[2] > liquid[2]
    end
end

# The vapor excess of the components that the microphysics reads is defined wherever the assumed PDF is, and at the
# same temperature: the liquid-water temperature of each component, with no lowest temperature. Here over variances
# of θˡⁱ up to (10 K)², with the components up to 15 K from the mean
@testset "The component vapor excess is that of the components the assumed PDF condenses" begin
    FT = Float64
    thermodynamics = model_thermodynamics(FT)
    constants = thermodynamics.constants
    p, θˡⁱ, qᵗ = 85000.0, 300.0, 0.012
    for pdf in (DoubleGaussianPDF(FT), EqualWidthDoubleGaussianPDF(FT), SkewedScalarDoubleGaussianPDF(FT)),
        θˡⁱ′² in (0.25, 25.0, 1e2), w′³ in (-2.0, 0.0, 2.0)

        moments = (w′³, θˡⁱ′², -0.25 * sqrt(θˡⁱ′²), -1.3e-5, 2.9e-5, 4e-9)
        cloud = assumed_pdf(pdf, thermodynamics, 1020.0, 980.0, 1000.0, p, θˡⁱ, 0.0, qᵗ, 1.0, moments, moments)
        excess = component_vapor_excess(pdf, thermodynamics, 1020.0, 980.0, 1000.0, p, θˡⁱ, 0.0, qᵗ, 1.0, moments, moments)
        @test all(isfinite, cloud)
        @test all(isfinite, excess)
        @test 0 < excess[1] < 1
        @test excess[4] ≥ 0 && excess[5] ≥ 0
        @test 0 ≤ cloud[1] ≤ 1
        @test cloud[2] ≥ 0
    end

    # Without variance the components are the mean: their vapor excess over liquid is that of the level, at its
    # liquid-water temperature
    still = ntuple(_ -> 0.0, 6)
    excess = component_vapor_excess(DoubleGaussianPDF(FT), thermodynamics, 1020.0, 980.0, 1000.0, p, θˡⁱ, 0.0, qᵗ, 1.0, still, still)
    Tˡ = liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p)
    ρ = p / (mixture_gas_constant(MoistureMassFractions(qᵗ), constants) * Tˡ)
    μ = qᵗ - Breeze.Thermodynamics.saturation_specific_humidity(Tˡ, ρ, constants, PlanarLiquidSurface())
    @test excess[2] ≈ μ rtol = 1e-12
    @test excess[3] ≈ μ rtol = 1e-12
    @test excess[4] == 0 && excess[5] == 0
end
