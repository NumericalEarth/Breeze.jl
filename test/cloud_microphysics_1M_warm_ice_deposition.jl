using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.ThermodynamicsInterface as TDI
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    PlanarIceSurface,
    density,
    saturation_specific_humidity,
    with_temperature

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

@testset "MPNE1M suppresses warm cloud-ice growth [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    microphysics = OneMomentCloudMicrophysics(FT;
                                              cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)))

    T = FT(276)
    qᵛ = FT(0.007)
    qᶜˡ = FT(0)
    qᶜⁱ = FT(0)
    qʳ = FT(0)
    qˢ = FT(0)

    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qᶜⁱ + qˢ)
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(101325)), T, constants)
    ρ = density(𝒰, constants)

    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    @test qᵛ > qᵛ⁺ⁱ

    ℳ = BreezeCloudMicrophysicsExt.MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢ)
    G = BreezeCloudMicrophysicsExt.mpne1m_tendencies(microphysics, ρ, ℳ, 𝒰, constants)

    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics1MParams(FT)
    reference = BMT.bulk_microphysics_tendencies(
        BMT.Instantaneous(),
        BMT.Microphysics1Moment(),
        mp,
        tps,
        ρ,
        T,
        qᵛ + qᶜˡ + qᶜⁱ + qʳ + qˢ,
        qᶜˡ,
        qᶜⁱ,
        qʳ,
        qˢ,
    )

    @test reference.dq_icl_dt == zero(FT)
    @test G.ρqᶜⁱ / ρ == zero(FT)
end

@testset "CloudMicrophysics 1M process options [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    disabled_options = (;
        cloud_liquid_formation = nothing,
        cloud_ice_formation = nothing,
        cloud_ice_melt = nothing,
        rain_autoconversion = nothing,
        snow_autoconversion = nothing,
        rain_condensation_evaporation = nothing,
        snow_deposition_sublimation = nothing,
        snow_melt = nothing,
        cloud_liquid_rain_accretion = nothing,
        cloud_liquid_snow_accretion = nothing,
        cloud_ice_rain_accretion = nothing,
        cloud_ice_snow_accretion = nothing,
        rain_snow_accretion = nothing,
    )

    evaluate_tendencies = function (parameters, T, qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢ;
                                    freezing_temperature = FT(273.15))
        categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories(
            FT;
            parameters,
            freezing_temperature,
        )
        cloud_formation = NonEquilibriumCloudFormation(nothing, CloudIce(FT))
        microphysics = OneMomentCloudMicrophysics(FT; categories, cloud_formation)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qᶜⁱ + qˢ)
        𝒰 = with_temperature(
            LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(101325)),
            T,
            constants,
        )
        ρ = density(𝒰, constants)
        ℳ = BreezeCloudMicrophysicsExt.MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢ)
        tendencies = @inferred BreezeCloudMicrophysicsExt.mpne1m_tendencies(
            microphysics,
            ρ,
            ℳ,
            𝒰,
            constants,
        )
        return tendencies, microphysics
    end

    disabled_parameters = CMP.Microphysics1MParams(FT; disabled_options...)

    # Supersaturation-dependent ice autoconversion transfers cloud ice to snow.
    options = merge(disabled_options, (; snow_autoconversion = CMP.WithSupersaturation(FT(25e-6))))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(1e-4), FT(0), FT(0))
    @test tendencies.ρqᶜⁱ < 0
    @test tendencies.ρqˢ > 0
    @test tendencies.ρqᶜⁱ ≈ -tendencies.ρqˢ

    # SublimationOnly suppresses supersaturated snow deposition.
    options = merge(disabled_options, (; snow_deposition_sublimation = CMP.SublimationOnly()))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(0), FT(0), FT(1e-4))
    @test all(iszero, tendencies)

    options = merge(disabled_options, (; snow_deposition_sublimation = CMP.DepositionAndSublimation()))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(0), FT(0), FT(1e-4))
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqˢ > 0

    # Numerical repair remains active when physical cloud formation is disabled.
    tendencies, = evaluate_tendencies(
        disabled_parameters,
        FT(250),
        FT(0.001),
        FT(-1e-4),
        FT(0),
        FT(1e-4),
        FT(0),
    )
    @test tendencies.ρqᶜˡ > 0
    @test tendencies.ρqʳ < 0
    @test sum(tendencies) ≈ zero(FT) atol=eps(FT)

    tendencies, = evaluate_tendencies(
        disabled_parameters,
        FT(250),
        FT(0.001),
        FT(0),
        FT(-1e-4),
        FT(0),
        FT(1e-4),
    )
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqᶜⁱ > 0
    @test sum(tendencies) ≈ zero(FT) atol=eps(FT)

    tendencies, = evaluate_tendencies(
        disabled_parameters,
        FT(250),
        FT(0.001),
        FT(0),
        FT(1e-4),
        FT(0),
        FT(-1e-4),
    )
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqˢ > 0
    @test sum(tendencies) ≈ zero(FT) atol=eps(FT)

    # TemperatureDependent formation uses the Frostenberg deposition timescale.
    frostenberg = CMP.Frostenberg2023(; σ = FT(1), a = FT(1), b = FT(1), T_freeze = FT(273.15))
    option = CMP.TemperatureDependent(FT(10), frostenberg)
    options = merge(disabled_options, (; cloud_ice_formation = option))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, microphysics = evaluate_tendencies(
        parameters,
        FT(250),
        FT(0.01),
        FT(0),
        FT(1e-4),
        FT(0),
        FT(0),
    )
    @test microphysics.cloud_formation.ice isa BreezeCloudMicrophysicsExt.TemperatureDependentIceFormation
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqᶜⁱ > 0

    # CloudIceMelt transfers cloud ice to liquid and honors the category freezing temperature.
    options = merge(disabled_options, (; cloud_ice_melt = CMP.CloudIceMelt()))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(280), FT(0.001), FT(0), FT(1e-4), FT(0), FT(0))
    @test tendencies.ρqᶜˡ > 0
    @test tendencies.ρqᶜⁱ < 0
    @test tendencies.ρqᶜˡ ≈ -tendencies.ρqᶜⁱ

    tendencies, = evaluate_tendencies(
        parameters,
        FT(280),
        FT(0.001),
        FT(0),
        FT(1e-4),
        FT(0),
        FT(0);
        freezing_temperature = FT(285),
    )
    @test all(iszero, tendencies)
end
