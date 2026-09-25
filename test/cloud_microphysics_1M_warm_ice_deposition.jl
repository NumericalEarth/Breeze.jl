include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.ThermodynamicsInterface as TDI
using Oceananigans
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

# Rebuild a Microphysics1MParams with selected `process_params` entries overridden
# (CloudMicrophysics 0.38 stores process parameters separately from the option markers).
override_process_params(parameters; overrides...) =
    CMP.Microphysics1MParams(;
        parameters.processes,
        process_params = merge(parameters.process_params, (; overrides...)),
        parameters.cloud,
        parameters.precip,
        parameters.air_properties,
        parameters.terminal_velocity,
    )

@testset "MPNE1M suppresses warm cloud-ice growth [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    microphysics = OneMomentCloudMicrophysics(FT;
                                              cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)))

    T = FT(276)
    qᵛ = FT(0.007)
    qᶜˡ = FT(0)
    qᶜⁱ = FT(0)
    qʳ = FT(0)
    qˢⁿ = FT(0)

    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qᶜⁱ + qˢⁿ)
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(101325)), T, constants)
    ρ = density(𝒰, constants)

    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    @test qᵛ > qᵛ⁺ⁱ

    ℳ = BreezeCloudMicrophysicsExt.MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢⁿ, zero(FT))
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
        zero(FT), # vertical velocity
        qᵛ + qᶜˡ + qᶜⁱ + qʳ + qˢⁿ,
        qᶜˡ,
        qᶜⁱ,
        qʳ,
        qˢⁿ,
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

    evaluate_tendencies = function (parameters, T, qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢⁿ;
                                    freezing_temperature = FT(273.15))
        categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories(
            FT;
            parameters,
            freezing_temperature,
        )
        cloud_formation = NonEquilibriumCloudFormation(nothing, CloudIce(FT))
        microphysics = OneMomentCloudMicrophysics(FT; categories, cloud_formation)
        q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, qᶜⁱ + qˢⁿ)
        𝒰 = with_temperature(
            LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(101325)),
            T,
            constants,
        )
        ρ = density(𝒰, constants)
        ℳ = BreezeCloudMicrophysicsExt.MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢⁿ, zero(FT))
        tendencies = @inferred BreezeCloudMicrophysicsExt.mpne1m_tendencies(
            microphysics,
            ρ,
            ℳ,
            𝒰,
            constants,
        )
        return tendencies, microphysics
    end

    disabled_microphysics = CMP.Microphysics1MParams(FT; disabled_options...)

    # Supersaturation-dependent ice autoconversion transfers cloud ice to snow.
    options = merge(disabled_options, (; snow_autoconversion = CMP.WithSupersaturation()))
    parameters = override_process_params(CMP.Microphysics1MParams(FT; options...);
                                         snow_autoconversion = (; r_ice_snow = FT(25e-6)))
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(1e-4), FT(0), FT(0))
    @test tendencies.ρqᶜⁱ < 0
    @test tendencies.ρqˢⁿ > 0
    @test tendencies.ρqᶜⁱ ≈ -tendencies.ρqˢⁿ

    # SublimationOnly suppresses supersaturated snow deposition.
    options = merge(disabled_options, (; snow_deposition_sublimation = CMP.SublimationOnly()))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(0), FT(0), FT(1e-4))
    @test all(iszero, tendencies)

    options = merge(disabled_options, (; snow_deposition_sublimation = CMP.DepositionAndSublimation()))
    parameters = CMP.Microphysics1MParams(FT; options...)
    tendencies, = evaluate_tendencies(parameters, FT(250), FT(0.01), FT(0), FT(0), FT(0), FT(1e-4))
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqˢⁿ > 0

    # Numerical repair remains active when physical cloud formation is disabled.
    tendencies, = evaluate_tendencies(
        disabled_microphysics,
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
        disabled_microphysics,
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
        disabled_microphysics,
        FT(250),
        FT(0.001),
        FT(0),
        FT(1e-4),
        FT(0),
        FT(-1e-4),
    )
    @test tendencies.ρqᵛ < 0
    @test tendencies.ρqˢⁿ > 0
    @test sum(tendencies) ≈ zero(FT) atol=eps(FT)

    # TemperatureDependent formation uses the Frostenberg deposition timescale.
    frostenberg = CMP.Frostenberg2023(; σ = FT(1), a = FT(1), b = FT(1), T_freeze = FT(273.15))
    options = merge(disabled_options, (; cloud_ice_formation = CMP.TemperatureDependent()))
    parameters = override_process_params(CMP.Microphysics1MParams(FT; options...);
                                         cloud_ice_formation = (; τ_relax = FT(10), frostenberg))
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

# Kessler rain autoconversion parameters whose timescale and threshold depend on the vertical velocity:
# a 10× faster conversion and a halved threshold in convective conditions.
function convective_rain_autoconversion(parameters)
    acnv = parameters.process_params.rain_autoconversion
    convective = CMP.KesslerAcnv(; acnv.τ_slow, τ_fast = acnv.τ_slow / 10,
                                 acnv.q_threshold_slow, q_threshold_fast = acnv.q_threshold_slow / 2,
                                 acnv.w_0, acnv.k)
    return override_process_params(parameters; rain_autoconversion = convective)
end

@testset "Rain autoconversion depends on the vertical velocity [$(FT)]" for FT in test_float_types()
    default = CMP.Microphysics1MParams(FT)
    @test default.process_params.rain_autoconversion isa CMP.KesslerAcnv
    convective = convective_rain_autoconversion(default)
    w₀ = convective.process_params.rain_autoconversion.w_0
    qᶜˡ = FT(1e-3)

    autoconversion(parameters, w) = BreezeCloudMicrophysicsExt.liquid_autoconversion(parameters, qᶜˡ, FT(w))

    # Default parameters: slow and fast values are equal, so the rate does not depend on w
    @test autoconversion(default, 0) > 0
    @test autoconversion(default, 10) == autoconversion(default, 0)
    @test autoconversion(default, -10) == autoconversion(default, 0)

    # Velocity-dependent parameters: quiescent rate at rest, faster in up- and downdrafts
    @test autoconversion(convective, 0) == autoconversion(default, 0)
    @test autoconversion(convective, w₀) > autoconversion(convective, 0)
    @test autoconversion(convective, 10w₀) > autoconversion(convective, w₀)
    @test autoconversion(convective, -w₀) == autoconversion(convective, w₀)
end

function autoconversion_test_model(::Type{FT}, cloud_formation, parameters) where FT
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants, base_pressure=101325, potential_temperature=290)
    categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories(FT; parameters)
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation, categories)
    return AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics)
end

# Rain tendency in the middle of a domain with uniform cloud liquid, no rain, and vertical velocity `w`.
# Without rain there is no rain advection, sedimentation, accretion, or evaporation, so the
# tendency is the autoconversion alone.
function rain_tendency_with_vertical_velocity(model, w)
    if haskey(model.microphysical_fields, :ρqᶜˡ)
        set!(model; θ=290, qᵗ=0.012)
        set!(model; qᶜˡ=1e-3)
    else # saturation adjustment: supersaturate to make cloud liquid
        set!(model; θ=290, qᵗ=0.02)
    end

    # Prescribe w directly, bypassing the pressure projection in `set!`
    interior(model.velocities.w, :, :, 2:4) .= w
    Breeze.AtmosphereModels.compute_tendencies!(model)
    precipitation = compute!(precipitation_rate(model, :liquid))

    return (tendency = Array(interior(model.timestepper.Gⁿ.ρqʳ))[1, 1, 2:3],
            precipitation = Array(interior(precipitation))[1, 1, 2:3])
end

@testset "Rain autoconversion in AtmosphereModel follows the vertical velocity [$(FT)]" for FT in test_float_types()
    default = CMP.Microphysics1MParams(FT)
    convective = convective_rain_autoconversion(default)
    w = 5 * convective.process_params.rain_autoconversion.w_0

    for (name, cloud_formation) in (("warm non-equilibrium", NonEquilibriumCloudFormation(CloudLiquid(FT), nothing)),
                                    ("mixed-phase non-equilibrium", NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))),
                                    ("warm saturation adjustment", SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())))
        @testset "$name" begin
            model = autoconversion_test_model(FT, cloud_formation, convective)
            at_rest = rain_tendency_with_vertical_velocity(model, 0)
            moving = rain_tendency_with_vertical_velocity(model, w)
            @test all(at_rest.tendency .> 0)
            @test all(moving.tendency .> at_rest.tendency)
            @test all(moving.precipitation .> at_rest.precipitation)

            # Default parameters: no dependence on w
            model = autoconversion_test_model(FT, cloud_formation, default)
            @test rain_tendency_with_vertical_velocity(model, w) == rain_tendency_with_vertical_velocity(model, 0)
        end
    end
end
