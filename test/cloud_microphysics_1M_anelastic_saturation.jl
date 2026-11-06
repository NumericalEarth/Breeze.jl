include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
using Oceananigans
using Test

using Breeze.AtmosphereModels: microphysical_tendency, prognostic_field_names
using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState,
    PlanarLiquidSurface,
    density,
    saturation_specific_humidity,
    temperature,
    with_temperature

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, MixedPhaseOneMomentState, WarmPhaseOneMomentState,
                                   rain_evaporation

# The anelastic core hands the one-moment tendencies its reference density ρᵣ, a dry profile that differs
# from the air's thermodynamic state density ρᵗʰ. ρᵣ weights the tendencies and, since the prognostic
# precipitation content is ρᵣ q, sets the precipitation size distributions; saturation is evaluated at ρᵗʰ.

function air_state(FT, constants, equilibrium; T, qᵛ, qᶜˡ, qʳ)
    q = MoistureMassFractions(FT(qᵛ), FT(qᶜˡ + qʳ), zero(FT))
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(95000)), FT(T), constants)
    return one_moment_state(equilibrium, FT(qᶜˡ), FT(qʳ)), 𝒰
end

one_moment_state(::WarmPhaseEquilibrium, qᶜˡ, qʳ) = WarmPhaseOneMomentState(qᶜˡ, qʳ, zero(qᶜˡ))
one_moment_state(::MixedPhaseEquilibrium, qᶜˡ, qʳ) = MixedPhaseOneMomentState(qᶜˡ, zero(qᶜˡ), qʳ, zero(qᶜˡ), zero(qᶜˡ))

@testset "One-moment saturation does not depend on the density that weights the tendencies [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)

    # Without precipitation the one-moment rates are saturation and autoconversion rates, which involve
    # no size distribution, so the specific rates do not depend on the weighting density.
    for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium(FT)),
        cloud_formation in (SaturationAdjustment(FT; equilibrium),
                            NonEquilibriumCloudFormation(CloudLiquid(FT),
                                                         equilibrium isa MixedPhaseEquilibrium ? CloudIce(FT) : nothing))

        microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
        ℳ, 𝒰 = air_state(FT, constants, equilibrium; T=292, qᵛ=10e-3, qᶜˡ=1e-3, qʳ=0)
        ρᵗʰ = density(𝒰, constants)
        ρᵣ = FT(1.01) * ρᵗʰ

        @test microphysical_tendency(microphysics, Val(:ρqʳ), ρᵗʰ, ℳ, 𝒰, constants) != 0

        for name in prognostic_field_names(microphysics)
            weighted_by_state = microphysical_tendency(microphysics, Val(name), ρᵗʰ, ℳ, 𝒰, constants)
            weighted_by_reference = microphysical_tendency(microphysics, Val(name), ρᵣ, ℳ, 𝒰, constants)
            @test weighted_by_reference / ρᵣ ≈ weighted_by_state / ρᵗʰ
        end
    end
end

@testset "One-moment rain size distribution uses the density that weights the tendencies [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium()))
    parameters = microphysics.categories.parameters
    rain = parameters.precip.rain
    rain_velocity = parameters.terminal_velocity.rain
    air = parameters.air_properties

    # Rain evaporating into subsaturated air, with no cloud liquid for it to collect
    ℳ, 𝒰 = air_state(FT, constants, WarmPhaseEquilibrium(); T=292, qᵛ=10e-3, qᶜˡ=0, qʳ=1e-4)
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    ρᵗʰ = density(𝒰, constants)
    ρᵣ = FT(1.05) * ρᵗʰ

    # The size distribution follows the prognostic content ρᵣ qʳ; the supersaturation is at ρᵗʰ.
    evaporation = microphysical_tendency(microphysics, Val(:ρqʳ), ρᵣ, ℳ, 𝒰, constants)
    @test evaporation < 0
    @test evaporation ≈ ρᵣ * rain_evaporation(rain, rain_velocity, air, q, ℳ.qʳ, ρᵣ, ρᵗʰ, T, constants)
    @test !(evaporation / ρᵣ ≈ rain_evaporation(rain, rain_velocity, air, q, ℳ.qʳ, ρᵗʰ, ρᵗʰ, T, constants))
end

@testset "Anelastic one-moment condensation begins at the air's own saturation [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 1000), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    dynamics = AnelasticDynamics(reference_state)
    pᵣ = Array(interior(reference_state.pressure))
    ρᵣ = Array(interior(reference_state.density))
    surface = PlanarLiquidSurface()

    function column_at_relative_humidity(ℛ)
        microphysics = OneMomentCloudMicrophysics(FT; cloud_formation=NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)))
        model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants)

        # qᵛ and the temperature depend on each other through the Exner function and the moist density
        qᵗ = fill(FT(0.015), 1, 1, 4)
        for iteration in 1:5
            set!(model; θ=299, qᵗ)
            T = Array(interior(model.temperature))
            for k in 1:4
                q = MoistureMassFractions(qᵗ[k], zero(FT), zero(FT))
                ρᵐ = density(T[k], pᵣ[k], q, constants)
                qᵗ[k] = ℛ * saturation_specific_humidity(T[k], ρᵐ, constants, surface)
            end
        end
        set!(model; θ=299, qᵗ)

        # The reference density must differ enough from the air's to tell the two saturations apart
        T = Array(interior(model.temperature))
        q = MoistureMassFractions(qᵗ[1], zero(FT), zero(FT))
        @test density(T[1], pᵣ[1], q, constants) < FT(0.995) * ρᵣ[1]

        time_step!(model, 1)
        return Array(interior(model.microphysical_fields.ρqᶜˡ))
    end

    @test maximum(column_at_relative_humidity(FT(0.999))) == 0
    @test minimum(column_at_relative_humidity(FT(1.001))) > 0
end
