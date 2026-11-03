include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: SpecificForcing, TemperatureForcing
using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity
using Oceananigans: Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: update_state!
using Test

# A humid column: 20 g/kg of vapor near the surface, so cᵖᵐ / cᵖᵈ ≈ 1.017
temperature_forcing_column(FT) = RectilinearGrid(default_arch, FT; size=4, z=(0, 400),
                                                 topology=(Flat, Flat, Bounded))

@testset "TemperatureForcing construction" begin
    grid = temperature_forcing_column(Float64)
    ∂ₜT = Returns(0.013)

    model = AtmosphereModel(grid; forcing=(; E=TemperatureForcing(∂ₜT)))
    @test model.forcing.ρE isa SpecificForcing
    @test model.forcing.ρE.forcing isa TemperatureForcing

    model = AtmosphereModel(grid; formulation=:StaticEnergy, forcing=(; E=TemperatureForcing(∂ₜT)))
    @test model.forcing.ρE.forcing isa TemperatureForcing

    # Only the specific energy key: the forcing returns cᵖᵐ ∂ₜT, a specific energy tendency
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; θ=TemperatureForcing(∂ₜT)))
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; ρE=TemperatureForcing(∂ₜT)))
end

@testset "TemperatureForcing warms humid air at the prescribed rate [$formulation, $(FT)]" for
        formulation in (:LiquidIcePotentialTemperature, :StaticEnergy), FT in test_float_types()

    Oceananigans.defaults.FloatType = FT
    grid = temperature_forcing_column(FT)
    constants = ThermodynamicConstants(FT)
    cᵖᵈ = constants.dry_air.heat_capacity
    ∂ₜT = FT(0.013)  # K s⁻¹
    qᵛ₀ = FT(0.020)
    Δt = FT(10)

    # The change of temperature over one step, for a forcing under the energy key `E`
    function warming_rate(forcing)
        model = AtmosphereModel(grid; formulation, thermodynamic_constants=constants, forcing, advection=nothing)
        set!(model; T=FT(295), qᵛ=qᵛ₀)
        T₀ = Array(interior(model.temperature))
        time_step!(model, Δt)
        T₁ = Array(interior(model.temperature))
        return (T₁ .- T₀) ./ Δt
    end

    local_conversion = warming_rate((; E=TemperatureForcing(Returns(∂ₜT))))
    dry_conversion = warming_rate((; E=Returns(cᵖᵈ * ∂ₜT)))

    cᵖᵐ = mixture_heat_capacity(MoistureMassFractions(qᵛ₀), constants)
    @test all(isapprox.(local_conversion, ∂ₜT; rtol=sqrt(eps(FT))))
    @test all(isapprox.(dry_conversion, ∂ₜT * cᵖᵈ / cᵖᵐ; rtol=sqrt(eps(FT))))
    @test 0.982 < cᵖᵈ / cᵖᵐ < 0.984  # the dry conversion heats 20 g/kg air 1.7% too little
end

@testset "TemperatureForcing uses the cloudy mixture heat capacity [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = temperature_forcing_column(FT)
    constants = ThermodynamicConstants(FT)
    cᵖᵈ = constants.dry_air.heat_capacity
    ∂ₜT = FT(0.013)
    microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())

    # Saturated: total water well above saturation at 285 K and ~1000 hPa
    function thermodynamic_tendency(forcing)
        model = AtmosphereModel(grid; microphysics, thermodynamic_constants=constants, forcing, advection=nothing)
        set!(model; θ=FT(285), qᵗ=FT(0.020))
        update_state!(model)
        qᵛ = Array(interior(model.microphysical_fields.qᵛ))
        qˡ = Array(interior(model.microphysical_fields.qˡ))
        return Array(interior(model.timestepper.Gⁿ.ρθ)), qᵛ, qˡ
    end

    Gᵗ, qᵛ, qˡ = thermodynamic_tendency((; E=TemperatureForcing(Returns(∂ₜT))))
    Gᵈ, _, _ = thermodynamic_tendency((; E=Returns(cᵖᵈ * ∂ₜT)))

    @test all(qˡ .> 0)
    cᵖᵐ = mixture_heat_capacity.(MoistureMassFractions.(qᵛ, qˡ), Ref(constants))
    @test all(isapprox.(Gᵗ ./ Gᵈ, cᵖᵐ ./ cᵖᵈ; rtol=sqrt(eps(FT))))
end
