include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Forcings: HeldSuarezParameters, equilibrium_temperature, thermal_relaxation_rate,
                       momentum_damping_rate, hydrostatic_surface_pressure
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: MultipleForcings
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: day, hours, kilometers, minutes
using Statistics: mean
using Test

column_grid(; latitude=30, Nz=20) = LatitudeLongitudeGrid(default_arch; size=Nz, latitude, longitude=0,
                                                          z=(0, 20kilometers), topology=(Flat, Flat, Bounded))

@testset "Held–Suarez pointwise formulas [$(FT)]" for FT in test_float_types()
    parameters = HeldSuarezParameters(FT)
    rtol = max(1e-6, 100eps(FT))

    # Tᵉ = max{200, [315 - 60 sin²φ - 10 log(p/p₀) cos²φ] (p/p₀)^(2/7)}
    @test equilibrium_temperature(parameters, 0, 1e5) ≈ 315 rtol=rtol
    @test equilibrium_temperature(parameters, 45, 5e4) ≈ (285 + 5log(2)) * 2^(-2/7) rtol=rtol
    @test equilibrium_temperature(parameters, 0, 1e4) == 200

    # kᵀ = kᵃ + (kˢ - kᵃ) max(0, (σ - 0.7) / 0.3) cos⁴φ
    @test thermal_relaxation_rate(parameters, 0, 1) ≈ 1 / 4day rtol=rtol
    @test thermal_relaxation_rate(parameters, 90, 1) ≈ 1 / 40day rtol=rtol
    @test thermal_relaxation_rate(parameters, 60, 0.85) ≈ (1/40 + 9/1280) / day rtol=rtol

    # kᵘ = kᶠ max(0, (σ - 0.7) / 0.3)
    @test momentum_damping_rate(parameters, 0.85) ≈ 1 / 2day rtol=rtol
    @test momentum_damping_rate(parameters, 0.69) == 0

    custom = HeldSuarezParameters(FT; friction_rate = 2 / day, boundary_layer_top = 0.8)
    @test momentum_damping_rate(custom, 0.9) ≈ 1 / day rtol=rtol
    @test HeldSuarezParameters{Float32}(custom).friction_rate ≈ 2 / day rtol=1e-6
end

@testset "Held–Suarez forcing in a column [$(formulation), $(FT)]" for FT in test_float_types(),
                                                                      formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    previous_float_type = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    try
        Nz = 20
        φ = 30
        grid = column_grid(latitude=φ; Nz)
        model = AtmosphereModel(grid; formulation, forcing=held_suarez_forcings(Float64))
        @test model.forcing.ρE.parameters isa HeldSuarezParameters{FT}
        @test model.forcing.ρw isa Returns

        # Isothermal 280 K is colder than Tᵉ near the surface at 30°N and warmer aloft
        set!(model, T=280, u=10, v=-5)

        parameters = model.forcing.ρE.parameters
        reference = model.dynamics.reference_state
        constants = model.thermodynamic_constants
        g = constants.gravitational_acceleration
        cᵖᵈ = constants.dry_air.heat_capacity
        evaluate(forcing, k) = @allowscalar forcing(1, 1, k, grid, model.clock, fields(model))

        # Extrapolating hydrostatically from the lowest cell center recovers the base pressure
        pˢ = @allowscalar hydrostatic_surface_pressure(1, 1, 1, grid, reference.pressure, reference.density, g)
        @test pˢ ≈ reference.base_pressure rtol=1e-3

        @allowscalar for k in (1, Nz)
            p = reference.pressure[1, 1, k]
            ρ = reference.density[1, 1, k]
            T = model.temperature[1, 1, k]
            kᵘ = momentum_damping_rate(parameters, p / pˢ)
            kᵀ = thermal_relaxation_rate(parameters, φ, p / pˢ)
            @test evaluate(model.forcing.ρu, k) ≈ - kᵘ * model.momentum.ρu[1, 1, k]
            @test evaluate(model.forcing.ρv, k) ≈ - kᵘ * model.momentum.ρv[1, 1, k]
            @test evaluate(model.forcing.ρE, k) ≈ - ρ * cᵖᵈ * kᵀ * (T - equilibrium_temperature(parameters, φ, p))
        end

        @test evaluate(model.forcing.ρE, 1) > 0 > evaluate(model.forcing.ρE, Nz)
        @test evaluate(model.forcing.ρu, 1) < 0 == evaluate(model.forcing.ρu, Nz)
    finally
        Oceananigans.defaults.FloatType = previous_float_type
    end
end

@testset "Held–Suarez forcing keys and grids" begin
    grid = column_grid(latitude=45, Nz=4)
    hs = held_suarez_forcings()

    model = AtmosphereModel(grid; forcing=merge(hs, (; ρE=(hs.ρE, Returns(1)))))
    @test model.forcing.ρE isa MultipleForcings
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; u=hs.ρu))

    box = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1e5), y=(0, 1e5), z=(0, 10kilometers))
    @test_throws ArgumentError AtmosphereModel(box; forcing=hs)
end

@testset "Held–Suarez friction damps the near-surface wind" begin
    Nz = 8
    grid = LatitudeLongitudeGrid(default_arch; size=(24, 12, Nz), longitude=(0, 360), latitude=(-80, 80), z=(0, 20kilometers))

    constants = ThermodynamicConstants()
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity
    Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
    T₀ = 250
    θᵣ(z) = T₀ * exp(g * z / (cᵖᵈ * T₀))

    # An isothermal atmosphere with a zonal wind that friction damps; the frictionless twin differs only by `friction_rate = 0`
    function model_with(forcing)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5, reference_potential_temperature = θᵣ)
        model = AtmosphereModel(grid; dynamics, forcing)
        set!(model, ρ = (λ, φ, z) -> 1e5 * exp(-g * z / (Rᵈ * T₀)) / (Rᵈ * T₀), θ = (λ, φ, z) -> θᵣ(z), u = (λ, φ, z) -> 10 * cosd(φ))
        return model
    end

    friction_model = model_with(held_suarez_forcings())
    frictionless_model = model_with(held_suarez_forcings(friction_rate=0))
    update_state!(friction_model)

    pressure = friction_model.dynamics.pressure
    density = friction_model.dynamics.total_density

    # The relaxation enters ρθ as its enthalpy tendency divided by cᵖᵈ Π, and nothing else tends ρθ in this uniform state
    i, j, k = 5, 4, 2
    @allowscalar begin
        Π = (pressure[i, j, k] / friction_model.dynamics.standard_pressure)^(Rᵈ / cᵖᵈ)
        FρE = friction_model.forcing.ρE(i, j, k, grid, friction_model.clock, fields(friction_model))
        @test FρE != 0
        @test friction_model.timestepper.Gⁿ.ρθ[i, j, k] ≈ FρE / (cᵖᵈ * Π) rtol=1e-6
    end

    # Rayleigh friction decays the near-surface wind at the rate kᵘ of its lowest level
    σ₁ = @allowscalar pressure[i, j, 1] / hydrostatic_surface_pressure(i, j, 1, grid, pressure, density, g)
    kᵘ = momentum_damping_rate(friction_model.forcing.ρu.parameters, σ₁)

    for model in (friction_model, frictionless_model)
        run!(Simulation(model; Δt=30minutes, stop_time=12hours, verbose=false))
    end

    rms(model, k) = sqrt(mean(abs2, Field(model.velocities.u, indices=(:, :, k))))
    @test rms(friction_model, 1) / rms(frictionless_model, 1) ≈ exp(-kᵘ * 12hours) rtol=0.01
    @test rms(friction_model, Nz) ≈ rms(frictionless_model, Nz) rtol=1e-3
end
