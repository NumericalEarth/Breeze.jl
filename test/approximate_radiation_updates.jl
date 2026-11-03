include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: update_radiation_between_solves!, _update_radiation!
using Breeze.CelestialMechanics: cos_solar_zenith_angle
using Dates
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Load both radiation extensions
using ClimaComms
using NCDatasets
using RRTMGP
using NumericalRadiation: NumericalRadiation

# A cloudy column at 45°N on the solstice, solved hourly, whose numeric clock starts at `epoch`
function hourly_column(grid; epoch, optics = GrayOptics(), surface_temperature = 300)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
    radiation = RadiativeTransferModel(grid, optics, constants;
                                       surface_temperature, surface_emissivity = 0.98, surface_albedo = 0.1,
                                       solar_position = ApparentSolarPosition(; epoch),
                                       schedule = TimeInterval(1hours))

    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, radiation)
    set!(model; θ = z -> 300 + 5e-6 * z, qᵗ = z -> 0.020 * exp(-z / 3000))
    return model
end

column(field) = vec(Array(interior(field)))
longwave_columns(radiation) = column.((radiation.upwelling_longwave_flux, radiation.downwelling_longwave_flux))
top_downwelling_shortwave(radiation) = last(column(radiation.downwelling_shortwave_flux))

@testset "Approximate radiation updates between solves [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "Longwave fluxes follow the surface temperature: $(nameof(typeof(optics)))" for optics in
            (GrayOptics(), ClearSkyOptics(), AllSkyOptics(), EcCKDOptics())

        grid = radiation_column_grid(FT, 16, 10kilometers)
        T₀ = Field{Center, Center, Nothing}(grid)
        set!(T₀, 300)
        model = hourly_column(grid; epoch = DateTime(2024, 6, 21, 12), optics, surface_temperature = T₀)
        radiation = model.radiation
        before = longwave_columns(radiation)

        set!(T₀, 305)
        update_radiation_between_solves!(radiation, model)
        updated = longwave_columns(radiation)

        _update_radiation!(radiation, model)
        solved = longwave_columns(radiation)

        # Gray longwave fluxes are linear in the surface emission, so the update matches a solve;
        # spectral ones nearly, since the spectrum of the emission shifts with the temperature
        rtol = optics isa GrayOptics ? sqrt(eps(FT)) : FT(0.02)
        @test updated[1] .- before[1] ≈ solved[1] .- before[1] rtol = rtol
        @test updated[2] .- before[2] ≈ solved[2] .- before[2] rtol = rtol
    end

    @testset "Shortwave fluxes follow the sun" begin
        epoch = DateTime(2024, 6, 21, 10)
        model = hourly_column(radiation_column_grid(FT, 16, 10kilometers); epoch)
        radiation = model.radiation
        S₀ = radiation.solar_constant

        # The solve scales its fluxes to the sun of the moment, and each update moves them with the sun
        @test top_downwelling_shortwave(radiation) ≈ -S₀ * cos_solar_zenith_angle(epoch, 0, 45) rtol = sqrt(eps(FT))

        model.clock.time = 45minutes
        update_radiation_between_solves!(radiation, model)
        @test top_downwelling_shortwave(radiation) ≈ -S₀ * cos_solar_zenith_angle(epoch + Minute(45), 0, 45) rtol = sqrt(eps(FT))
    end

    @testset "A sun rising between solves" begin
        # The sun rises at 45°N on the solstice at about 04:13 UTC on the prime meridian
        epoch = DateTime(2024, 6, 21, 3, 30)
        model = hourly_column(radiation_column_grid(FT, 16, 10kilometers); epoch)
        radiation = model.radiation
        @test all(iszero, column(radiation.downwelling_shortwave_flux))

        model.clock.time = 55minutes
        update_radiation_between_solves!(radiation, model)
        μ₀ = cos_solar_zenith_angle(epoch + Minute(55), 0, 45)
        @test μ₀ > 0
        @test top_downwelling_shortwave(radiation) ≈ -radiation.solar_constant * μ₀ rtol = sqrt(eps(FT))
        @test @allowscalar radiation.downwelling_shortwave_flux[1, 1, 1] < 0
    end

    @testset "Time stepping between solves" begin
        model = hourly_column(radiation_column_grid(FT, 16, 10kilometers); epoch = DateTime(2024, 6, 21, 8))
        radiation = model.radiation
        solved = top_downwelling_shortwave(radiation)

        for _ in 1:3
            time_step!(model, 60)
        end

        # The morning sun climbs between solves, and the column absorbs the net flux into it
        @test top_downwelling_shortwave(radiation) < solved
        ℐ_net = column(radiation.upwelling_longwave_flux) .+ column(radiation.downwelling_longwave_flux) .+
                column(radiation.upwelling_shortwave_flux) .+ column(radiation.downwelling_shortwave_flux)
        Δz = 10kilometers / 16
        @test Δz * sum(column(radiation.flux_divergence)) ≈ ℐ_net[1] - ℐ_net[end] rtol = sqrt(eps(FT))
    end
end
