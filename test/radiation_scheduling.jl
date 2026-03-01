using Breeze
using Dates: DateTime, Second
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Load RRTMGP to trigger the extension
using ClimaComms
using RRTMGP

@testset "Radiation scheduling" begin
    @testset "IterationInterval schedule [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch; size=Nz, x=FT(0), y=FT(45), z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Update radiation every 3 iterations
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           schedule = IterationInterval(3))

        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)

        # set! triggers update_state! at iteration 0 → radiation always fires
        θ(z) = 300 + FT(0.01) * z / 1000
        qᵗ(z) = FT(0.015) * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_lw_up = radiation.upwelling_longwave_flux
        flux_div = radiation.flux_divergence

        # Radiation should have been computed at iteration 0
        @allowscalar begin
            @test ℐ_lw_up[1, 1, 1] > 100
            @test any(interior(flux_div) .!= 0)
        end

        # Zero out the fluxes to detect whether radiation fires
        interior(ℐ_lw_up) .= 0
        interior(flux_div) .= 0

        # Advance to iteration 1 → should NOT update (next update at iteration 3)
        model.clock.iteration = 1
        model.clock.time = DateTime(2024, 6, 21, 16, 0, 1)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] == 0  # Radiation was skipped

        # Advance to iteration 2 → should NOT update
        model.clock.iteration = 2
        model.clock.time = DateTime(2024, 6, 21, 16, 0, 2)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] == 0  # Still skipped

        # Advance to iteration 3 → SHOULD update (multiple of 3)
        model.clock.iteration = 3
        model.clock.time = DateTime(2024, 6, 21, 16, 0, 3)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100  # Radiation fired

        # Verify the flux divergence was also recomputed
        @test any(Array(interior(flux_div)) .!= 0)
    end

    @testset "TimeInterval schedule [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch; size=Nz, x=FT(0), y=FT(45), z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Update radiation every 10 seconds (use numeric clock for TimeInterval compatibility)
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           coordinate = (FT(0), FT(45)),
                                           epoch = DateTime(2024, 6, 21, 16, 0, 0),
                                           schedule = TimeInterval(10))

        clock = Clock(time=FT(0))
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)

        θ(z) = 300 + FT(0.01) * z / 1000
        qᵗ(z) = FT(0.015) * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_lw_up = radiation.upwelling_longwave_flux
        flux_div = radiation.flux_divergence

        # Radiation should have been computed at iteration 0
        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100

        # Initialize the schedule so it tracks time from t=0
        Oceananigans.initialize!(radiation.schedule, model)

        # Zero out fluxes to detect updates
        interior(ℐ_lw_up) .= 0
        interior(flux_div) .= 0

        # Advance by 5 seconds → should NOT update (interval is 10s)
        model.clock.iteration = 1
        model.clock.time = FT(5)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] == 0  # Radiation was skipped

        # Advance to 10 seconds → SHOULD update
        model.clock.iteration = 2
        model.clock.time = FT(10)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100  # Radiation fired

        # Verify flux divergence was also recomputed
        @test any(Array(interior(flux_div)) .!= 0)

        # Zero out again
        interior(ℐ_lw_up) .= 0
        interior(flux_div) .= 0

        # Advance to 15 seconds → should NOT update (next at 20s)
        model.clock.iteration = 3
        model.clock.time = FT(15)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] == 0  # Skipped

        # Advance to 20 seconds → SHOULD update
        model.clock.iteration = 4
        model.clock.time = FT(20)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100  # Radiation fired
    end

    @testset "Fixed cosine zenith angle [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100kilometers), y=(0, 100kilometers),
                               z=(0, 10kilometers), topology=(Periodic, Periodic, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # RCEMIP-style perpetual insolation with fixed cos(zenith)
        cos_θ = FT(cosd(42.04))
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           coordinate = cos_θ)

        clock = Clock(time=FT(0))
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)

        θ(x, y, z) = 300 + FT(0.01) * z / 1000
        qᵗ(x, y, z) = FT(0.015) * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_sw_dn = radiation.downwelling_shortwave_flux

        # Shortwave should be present (perpetual sun)
        @test all(Array(interior(ℐ_sw_dn)) .<= 0)  # Downwelling is negative

        # TOA shortwave should be approximately -solar_constant * cos_θ
        @allowscalar begin
            ℐ_sw_toa = ℐ_sw_dn[1, 1, Nz + 1]
            @test ℐ_sw_toa < 0
            @test abs(ℐ_sw_toa) ≤ 1361
        end
    end
end
