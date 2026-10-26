include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Dates: DateTime, Second
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Load RRTMGP and NumericalRadiation to trigger the extensions
using ClimaComms
using RRTMGP
using NCDatasets
using NumericalRadiation: NumericalRadiation

# The gray RRTMGP optics and the ecCKD optics of the NumericalRadiation extension share the
# scheduling machinery of `RadiativeTransferModel`
const OPTICS = (GrayOptics(), EcCKDOptics())

@testset "Radiation scheduling" begin
    @testset "IterationInterval schedule [$(FT), $(summary(optics))]" for FT in test_float_types(), optics in OPTICS
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch; size=Nz, x=FT(0), y=FT(45), z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         base_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Update radiation every 3 iterations
        radiation = RadiativeTransferModel(grid, optics, constants;
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

    @testset "TimeInterval schedule [$(FT), $(summary(optics))]" for FT in test_float_types(), optics in OPTICS
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch; size=Nz, x=FT(0), y=FT(45), z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         base_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Update radiation every 10 seconds (use numeric clock for TimeInterval compatibility)
        radiation = RadiativeTransferModel(grid, optics, constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_position = ApparentSolarPosition(coordinate = (FT(0), FT(45)),
                                                                                  epoch      = DateTime(2024, 6, 21, 16, 0, 0)),
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

    @testset "Checkpoint state [$(FT), $(summary(optics))]" for FT in test_float_types(), optics in OPTICS
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(default_arch; size=8, x=FT(0), y=FT(45), z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        function radiating_model()
            constants = ThermodynamicConstants()
            reference_state = ReferenceState(grid, constants;
                                             base_pressure = 101325,
                                             potential_temperature = 300)
            radiation = RadiativeTransferModel(grid, optics, constants;
                                               surface_temperature = 300,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_position = ApparentSolarPosition(coordinate = (FT(0), FT(45)),
                                                                                      epoch      = DateTime(2024, 6, 21, 16, 0, 0)),
                                               schedule = TimeInterval(10))
            model = AtmosphereModel(grid; clock = Clock(time=FT(0)), dynamics = AnelasticDynamics(reference_state),
                                    formulation = :LiquidIcePotentialTemperature, radiation)
            return model, radiation
        end

        model, radiation = radiating_model()
        θ(z) = 300 + FT(0.01) * z / 1000
        qᵗ(z) = FT(0.015) * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)
        Oceananigans.initialize!(radiation.schedule, model)
        model.clock.iteration = 1
        model.clock.time = FT(10)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)

        # Only the schedule and the fluxes the model reads between solves are saved, not the solvers
        state = Oceananigans.prognostic_state(radiation)
        flux_names = (:upwelling_longwave_flux, :downwelling_longwave_flux,
                      :upwelling_shortwave_flux, :downwelling_shortwave_flux, :flux_divergence)
        @test keys(state) == (:schedule, flux_names...)

        _, restored = radiating_model()
        Oceananigans.restore_prognostic_state!(restored, state)
        @test restored.schedule.actuations == radiation.schedule.actuations == 1
        @allowscalar @test restored.upwelling_longwave_flux[1, 1, 1] > 100
        for name in flux_names
            @test Array(interior(getproperty(restored, name))) == Array(interior(getproperty(radiation, name)))
        end
    end

    @testset "Multi-column grid [$(FT), $(summary(optics))]" for FT in test_float_types(), optics in OPTICS
        Oceananigans.defaults.FloatType = FT
        Nx, Ny, Nz = 4, 4, 8
        grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz),
                               x=(0, 100kilometers), y=(0, 100kilometers), z=(0, 10kilometers),
                               topology=(Periodic, Periodic, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         base_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, optics, constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_position = ApparentSolarPosition(coordinate = (FT(0), FT(45))))

        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)

        θ(x, y, z) = 300 + FT(0.01) * z / 1000
        qᵗ(x, y, z) = FT(0.015) * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_lw_up = radiation.upwelling_longwave_flux
        ℐ_sw_dn = radiation.downwelling_shortwave_flux

        # Longwave should be computed for all columns
        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100

        # Shortwave should be non-zero (sun above horizon at 45°N summer solstice)
        @allowscalar @test ℐ_sw_dn[1, 1, Nz + 1] < 0  # Downwelling is negative

        # All columns should have the same fluxes (uniform ICs, same coordinate)
        lw_array = Array(interior(ℐ_lw_up))
        @test all(lw_array[:, :, 1] .== lw_array[1, 1, 1])
    end
end
