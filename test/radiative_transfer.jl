using Breeze
using Dates
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Load RRTMGP to trigger the extension
using ClimaComms
using RRTMGP

#####
##### Unit tests
#####

@testset "GrayRadiation construction" begin
    @testset "Single column grid [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        radiation = GrayRadiation(grid;
                                  surface_temperature = 300,
                                  surface_emissivity = 0.98,
                                  surface_albedo = 0.1,
                                  solar_constant = 1361)

        @test radiation !== nothing
        @test radiation.surface_temperature == FT(300)
        @test radiation.surface_emissivity == FT(0.98)
        @test radiation.surface_albedo == FT(0.1)
        @test radiation.solar_constant == FT(1361)

        # Check flux fields are created
        @test radiation.upwelling_longwave_flux !== nothing
        @test radiation.downwelling_longwave_flux !== nothing
        @test radiation.downwelling_shortwave_flux !== nothing

        # Check flux fields have correct size (Nz+1 levels)
        @test size(radiation.upwelling_longwave_flux) == (1, 1, Nz + 1)
        @test size(radiation.downwelling_longwave_flux) == (1, 1, Nz + 1)
        @test size(radiation.downwelling_shortwave_flux) == (1, 1, Nz + 1)
    end
end

@testset "GrayRadiation with AtmosphereModel" begin
    @testset "Model construction [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants,
                                         base_pressure = 101325,
                                         potential_temperature = 300)
        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = GrayRadiation(grid;
                                  surface_temperature = 300,
                                  surface_emissivity = 0.98,
                                  surface_albedo = 0.1,
                                  solar_constant = 1361)

        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))
        model = AtmosphereModel(grid; clock, formulation, radiation)

        @test model.radiative_transfer !== nothing
        @test model.radiative_transfer === radiation
    end

    @testset "Radiation update during set! [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants,
                                         base_pressure = 101325,
                                         potential_temperature = 300)
        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = GrayRadiation(grid;
                                  surface_temperature = 300,
                                  surface_emissivity = 0.98,
                                  surface_albedo = 0.1,
                                  solar_constant = 1361)

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, formulation, radiation)

        # Set initial condition - this should trigger radiation update
        set!(model; θ = 300)

        # Check that longwave fluxes are computed (should be non-zero)
        @allowscalar begin
            # Surface upwelling LW should be approximately σT⁴/π ≈ 145 W/m² (for Planck source)
            # or σT⁴ ≈ 459 W/m² for total flux
            F_lw_up_sfc = radiation.upwelling_longwave_flux[1, 1, 1]
            @test F_lw_up_sfc > 100  # Should be significant
            @test F_lw_up_sfc < 600  # But not unreasonably large

            # TOA downwelling LW should be small (space is cold)
            F_lw_dn_toa = radiation.downwelling_longwave_flux[1, 1, Nz + 1]
            @test F_lw_dn_toa < 10

            # Shortwave direct beam at TOA should be solar_constant * cos(zenith)
            F_sw_toa = radiation.downwelling_shortwave_flux[1, 1, Nz + 1]
            @test F_sw_toa > 0
            @test F_sw_toa <= 1361  # Cannot exceed solar constant
        end
    end
end

#####
##### Integration tests
#####

@testset "Single column radiation integration" begin
    @testset "Full radiation profile [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT

        # Setup similar to single_column_radiation.jl example but lower resolution
        Nz = 32
        λ, φ = -70.9, 42.5  # Beverly, MA

        grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        surface_temperature = FT(300)

        reference_state = ReferenceState(grid, constants,
                                         base_pressure = 101325,
                                         potential_temperature = surface_temperature)

        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = GrayRadiation(grid;
                                  surface_temperature,
                                  surface_emissivity = FT(0.98),
                                  surface_albedo = FT(0.1),
                                  solar_constant = FT(1361))

        clock = Clock(time=DateTime(2024, 9, 27, 16, 0, 0))
        microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; clock, formulation, microphysics, radiation)

        # Initial condition: simple temperature profile
        θ₀ = formulation.reference_state.potential_temperature
        cᵖᵈ = constants.dry_air.heat_capacity
        g = constants.gravitational_acceleration
        Γ = g / cᵖᵈ
        θ_profile(z) = θ₀ + Γ * z / 1000

        q₀ = FT(0.015)
        Hᵗ = FT(2500)
        qᵗ_profile(z) = q₀ * exp(-z / Hᵗ)

        set!(model; θ=θ_profile, qᵗ=qᵗ_profile)

        # Test radiation profile properties
        @allowscalar begin
            # Longwave upwelling should decrease with altitude (less emission from colder air)
            F_lw_up_sfc = radiation.upwelling_longwave_flux[1, 1, 1]
            F_lw_up_toa = radiation.upwelling_longwave_flux[1, 1, Nz + 1]
            @test F_lw_up_sfc > F_lw_up_toa

            # Longwave downwelling should decrease with altitude (less back-radiation above)
            F_lw_dn_sfc = radiation.downwelling_longwave_flux[1, 1, 1]
            F_lw_dn_toa = radiation.downwelling_longwave_flux[1, 1, Nz + 1]
            @test F_lw_dn_sfc > F_lw_dn_toa

            # Shortwave direct beam should decrease with depth (absorption)
            F_sw_toa = radiation.downwelling_shortwave_flux[1, 1, Nz + 1]
            F_sw_sfc = radiation.downwelling_shortwave_flux[1, 1, 1]
            @test F_sw_toa > F_sw_sfc
            @test F_sw_sfc > 0  # Some radiation reaches surface
        end
    end
end
