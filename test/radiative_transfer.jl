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

@testset "GrayRadiativeTransferModel construction" begin
    @testset "Single column grid [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        radiation = GrayRadiativeTransferModel(grid, constants;
                                               surface_temperature = 300,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

        @test radiation !== nothing
        @test radiation.surface_temperature.constant == FT(300)
        @test radiation.surface_emissivity == FT(0.98)
        @test radiation.surface_albedo.constant == FT(0.1)
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

@testset "GrayRadiativeTransferModel with AtmosphereModel" begin
    @testset "Model construction [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = GrayRadiativeTransferModel(grid, constants;
                                               surface_temperature = 300,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))
        model = AtmosphereModel(grid; clock, formulation, radiation)

        @test model.radiative_transfer !== nothing
        @test model.radiative_transfer === radiation
    end

    @testset "Radiatiative transfer basic tests [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = GrayRadiativeTransferModel(grid, constants;
                                               surface_temperature = 300,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, formulation, radiation)

        # Set initial condition - this should trigger radiation update
        θ(z) = 300 + 0.01 * z / 1000
        qᵗ(z) = 0.015 * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        # Check that longwave fluxes are computed (should be non-zero)
        # Sign convention: positive = upward, negative = downward
        @allowscalar begin
            # Surface upwelling LW should be approximately σT⁴ ≈ 459 W/m² (positive)
            ℐ_lw_up_sfc = radiation.upwelling_longwave_flux[1, 1, 1]
            @test ℐ_lw_up_sfc ≈ 459 # W/m²

            # TOA downwelling LW should be small (space is cold), negative sign
            ℐ_lw_dn_toa = radiation.downwelling_longwave_flux[1, 1, Nz + 1]
            @test abs(ℐ_lw_dn_toa) < 10

            # Shortwave direct beam at TOA should be solar_constant * cos(zenith)
            # Sign convention: downwelling is negative
            ℐ_sw_toa = radiation.downwelling_shortwave_flux[1, 1, Nz + 1]
            @test ℐ_sw_toa < 0  # Downwelling is negative
            @test abs(ℐ_sw_toa) <= 1361  # Magnitude cannot exceed solar constant

        ℐ_lw_up_sfc = @allowscalar ℐ_lw_up[1, 1, 1]
        end

        @test all(interior(ℐ_lw_up) .>= 0)  # Upwelling should be positive
        @test all(interior(ℐ_lw_dn) .<= 0)  # Downwelling should be negative
        @test all(interior(ℐ_sw_dn) .<= 0)  # Downwelling should be negative
    end
end
