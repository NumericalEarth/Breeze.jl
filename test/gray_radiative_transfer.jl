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
    @testset "Single column grid [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()

        @test_throws ArgumentError RadiativeTransferModel(grid, nothing, constants)

        @testset "Number-based surface properties" begin
            radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                               surface_temperature = 300,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

            @test radiation !== nothing
            @test radiation.surface_properties.surface_temperature.constant == FT(300)
            @test radiation.surface_properties.surface_emissivity.constant == FT(0.98)
            @test radiation.surface_properties.direct_surface_albedo.constant == FT(0.1)
            @test radiation.surface_properties.diffuse_surface_albedo.constant == FT(0.1)
            @test radiation.solar_constant == FT(1361)

            # Check flux fields are created
            @test radiation.upwelling_longwave_flux !== nothing
            @test radiation.downwelling_longwave_flux !== nothing
            @test radiation.downwelling_shortwave_flux !== nothing

            # Check flux divergence field
            @test radiation.flux_divergence !== nothing
            @test size(radiation.flux_divergence) == (1, 1, Nz)

            # Check schedule
            @test radiation.schedule !== nothing

            # Check flux fields have correct size (Nz+1 levels)
            @test size(radiation.upwelling_longwave_flux) == (1, 1, Nz + 1)
            @test size(radiation.downwelling_longwave_flux) == (1, 1, Nz + 1)
            @test size(radiation.downwelling_shortwave_flux) == (1, 1, Nz + 1)

            radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                               surface_temperature = 300,
                                               direct_surface_albedo = 0.15,
                                               diffuse_surface_albedo = 0.2)

            @test radiation.surface_properties.direct_surface_albedo.constant == FT(0.15)
            @test radiation.surface_properties.diffuse_surface_albedo.constant == FT(0.2)
        end

        @testset "Field-based surface properties" begin
            T₀ = set!(CenterField(grid), 300)
            α₀ = set!(CenterField(grid), 0.1)
            ε₀ = set!(CenterField(grid), 0.98)

            radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                               surface_temperature = T₀,
                                               surface_emissivity = ε₀,
                                               surface_albedo = α₀)

            @test radiation !== nothing

            @allowscalar begin
                @test first(radiation.surface_properties.surface_temperature) == FT(300)
                @test first(radiation.surface_properties.surface_emissivity) == FT(0.98)
                @test first(radiation.surface_properties.direct_surface_albedo) == FT(0.1)
                @test first(radiation.surface_properties.diffuse_surface_albedo) == FT(0.1)
                @test radiation.solar_constant == FT(1361)
            end
        end

        @testset "Invalid surface properties" begin
            @test_throws ArgumentError RadiativeTransferModel(grid, GrayOptics(), constants;
                                                            surface_temperature = 300,
                                                            surface_albedo = 0.15,
                                                            direct_surface_albedo = 0.15,
                                                            diffuse_surface_albedo = 0.2)

            @test_throws ArgumentError RadiativeTransferModel(grid, GrayOptics(), constants;
                                                            surface_temperature = 300,
                                                            surface_albedo = 0.15,
                                                            diffuse_surface_albedo = 0.2)
        end

    end
end

@testset "GrayRadiativeTransferModel with AtmosphereModel" begin
    @testset "Model construction [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361)

        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))
        model = AtmosphereModel(grid; clock, dynamics, formulation=:LiquidIcePotentialTemperature, radiation)

        @test model.radiation !== nothing
        @test model.radiation === radiation
    end

    @testset "Radiatiative transfer basic tests [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361)

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, dynamics, formulation=:LiquidIcePotentialTemperature, radiation)

        # Set initial condition - this should trigger radiation update
        θ(z) = 300 + 0.01 * z / 1000
        qᵗ(z) = 0.015 * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        # Check that longwave fluxes are computed (should be non-zero)
        # Sign convention: positive = upward, negative = downward
        ℐ_lw_up = radiation.upwelling_longwave_flux
        ℐ_lw_dn = radiation.downwelling_longwave_flux
        ℐ_sw_dn = radiation.downwelling_shortwave_flux

        @allowscalar begin
            # Surface upwelling LW should be approximately σT⁴ ≈ 459 W/m² (positive)
            ℐ_lw_up_sfc = ℐ_lw_up[1, 1, 1]
            @test ℐ_lw_up_sfc > 100  # Should be significant
            @test ℐ_lw_up_sfc < 600  # But not unreasonably large

            # TOA downwelling LW should be small (space is cold), negative sign
            ℐ_lw_dn_toa = ℐ_lw_dn[1, 1, Nz + 1]
            @test abs(ℐ_lw_dn_toa) < 10

            # Shortwave direct beam at TOA should be solar_constant * cos(zenith)
            # Sign convention: downwelling is negative
            ℐ_sw_toa = ℐ_sw_dn[1, 1, Nz + 1]
            @test ℐ_sw_toa < 0  # Downwelling is negative
            @test abs(ℐ_sw_toa) ≤ 1361  # Magnitude cannot exceed solar constant
        end

        @test all(interior(ℐ_lw_up) .≥ 0)  # Upwelling should be positive
        @test all(interior(ℐ_lw_dn) .≤ 0)  # Downwelling should be negative
        @test all(interior(ℐ_sw_dn) .≤ 0)  # Downwelling should be negative
    end
end

@testset "GrayRadiativeTransferModel solar_position" begin
    @testset "Type construction" begin
        @test ApparentSolarPosition() isa AbstractSolarPosition
        @test ApparentSolarPosition() isa ApparentSolarPosition
        @test FixedCosineZenith(0.5) isa AbstractSolarPosition
        @test FixedCosineZenith(0.5) isa FixedCosineZenith

        sp = ApparentSolarPosition(coordinate = (-70.9, 42.5), epoch = DateTime(2024, 1, 1))
        @test sp.coordinate == (-70.9, 42.5)
        @test sp.epoch == DateTime(2024, 1, 1)

        @test FixedCosineZenith(0.5).cos_zenith == 0.5
    end

    @testset "FixedCosineZenith: numeric clock, no epoch [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # The motivating case: a numeric clock with a fixed cos(θ_z).
        # Used to crash because `update_solar_zenith_angle!` had no path for
        # numeric clock + epoch=nothing. Now this works via FixedCosineZenith.
        cos_θz = convert(FT, 0.5)
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361,
                                           solar_position = FixedCosineZenith(cos_θz))

        @test radiation.solar_position isa FixedCosineZenith
        @test radiation.solar_position.cos_zenith == cos_θz

        # cos_zenith BC array is filled at construction.
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] == cos_θz

        # Numeric clock — no epoch required.
        clock = Clock(time = zero(FT))
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)
        set!(model; θ = 300, qᵗ = 0)

        # cos_zenith stays at the fixed value across the radiation update.
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] == cos_θz

        # TOA downwelling SW magnitude is solar_constant * cos_zenith.
        # Sign convention: downwelling is negative.
        @allowscalar begin
            ℐ_sw_toa = radiation.downwelling_shortwave_flux[1, 1, Nz + 1]
            @test ℐ_sw_toa < 0
            @test abs(ℐ_sw_toa) ≈ 1361 * cos_θz rtol = 1e-3
        end
    end

    @testset "ApparentSolarPosition with epoch + numeric clock" begin
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # A numeric clock works with apparent sun when an epoch is supplied.
        epoch = DateTime(2024, 6, 21, 12, 0, 0)  # solstice noon
        solar_position = ApparentSolarPosition(epoch = epoch)
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_position = solar_position)

        clock = Clock(time = 0.0)
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)
        set!(model; θ = 300, qᵗ = 0)

        # cos_zenith should be populated and positive (sun above horizon at solstice noon, 45N).
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] > 0
    end

    @testset "AbstractSolarPosition show" begin
        # Default ApparentSolarPosition — exercises _show_coordinate(::Nothing) and _show_epoch(::Nothing)
        @test repr(ApparentSolarPosition()) ==
            "ApparentSolarPosition(coordinate=<from grid>, epoch=<from clock>)"

        # Explicit coordinate — exercises _show_coordinate(::Tuple) and prettysummary on its elements
        @test repr(ApparentSolarPosition(coordinate = (-70.9, 42.5))) ==
            "ApparentSolarPosition(coordinate=(-70.9, 42.5), epoch=<from clock>)"

        # Explicit epoch — exercises _show_epoch(epoch::DateTime)
        @test repr(ApparentSolarPosition(epoch = DateTime(2024, 1, 1))) ==
            "ApparentSolarPosition(coordinate=<from grid>, epoch=2024-01-01T00:00:00)"

        @test repr(FixedCosineZenith(0.5)) == "FixedCosineZenith(cos_zenith = 0.5)"

        @test repr(DiurnalSolarPosition(latitude = 30)) ==
            "DiurnalSolarPosition(latitude = 30.0°, declination = 0.0°, day_length = 86400.0 s, noon_offset = 0.0 s)"
    end

    @testset "DiurnalSolarPosition: construction and defaults [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        # Default FT comes from Oceananigans.defaults.FloatType
        sp = DiurnalSolarPosition(latitude = 30)
        @test sp isa AbstractSolarPosition
        @test sp isa DiurnalSolarPosition
        @test sp.latitude    == FT(30)
        @test sp.declination == FT(0)
        @test sp.day_length  == FT(86400)
        @test sp.noon_offset == FT(0)
        @test typeof(sp.latitude)    === FT
        @test typeof(sp.day_length)  === FT

        # Positional FT override forces the precision regardless of input types
        sp_f32 = DiurnalSolarPosition(Float32, latitude = 30, declination = 23.5)
        @test typeof(sp_f32.latitude)    === Float32
        @test typeof(sp_f32.declination) === Float32
        @test sp_f32.latitude    ≈ 30
        @test sp_f32.declination ≈ Float32(23.5)
    end

    @testset "DiurnalSolarPosition: diurnal cycle physics" begin
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Perpetual equinox at 30°N, noon at t = 0
        latitude = 30
        sp = DiurnalSolarPosition(latitude = latitude)
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_position = sp)

        clock = Clock(time = 0.0)
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)
        set!(model; θ = 300, qᵗ = 0)

        # At t = 0 (noon), cos(θ_z) = cos(latitude) for δ = 0
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] ≈ cos(deg2rad(latitude))

        # Step clock to midnight (t = day/2) and re-update — cos(θ_z) clamped to 0
        model.clock.time = 43200.0  # 12 h
        model.clock.iteration = 1
        ext = Base.get_extension(Breeze, :BreezeRRTMGPExt)
        ext.update_solar_zenith_angle!(radiation.shortwave_solver, sp, grid, model.clock)
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] == 0

        # Step to t = day (back to noon) — should match the initial value
        model.clock.time = 86400.0
        ext.update_solar_zenith_angle!(radiation.shortwave_solver, sp, grid, model.clock)
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] ≈ cos(deg2rad(latitude))
    end

    @testset "DiurnalSolarPosition: perpetual solstice declination" begin
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Perpetual June-solstice analog: 23.5° declination
        latitude = 45.0
        declination = 23.5
        sp = DiurnalSolarPosition(latitude = latitude, declination = declination)
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_position = sp)

        clock = Clock(time = 0.0)
        model = AtmosphereModel(grid; clock, dynamics,
                                formulation = :LiquidIcePotentialTemperature, radiation)
        set!(model; θ = 300, qᵗ = 0)

        # At noon with non-zero declination:
        #   cos(θ_z) = sin(φ) sin(δ) + cos(φ) cos(δ)
        expected_noon = sin(deg2rad(latitude)) * sin(deg2rad(declination)) +
                        cos(deg2rad(latitude)) * cos(deg2rad(declination))
        @allowscalar @test radiation.shortwave_solver.bcs.cos_zenith[1] ≈ expected_noon
    end

    @testset "DiurnalSolarPosition + DateTime clock → ArgumentError" begin
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_position = DiurnalSolarPosition(latitude = 30))

        # DateTime clock is incompatible with the idealized diurnal cycle.
        clock = Clock(time = DateTime(2024, 6, 21, 12, 0, 0))
        @test_throws ArgumentError begin
            model = AtmosphereModel(grid; clock, dynamics,
                                    formulation = :LiquidIcePotentialTemperature, radiation)
            set!(model; θ = 300, qᵗ = 0)
        end
    end

    @testset "ApparentSolarPosition + numeric clock without epoch → ArgumentError" begin
        Nz = 16
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # The combination that previously crashed silently: ApparentSolarPosition
        # (default epoch=nothing) combined with a floating-point clock. We now
        # raise an actionable ArgumentError on the first radiation update.
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           solar_position = ApparentSolarPosition())

        clock = Clock(time = 0.0)  # numeric, no epoch

        # The first radiation update fires during AtmosphereModel construction
        # (iteration 0 always triggers an update), so the throw lands there
        # rather than inside `set!`. Wrap both calls so whichever raises is caught.
        @test_throws ArgumentError begin
            model = AtmosphereModel(grid; clock, dynamics,
                                    formulation = :LiquidIcePotentialTemperature, radiation)
            set!(model; θ = 300, qᵗ = 0)
        end
    end
end
