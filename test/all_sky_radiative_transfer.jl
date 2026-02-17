using Breeze
using Dates
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Trigger RRTMGP + netCDF lookup table loading
using ClimaComms
using NCDatasets
using RRTMGP

@testset "All-sky full-spectrum RadiativeTransferModel [$FT]" for FT in test_float_types()

    @testset "Constructor argument errors" begin
        Oceananigans.defaults.FloatType = FT
        topology = (Flat, Flat, Bounded)
        grid = RectilinearGrid(default_arch; size=4, x=0, y=45, z=(0, 10kilometers), topology)
        constants = ThermodynamicConstants()

        # Error: providing surface_albedo along with direct_surface_albedo
        @test_throws ArgumentError RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          direct_surface_albedo = 0.1)

        # Error: providing surface_albedo along with diffuse_surface_albedo
        @test_throws ArgumentError RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          diffuse_surface_albedo = 0.1)

        # Error: providing no albedo at all
        @test_throws ArgumentError RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                                          surface_temperature = 300)
    end

    @testset "Single column grid with clouds [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        topology = (Flat, Flat, Bounded)
        grid = RectilinearGrid(default_arch; size=16, x=0, y=45, z=(0, 10kilometers), topology)
        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Create clear-sky and all-sky radiation models
        clear_sky_radiation = RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                     surface_temperature = 300,
                                                     surface_emissivity = 0.98,
                                                     surface_albedo = 0.1,
                                                     solar_constant = 1361)

        all_sky_radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                                   surface_temperature = 300,
                                                   surface_emissivity = 0.98,
                                                   surface_albedo = 0.1,
                                                   solar_constant = 1361,
                                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                                   ice_effective_radius = ConstantRadiusParticles(30e-6))

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))

        # Use saturation adjustment microphysics to generate cloud condensate
        microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

        clear_model = AtmosphereModel(grid; clock, dynamics, microphysics,
                                      formulation=:LiquidIcePotentialTemperature,
                                      radiation=clear_sky_radiation)

        all_sky_model = AtmosphereModel(grid; clock, dynamics, microphysics,
                                        formulation=:LiquidIcePotentialTemperature,
                                        radiation=all_sky_radiation)

        # Set up an initial condition that produces clouds in some layers
        # High moisture in the mid-troposphere will saturate and produce cloud condensate
        θ(z) = 300 + 0.005 * z / 1000  # Weak lapse rate
        qᵗ(z) = 0.020 * exp(-z / 3000)  # High moisture to ensure saturation

        set!(clear_model; θ=θ, qᵗ=qᵗ)
        set!(all_sky_model; θ=θ, qᵗ=qᵗ)

        # Get fluxes
        ℐ_lw_up_clear = clear_sky_radiation.upwelling_longwave_flux
        ℐ_lw_dn_clear = clear_sky_radiation.downwelling_longwave_flux
        ℐ_sw_dn_clear = clear_sky_radiation.downwelling_shortwave_flux

        ℐ_lw_up_allsky = all_sky_radiation.upwelling_longwave_flux
        ℐ_lw_dn_allsky = all_sky_radiation.downwelling_longwave_flux
        ℐ_sw_dn_allsky = all_sky_radiation.downwelling_shortwave_flux

        # Basic sanity: sign convention and finite values for all-sky
        @test all(isfinite, interior(ℐ_lw_up_allsky))
        @test all(isfinite, interior(ℐ_lw_dn_allsky))
        @test all(isfinite, interior(ℐ_sw_dn_allsky))

        # Allow small numerical tolerance (wider for Float32)
        ε = FT == Float32 ? FT(1e-2) : FT(1e-6)
        @test all(interior(ℐ_lw_up_allsky) .>= -ε)
        @test all(interior(ℐ_lw_dn_allsky) .<= ε)
        @test all(interior(ℐ_sw_dn_allsky) .<= ε)

        # Surface upwelling LW should be significant
        @allowscalar @test ℐ_lw_up_allsky[1, 1, 1] > 100

        # Check that cloud condensate was actually produced
        qˡ = all_sky_model.microphysical_fields.qˡ
        total_cloud_liquid = sum(interior(qˡ))
        @test total_cloud_liquid > 0  # Should have some cloud liquid

        # All-sky should differ from clear-sky when clouds are present
        # The difference should be noticeable in at least one flux component
        lw_up_diff = sum(abs, interior(ℐ_lw_up_allsky) .- interior(ℐ_lw_up_clear))
        sw_dn_diff = sum(abs, interior(ℐ_sw_dn_allsky) .- interior(ℐ_sw_dn_clear))

        # At least one of LW or SW should show a difference due to clouds
        # (The magnitude depends on cloud amount, but should be non-zero)
        @test (lw_up_diff > 0) || (sw_dn_diff > 0)
    end

    @testset "Custom effective radius models [$FT]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        topology = (Flat, Flat, Bounded)
        grid = RectilinearGrid(default_arch; size=8, x=0, y=45, z=(0, 10kilometers), topology)
        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        # Test with custom effective radii
        small_droplets = ConstantRadiusParticles(5e-6)
        large_ice = ConstantRadiusParticles(50e-6)

        radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                           surface_temperature = 300,
                                           surface_albedo = 0.1,
                                           liquid_effective_radius = small_droplets,
                                           ice_effective_radius = large_ice)

        # Verify the effective radius models are stored correctly
        @test radiation.liquid_effective_radius.radius == FT(5e-6)
        @test radiation.ice_effective_radius.radius == FT(50e-6)

        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))
        microphysics = SaturationAdjustment()
        model = AtmosphereModel(grid; clock, dynamics, microphysics, radiation)

        θ(z) = 300
        qᵗ(z) = 0.015 * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        # Should run without error
        @test all(isfinite, interior(radiation.upwelling_longwave_flux))
    end
end
