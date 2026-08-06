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

@testset "Clear-sky full-spectrum RadiativeTransferModel" begin

    @testset "Constructor argument errors" begin
        FT = Float64
        Oceananigans.defaults.FloatType = FT

        Nz = 4
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))
        constants = ThermodynamicConstants()

        # Error: providing surface_albedo along with direct_surface_albedo
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          direct_surface_albedo = 0.1)

        # Error: providing surface_albedo along with diffuse_surface_albedo
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          diffuse_surface_albedo = 0.1)

        # Error: providing surface_albedo along with both direct and diffuse
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          direct_surface_albedo = 0.1,
                                                          diffuse_surface_albedo = 0.1)

        # Error: providing only direct_surface_albedo without diffuse
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          direct_surface_albedo = 0.1)

        # Error: providing only diffuse_surface_albedo without direct
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300,
                                                          diffuse_surface_albedo = 0.1)

        # Error: providing no albedo at all
        @test_throws ArgumentError RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                                          surface_temperature = 300)
    end
    @testset "Single column grid [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT

        Nz = 8
        grid = RectilinearGrid(default_arch; size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361)

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 12, 0, 0))
        model = AtmosphereModel(grid; clock, dynamics, formulation=:LiquidIcePotentialTemperature, radiation)

        θ(z) = 300 + 0.01 * z / 1000
        qᵗ(z) = 0.015 * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_lw_up = radiation.upwelling_longwave_flux
        ℐ_lw_dn = radiation.downwelling_longwave_flux
        ℐ_sw_up = radiation.upwelling_shortwave_flux
        ℐ_sw_dn = radiation.downwelling_shortwave_flux

        # Basic sanity: sign convention and finite values
        @test all(isfinite, interior(ℐ_lw_up))
        @test all(isfinite, interior(ℐ_lw_dn))
        @test all(isfinite, interior(ℐ_sw_up))
        @test all(isfinite, interior(ℐ_sw_dn))

        # Allow small numerical tolerance (wider for Float32)
        ε = FT == Float32 ? FT(1e-2) : FT(1e-6)
        @test all(interior(ℐ_lw_up) .≥ -ε)
        @test all(interior(ℐ_lw_dn) .≤ ε)
        @test all(interior(ℐ_sw_up) .≥ -ε)
        @test all(interior(ℐ_sw_dn) .≤ ε)

        @allowscalar begin
            # Surface upwelling LW should be significant.
            @test ℐ_lw_up[1, 1, 1] > 100

            # The nonzero-albedo surface should reflect shortwave upward.
            @test ℐ_sw_up[1, 1, 1] > 0
        end

        # The Oceananigans output field must retain RRTMGP's upward shortwave flux.
        # (`longwave_solver` holds the combined RRTMGP solver for full-spectrum optics.)
        rrtmgp_sw_up = Array(RRTMGP.sw_flux_up(radiation.longwave_solver))
        @test vec(Array(interior(ℐ_sw_up))) == vec(rrtmgp_sw_up)

        # Integrating -∂ℐ_net/∂z over the column must recover the complete
        # boundary-flux difference, including reflected and scattered shortwave.
        # This fails if `flux_divergence` drops any of the four streams.
        ℐ_net = vec(Array(interior(ℐ_lw_up))) .+
                vec(Array(interior(ℐ_lw_dn))) .+
                vec(Array(interior(ℐ_sw_up))) .+
                vec(Array(interior(ℐ_sw_dn)))

        # The signed Breeze sum must equal RRTMGP's longwave-plus-shortwave net flux.
        rrtmgp_net = vec(Array(RRTMGP.lw_flux_net(radiation.longwave_solver))) .+
                     vec(Array(RRTMGP.sw_flux_net(radiation.longwave_solver)))
        @test ℐ_net ≈ rrtmgp_net rtol = sqrt(eps(FT))

        Δz = 10kilometers / Nz  # the grid is uniform in z
        column_heating = Δz * sum(Array(interior(radiation.flux_divergence)))
        @test column_heating ≈ ℐ_net[1] - ℐ_net[end] rtol = sqrt(eps(FT))
    end
end
