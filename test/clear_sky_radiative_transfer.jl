using Breeze
using Dates
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Units
using Test

# Trigger RRTMGP + netCDF lookup table loading
using ClimaComms
using NCDatasets
using RRTMGP

@testset "Clear-sky full-spectrum RadiativeTransferModel" begin
    @testset "Single column grid [$(FT)]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT

        Nz = 8
        # TODO: GPU support for clear-sky radiation needs to be fixed
        grid = RectilinearGrid(CPU(); size=Nz, x=0.0, y=45.0, z=(0, 10kilometers),
                               topology=(Flat, Flat, Bounded))

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = 101325,
                                         potential_temperature = 300)
        formulation = AnelasticFormulation(reference_state,
                                           thermodynamics = :LiquidIcePotentialTemperature)

        radiation = RadiativeTransferModel(grid, :clear_sky, constants;
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361)

        # Use noon on summer solstice at 45°N for good solar illumination
        clock = Clock(time=DateTime(2024, 6, 21, 16, 0, 0))
        model = AtmosphereModel(grid; clock, formulation, radiation)

        θ(z) = 300 + 0.01 * z / 1000
        qᵗ(z) = 0.015 * exp(-z / 2500)
        set!(model; θ=θ, qᵗ=qᵗ)

        ℐ_lw_up = radiation.upwelling_longwave_flux
        ℐ_lw_dn = radiation.downwelling_longwave_flux
        ℐ_sw_dn = radiation.downwelling_shortwave_flux

        # Basic sanity: sign convention and finite values
        @test all(isfinite, interior(ℐ_lw_up))
        @test all(isfinite, interior(ℐ_lw_dn))
        @test all(isfinite, interior(ℐ_sw_dn))

        # Allow small numerical tolerance (wider for Float32)
        ε = FT == Float32 ? FT(1e-2) : FT(1e-6)
        @test all(interior(ℐ_lw_up) .>= -ε)
        @test all(interior(ℐ_lw_dn) .<= ε)
        @test all(interior(ℐ_sw_dn) .<= ε)

        # Surface upwelling LW should be significant
        @test ℐ_lw_up[1, 1, 1] > 100
    end
end
