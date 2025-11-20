using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

# Only run radiative transfer tests if RRTMGP extension is available
# Check if the extension has been loaded by checking for RadiativeTransferModel
if isdefined(Breeze, :RadiativeTransferModel)
    using Breeze: RadiativeTransferModel

    @testset "RadiativeTransferModel construction [$(FT)]" for FT in (Float32, Float64)
        grid = RectilinearGrid(default_arch, FT; size=(4, 4, 10), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
        
        # Test basic construction
        rtm = RadiativeTransferModel(
            grid;
            surface_emissivity = 0.98,
            surface_albedo_direct = 0.1,
            surface_albedo_diffuse = 0.1,
            cos_zenith = 0.5,
            toa_solar_flux = 1360.0,
            toa_longwave_flux = 0.0
        )
        
        @test rtm !== nothing
        @test rtm.grid_params.nlay == 10
        @test rtm.grid_params.ncol == 16  # 4 * 4
        
        # Test that surface properties are stored
        @test length(rtm.surface_properties.surface_temperature) == 16
        @test size(rtm.surface_properties.surface_emissivity) == (1, 16)
        
        # Test that atmospheric state is initialized
        @test rtm.atmospheric_state !== nothing
        @test size(rtm.atmospheric_state.p_lev) == (11, 16)  # nlev, ncol
        @test size(rtm.atmospheric_state.p_lay) == (10, 16)  # nlay, ncol
    end

    @testset "RadiativeTransferModel with AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
        grid = RectilinearGrid(default_arch, FT; size=(2, 2, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
        thermo = ThermodynamicConstants(FT)
        
        reference_state = ReferenceState(grid, thermo, base_pressure=FT(101325), potential_temperature=FT(288))
        formulation = AnelasticFormulation(reference_state)
        
        # Create radiative transfer model
        rtm = RadiativeTransferModel(
            grid;
            surface_emissivity = 0.98,
            cos_zenith = 0.5,
            toa_solar_flux = 1360.0
        )
        
        # Create atmosphere model with radiative transfer
        model = AtmosphereModel(
            grid;
            thermodynamics=thermo,
            formulation=formulation,
            radiative_transfer=rtm
        )
        
        @test model.radiative_transfer !== nothing
        @test model.radiative_transfer === rtm
        
        # Test that we can update radiative fluxes
        set!(model; θ = 288.0)
        update_state!(model)
        
        # Update radiative fluxes (should not error)
        @allowscalar begin
            Breeze.AtmosphereModels._update_radiative_fluxes!(rtm, model)
        end
        
        # Check that fluxes exist
        @test rtm.flux_lw !== nothing
        @test rtm.flux_sw !== nothing
    end
else
    @testset "RadiativeTransferModel (RRTMGP not available)" begin
        @test_skip "RRTMGP extension not loaded - skipping radiative transfer tests"
    end
end

