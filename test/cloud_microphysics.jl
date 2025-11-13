using Breeze
using Test

# Check if CloudMicrophysics extension is available
const CLOUDMICROPHYSICS_AVAILABLE = isdefined(Breeze, :BreezeCloudMicrophysicsExt)

if CLOUDMICROPHYSICS_AVAILABLE
    using Breeze.BreezeCloudMicrophysicsExt
    using CloudMicrophysics
end

@testset "CloudMicrophysics Extension" begin
    if !CLOUDMICROPHYSICS_AVAILABLE
        @test_broken false "CloudMicrophysics extension not available - CloudMicrophysics.jl may not be installed"
    else
        @testset "Extension loads correctly" begin
            @test isdefined(Breeze, :BreezeCloudMicrophysicsExt)
            @test BreezeCloudMicrophysicsExt isa Module
        end

        @testset "CloudMicrophysics integration" begin
            # TODO: Add tests for CloudMicrophysics scheme integration
            # This will include tests for:
            # - Creating CloudMicrophysics schemes
            # - Using CloudMicrophysics schemes with AtmosphereModel
            # - Verifying compute_thermodynamic_state extensions
            # - Verifying moisture_mass_fractions extensions
            # - Verifying prognostic_field_names, materialize_microphysical_fields,
            #   and update_microphysical_fields! extensions
        end
    end
end

