using Breeze
using CloudMicrophysics
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt:
    ZeroMomentCloudMicrophysics,
    OneMomentCloudMicrophysics

@testset "CloudMicrophysics Extension" begin
    @test ZeroMomentCloudMicrophysics() isa BulkMicrophysics
    @test OneMomentCloudMicrophysics() isa BulkMicrophysics
end

