# Minimal working example: AtmosphereModel with 1M mixed-phase microphysics
# This MWE builds a model to test the density parameter interface

using Breeze
using Oceananigans
using CloudMicrophysics
# using CloudMicrophysics.Parameters: Rain, Snow, CloudIce, CloudLiquid, CollisionEff

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: 
    ZeroMomentCloudMicrophysics,
    OneMomentCloudMicrophysics

grid = RectilinearGrid(size=(8, 8, 8), extent=(1000, 1000, 1000))
model = AtmosphereModel(grid; microphysics=nothing)
model = AtmosphereModel(grid; microphysics=SaturationAdjustment())
model = AtmosphereModel(grid; microphysics=ZeroMomentCloudMicrophysics())
# model = AtmosphereModel(grid; microphysics=OneMomentCloudMicrophysics())
