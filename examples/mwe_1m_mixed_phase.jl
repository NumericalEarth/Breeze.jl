# Minimal working example: AtmosphereModel with 1M mixed-phase microphysics
# This MWE builds a model to test the density parameter interface

using Breeze
using Oceananigans
using CloudMicrophysics
# using CloudMicrophysics.Parameters: Rain, Snow, CloudIce, CloudLiquid, CollisionEff

# Create 1M mixed-phase microphysics parameters
FT = Float64

categories = FourCategories()
nucleation = SaturationAdjustment()
microphysics = BulkMicrophysics(nucleation, categories)

@show microphysics

# # Create a simple grid
# grid = RectilinearGrid(size=(8, 8, 8), extent=(1000, 1000, 1000))

# # Create thermodynamic constants
# thermo = ThermodynamicConstants()

# # Create reference state and formulation
# reference_state = ReferenceState(grid, thermo)
# formulation = AnelasticFormulation(reference_state)

# # Create clouds scheme (mixed-phase saturation adjustment)
# clouds = Breeze.Microphysics.SaturationAdjustment(FT; equilibrium=Breeze.Microphysics.MixedPhaseEquilibrium(FT))

# # Create BulkMicrophysics with clouds and precipitation
# microphysics = Breeze.Microphysics.BulkMicrophysics(FT, clouds, precipitation)

# # Build the model (this will call update_state internally)
# model = AtmosphereModel(grid;
#                        thermodynamics=thermo,
#                        formulation=formulation,
#                        microphysics=microphysics)

# println("Model built successfully!")
# println("Microphysics type: ", typeof(model.microphysics))
# println("Microphysical fields: ", keys(model.microphysical_fields))
