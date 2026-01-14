module ParcelDynamics

export
    # Types
    ParcelModel,
    ParcelState,
    EnvironmentalProfile,

    # Functions
    step_parcel!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS

using Breeze.AtmosphereModels: AbstractMicrophysicalState, NothingMicrophysicalState,
    microphysical_tendency
using Breeze.Thermodynamics: temperature, saturation_specific_humidity, MoistureMassFractions

# include("environmental_profile.jl")
include("parcel_dynamics.jl")

end # module ParcelDynamics
