module ParcelDynamics

export
    # Types
    ParcelDynamics,
    ParcelModel,
    ParcelState,
    EnvironmentalProfile,

    # Functions
    adiabatic_adjustment,
    environmental_velocity,
    environmental_pressure,
    environmental_density

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS

using Breeze.AtmosphereModels: NothingMicrophysicalState, microphysical_tendency
using Breeze.Thermodynamics: MoistureMassFractions

include("environmental_profile.jl")
include("parcel_dynamics.jl")

end # module ParcelDynamics
