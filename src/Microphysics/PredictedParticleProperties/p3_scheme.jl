#####
##### Predicted Particle Properties (P3) Microphysics Scheme
#####
##### Main type combining ice, rain, and cloud properties.
#####

"""
    PredictedParticlePropertiesMicrophysics

The Predicted Particle Properties (P3) microphysics scheme. See the constructor
[`PredictedParticlePropertiesMicrophysics()`](@ref) for usage and documentation.
"""
struct PredictedParticlePropertiesMicrophysics{FT, ICE, RAIN, CLOUD, PRP, BC}
    # Shared physical constants
    water_density :: FT
    # Top-level thresholds
    minimum_mass_mixing_ratio :: FT
    minimum_number_mixing_ratio :: FT
    # Property containers
    ice :: ICE
    rain :: RAIN
    cloud :: CLOUD
    # Process rate parameters
    process_rates :: PRP
    # Boundary condition
    precipitation_boundary_condition :: BC
end

"""
$(TYPEDSIGNATURES)

Construct the Predicted Particle Properties (P3) microphysics scheme.

P3 is a bulk microphysics scheme that uses a **single ice category** with
continuously predicted properties, rather than discrete categories like
cloud ice, snow, graupel, and hail. As ice particles grow and rime, their
properties evolve smoothly without artificial category conversions.

# Physical Concept

Traditional schemes force growing ice particles through discrete transitions:

    cloud ice → snow → graupel → hail

Each transition requires ad-hoc conversion parameters. P3 instead tracks:

- **Rime fraction** ``Fᶠ``: What fraction of mass is rime?
- **Rime density** ``ρᶠ``: How dense is the rime layer?
- **Liquid fraction** ``Fˡ``: Liquid water coating from partial melting

From these, particle characteristics (mass, fall speed, collection efficiency)
are diagnosed continuously.

# Three-Moment Ice

P3 v5.5 carries three prognostic moments for ice particles:
1. **Mass** (``qⁱ``): Total ice mass
2. **Number** (``nⁱ``): Ice particle number concentration
3. **Reflectivity** (``zⁱ``): Sixth moment of size distribution

The third moment improves representation of precipitation-sized particles
and enables better simulation of radar reflectivity.

# Prognostic Variables

The scheme tracks 9 prognostic densities:

| Variable | Description |
|----------|-------------|
| ``ρqᶜˡ`` | Cloud liquid mass |
| ``ρqʳ``, ``ρnʳ`` | Rain mass and number |
| ``ρqⁱ``, ``ρnⁱ`` | Ice mass and number |
| ``ρqᶠ``, ``ρbᶠ`` | Rime mass and volume |
| ``ρzⁱ`` | Ice 6th moment (reflectivity) |
| ``ρqʷⁱ`` | Liquid water on ice |

# Keyword Arguments

- `water_density`: Liquid water density [kg/m³] (default 1000)
- `precipitation_boundary_condition`: Boundary condition for surface precipitation
  (default `nothing` = open boundary, precipitation exits domain)

# Example

```julia
using Breeze

# Create P3 scheme with default parameters
microphysics = PredictedParticlePropertiesMicrophysics()

# Get prognostic field names for model setup
fields = prognostic_field_names(microphysics)
```

# References

This implementation follows P3 v5.5 from the
[P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

Key papers describing P3:
- [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization): Original scheme
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Three-moment ice
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Predicted liquid fraction
- [Morrison et al. (2025)](@cite Morrison2025complete3moment): Complete implementation

See also the [P3 documentation](@ref p3_overview) for detailed physics.
"""
function PredictedParticlePropertiesMicrophysics(FT::Type{<:AbstractFloat} = Float64;
                                                  water_density = 1000,
                                                  precipitation_boundary_condition = nothing)
    return PredictedParticlePropertiesMicrophysics(
        FT(water_density),
        FT(1e-14),   # minimum_mass_mixing_ratio [kg/kg]
        FT(1e-16),   # minimum_number_mixing_ratio [1/kg]
        IceProperties(FT),
        RainProperties(FT),
        CloudDropletProperties(FT),
        ProcessRateParameters(FT),
        precipitation_boundary_condition
    )
end

# Shorthand alias
const P3Microphysics = PredictedParticlePropertiesMicrophysics

Base.summary(::PredictedParticlePropertiesMicrophysics) = "PredictedParticlePropertiesMicrophysics"

function Base.show(io::IO, p3::PredictedParticlePropertiesMicrophysics)
    print(io, summary(p3), '\n')
    print(io, "├── ρʷ: ", p3.water_density, " kg/m³\n")
    print(io, "├── qmin: ", p3.minimum_mass_mixing_ratio, " kg/kg\n")
    print(io, "├── ice: ", summary(p3.ice), "\n")
    print(io, "├── rain: ", summary(p3.rain), "\n")
    print(io, "├── cloud: ", summary(p3.cloud), "\n")
    print(io, "└── process_rates: ", summary(p3.process_rates))
end

# Note: prognostic_field_names is implemented in p3_interface.jl to extend
# AtmosphereModels.prognostic_field_names
