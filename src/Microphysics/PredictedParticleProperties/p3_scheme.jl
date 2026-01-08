#####
##### Predicted Particle Properties (P3) Microphysics Scheme
#####
##### Main type combining ice, rain, and cloud properties.
#####

"""
    PredictedParticlePropertiesMicrophysics{FT, ICE, RAIN, CLOUD, BC}

The Predicted Particle Properties (P3) microphysics scheme.

P3 uses a single ice category with predicted properties (rime fraction,
rime density, liquid fraction) rather than multiple discrete categories
(cloud ice, snow, graupel, hail). This allows continuous evolution of
ice particle characteristics.

# Prognostic Variables

Cloud liquid:
- `ρqᶜˡ`: Cloud liquid mass density [kg/m³]
- `ρnᶜˡ`: Cloud droplet number density [1/m³] (if prognostic)

Rain:
- `ρqʳ`: Rain mass density [kg/m³]
- `ρnʳ`: Rain number density [1/m³]

Ice (single category with predicted properties):
- `ρqⁱ`: Total ice mass density [kg/m³]
- `ρnⁱ`: Ice number density [1/m³]
- `ρqᶠ`: Frost/rime mass density [kg/m³]
- `ρbᶠ`: Frost/rime volume density [m³/m³]
- `ρzⁱ`: Ice 6th moment (reflectivity) [m⁶/m³] (3-moment)
- `ρqʷⁱ`: Water on ice mass density [kg/m³] (liquid fraction)

# Fields

## Top-level parameters
- `water_density`: Liquid water density ρʷ [kg/m³] (shared by cloud and rain)
- `minimum_mass_mixing_ratio`: Threshold below which hydrometeor is ignored [kg/kg]
- `minimum_number_mixing_ratio`: Threshold for number concentration [1/kg]

## Property containers
- `ice`: [`IceProperties`](@ref) - ice particle properties and integrals
- `rain`: [`RainProperties`](@ref) - rain properties and integrals
- `cloud`: [`CloudDropletProperties`](@ref) - cloud droplet properties
- `precipitation_boundary_condition`: Boundary condition for precipitation at surface

# References

- Morrison and Milbrandt (2015), J. Atmos. Sci. - Original P3 scheme
- Milbrandt and Morrison (2016), J. Atmos. Sci. - 3-moment ice
- Milbrandt et al. (2024), J. Adv. Model. Earth Syst. - Predicted liquid fraction
"""
struct PredictedParticlePropertiesMicrophysics{FT, ICE, RAIN, CLOUD, BC}
    # Shared physical constants
    water_density :: FT
    # Top-level thresholds
    minimum_mass_mixing_ratio :: FT
    minimum_number_mixing_ratio :: FT
    # Property containers
    ice :: ICE
    rain :: RAIN
    cloud :: CLOUD
    # Boundary condition
    precipitation_boundary_condition :: BC
end

"""
$(TYPEDSIGNATURES)

Construct a `PredictedParticlePropertiesMicrophysics` scheme with default parameters.

This creates the full P3 v5.5 scheme with:
- 3-moment ice (mass, number, reflectivity)
- Predicted liquid fraction on ice
- Predicted rime fraction and density

# Keyword Arguments
- `water_density`: Liquid water density [kg/m³], default 1000
- `precipitation_boundary_condition`: Boundary condition at surface for precipitation.
  Default is `nothing` which uses open boundary (precipitation exits domain).

# Example

```julia
using Breeze

microphysics = PredictedParticlePropertiesMicrophysics()
```
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
    print(io, "└── cloud: ", summary(p3.cloud))
end

#####
##### Prognostic field names
#####

"""
    prognostic_field_names(::PredictedParticlePropertiesMicrophysics)

Return prognostic field names for the P3 scheme.

P3 v5.5 with 3-moment ice and predicted liquid fraction has 9 prognostic fields:
- Cloud: ρqᶜˡ (number is prescribed, not prognostic)
- Rain: ρqʳ, ρnʳ
- Ice: ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ
"""
function prognostic_field_names(::PredictedParticlePropertiesMicrophysics)
    # Cloud number is prescribed (not prognostic) in this implementation
    cloud_names = (:ρqᶜˡ,)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρzⁱ, :ρqʷⁱ)
    
    return tuple(cloud_names..., rain_names..., ice_names...)
end

