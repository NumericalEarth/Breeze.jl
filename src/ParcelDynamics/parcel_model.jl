#####
##### ParcelModel: container for parcel simulation
#####

"""
$(TYPEDEF)

A Lagrangian parcel model for simulating adiabatic ascent with microphysics.

The `ParcelModel` encapsulates all components needed to evolve an air parcel
through a prescribed atmospheric sounding. It combines:

- **Environmental profile**: Prescribed temperature, pressure, humidity profiles
- **Microphysics scheme**: Determines how cloud condensate forms and evolves
- **Thermodynamic constants**: Physical parameters for equations of state

The parcel moves through the environment, conserving entropy (for adiabatic ascent)
while the microphysics scheme computes tendencies for cloud liquid, rain, etc.

# Fields
$(TYPEDFIELDS)

# Usage

```julia
using Breeze.ParcelDynamics
using CloudMicrophysics

# Create environmental profile
profile = EnvironmentalProfile(
    temperature = z -> 288.15 - 0.0065z,
    pressure = z -> 101325 * (1 - 2.25577e-5 * z)^5.25588,
    density = z -> 1.225 * (1 - 2.25577e-5 * z)^4.25588,
    specific_humidity = z -> 0.015 * exp(-z/2500),
    w = z -> 1.0  # 1 m/s updraft
)

# Create microphysics scheme
microphysics = OneMomentCloudMicrophysics()

# Build parcel model
model = ParcelModel(profile, microphysics, constants)

# Initialize parcel state at surface
state = ParcelState(x=0, y=0, z=0, ρ=1.2, qᵗ=0.015, 𝒰, ℳ)

# Time step
new_state = step_parcel!(state, model, Δt=1.0)
```

See also [`EnvironmentalProfile`](@ref), [`ParcelState`](@ref), [`step_parcel!`](@ref).
"""
struct ParcelModel{EP, MP, C}
    "Environmental atmospheric profile"
    profile :: EP

    "Microphysics scheme (e.g., `OneMomentCloudMicrophysics`)"
    microphysics :: MP

    "Thermodynamic constants"
    constants :: C
end

function Base.show(io::IO, model::ParcelModel)
    print(io, "ParcelModel(")
    print(io, "\n  profile = ", typeof(model.profile).name.name)
    print(io, "\n  microphysics = ", typeof(model.microphysics).name.name)
    print(io, "\n)")
end
