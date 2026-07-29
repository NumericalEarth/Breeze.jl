#####
##### Predicted Particle Properties (P3) Microphysics Scheme
#####
##### Main type combining ice, rain, and cloud properties.
#####

using Breeze.AtmosphereModels: SpeciesBorrowing

using Artifacts: @artifact_str
using LazyArtifacts: LazyArtifacts

"""
    PredictedParticlePropertiesMicrophysics

The Predicted Particle Properties (P3) microphysics scheme. See the constructor
[`PredictedParticlePropertiesMicrophysics()`](@ref) for usage and documentation.
"""
struct PredictedParticlePropertiesMicrophysics{FT, ICE, RAIN, CLOUD, PRP, BC, NMC, AERO, WRS}
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
    # Repair of negative densities produced by the (non-positive-definite) advection operator
    negative_moisture_correction :: NMC
    # Aerosol activation (nothing = prescribed CCN, AerosolActivation = prognostic CCN)
    aerosol :: AERO
    # Warm-rain (autoconversion/accretion/self-collection) scheme selector
    warm_rain_scheme :: WRS
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

The three-moment scheme improves representation of precipitation-sized
particles and enables better simulation of radar reflectivity. The default
runtime path is 2-moment ice; pass `three_moment_ice = true` to enable the
3-moment path, which uses `lookupTable_3` for distribution parameter closure.

# Prognostic Variables

The scheme tracks 8 prognostic densities by default, and up to 12 with every option on:

| Variable | Description | Carried when |
|----------|-------------|--------------|
| ``ρqᶜˡ`` | Cloud liquid mass | always |
| ``ρqʳ``, ``ρnʳ`` | Rain mass and number | always |
| ``ρqⁱ``, ``ρnⁱ`` | Ice mass and number | always |
| ``ρqᶠ``, ``ρbᶠ`` | Rime mass and volume | always |
| ``ρqʷⁱ`` | Liquid water on ice | always |
| ``ρz̃ⁱ`` | Advected square-root ice sixth moment, ``ρ sqrt(zⁱ nⁱ)`` | `three_moment_ice` |
| ``ρsˢᵃᵗ`` | Predicted supersaturation | `predict_supersaturation` |
| ``ρnᶜˡ``, ``ρnᵃ`` | Cloud number and unactivated aerosol number | `aerosol` |

Each optional group is gated on a type, so a configuration that does not use one neither
allocates nor advects it. Cloud droplet number is prognostic only with an
`AerosolActivation`: the default prescribed-Nᶜ path (Fortran `log_predictNc = .false.`)
takes it from the scheme parameter `cloud.number_concentration`.

# Keyword Arguments

- `lookup_tables`: Path to a directory containing Fortran P3 lookup table files
  (default to the artifact `P3_lookup_tables` in `Artifacts.toml`).
- `three_moment_ice`: 2-moment (`false`, default) or 3-moment (`true`) ice.
  Pass `nothing` to auto-detect from file presence (prefers 3-moment if
  available).
- `water_density`: Liquid water density [kg/m³] (default 1000)
- `precipitation_boundary_condition`: Boundary condition for surface precipitation.
  `nothing` (default) is an open surface: the diagnosed fall speed is retained at the
  bottom face, so all sedimenting species leave the domain. `ImpenetrableBoundaryCondition()`
  zeroes the fall speed there instead, so precipitation accumulates in the lowest cell.
- `negative_moisture_correction`: Repair of negative densities left by the advection
  operator, applied at the top of `update_state!`. Defaults to `SpeciesBorrowing()`,
  which borrows along the chain ``ρqʷⁱ ← ρqⁱ ← ρqʳ ← ρqᶜˡ ← ρqᵛ``, zeroes number and
  rime fields orphaned by a vanishing ice mass, and clamps negative number, rime, and
  reflectivity densities. Pass `SpeciesBorrowing(vertical_borrowing = VerticalBorrowing())`
  to additionally redistribute leftover vapor deficits within each column, or `nothing`
  to disable the repair (P3's process rates then see `clamp_positive`ed values while the
  prognostic fields keep their negative mass).

# Prognostic CCN Activation

Pass `aerosol = AerosolActivation(AerosolMode())` to enable prognostic cloud
droplet number from aerosol activation physics (Morrison & Grabowski 2007).
When `aerosol = nothing` (default), cloud droplet number uses the prescribed
`CloudDropletProperties.number_concentration`.

# Example

```julia
using Breeze

# Tables auto-download on first use
microphysics = PredictedParticlePropertiesMicrophysics()
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
                                                 lookup_tables = artifact"P3_lookup_tables",
                                                 three_moment_ice = false,
                                                 water_density = 1000,
                                                 precipitation_boundary_condition = nothing,
                                                 negative_moisture_correction = SpeciesBorrowing(),
                                                 aerosol = nothing,
                                                 cloud = nothing,
                                                 process_rates = nothing,
                                                 predict_supersaturation = false,
                                                 warm_rain_scheme = KhairoutdinovKogan2000())
    if isnothing(process_rates)
        process_rates = ProcessRateParameters(FT; predict_supersaturation)
    end
    return read_fortran_lookup_tables(lookup_tables; FT, three_moment_ice,
                                      water_density, precipitation_boundary_condition,
                                      negative_moisture_correction,
                                      aerosol, cloud, process_rates, warm_rain_scheme)
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
    print(io, "├── process_rates: ", summary(p3.process_rates), "\n")
    print(io, "├── negative_moisture_correction: ",
          isnothing(p3.negative_moisture_correction) ? "nothing (no repair)" :
              summary(p3.negative_moisture_correction), "\n")
    print(io, "├── aerosol: ", isnothing(p3.aerosol) ? "nothing (prescribed CCN)" : summary(p3.aerosol), "\n")
    print(io, "└── warm_rain_scheme: ", summary(p3.warm_rain_scheme))
end

# Note: prognostic_field_names is implemented in p3_interface.jl to extend
# AtmosphereModels.prognostic_field_names
