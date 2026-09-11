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

# Two-Moment Ice

The scheme carries two prognostic moments for ice particles:
1. **Mass** (``qⁱ``): Total ice mass
2. **Number** (``nⁱ``): Ice particle number concentration

# Prognostic Variables

The scheme tracks 8 prognostic densities by default, and up to 11 with every option on:

| Variable | Description | Carried when |
|----------|-------------|--------------|
| ``ρqᶜˡ`` | Cloud liquid mass | always |
| ``ρqʳ``, ``ρnʳ`` | Rain mass and number | always |
| ``ρqⁱ``, ``ρnⁱ`` | Ice mass and number | always |
| ``ρqᶠ``, ``ρbᶠ`` | Rime mass and volume | always |
| ``ρqʷⁱ`` | Liquid water on ice | always |
| ``ρsᵛ⁺ˡ`` | Predicted liquid supersaturation | `predict_supersaturation` |
| ``ρnᶜˡ``, ``ρnᵃ`` | Cloud number and unactivated aerosol number | `aerosol` |

Each optional group is gated on a type, so a configuration that does not use one neither
allocates nor advects it. Cloud droplet number is prognostic only with an
`AerosolActivation`: the default prescribed-Nᶜˡ path takes it from the scheme
parameter `cloud.number_concentration`.

# Keyword Arguments

- `thermodynamic_constants`: Source of shared phase and dry-air properties.
- `lookup_tables`: Path to a directory containing P3 lookup table files
  (default to the artifact `P3_lookup_tables` in `Artifacts.toml`).
- `minimum_mass_mixing_ratio`: Mass below which a species is treated as absent
  [kg/kg] (default 10⁻¹⁴)
- `minimum_number_mixing_ratio`: Number below which a population is treated as
  absent [kg⁻¹] (default 10⁻¹⁶)
- `cloud`: [`CloudDroplets`](@ref) holding the prescribed droplet number and the
  [`CloudShape`](@ref) every μᶜˡ diagnosis reads. `nothing` (default) uses
  `CloudDroplets(FT)`.
- `rain`: [`RainDrops`](@ref) skeleton holding the
  [`RainFallSpeed`](@ref) the startup quadrature integrates and the
  [`RainVentilation`](@ref) the evaporation and coupled-adjustment rates read.
  `nothing` (default) uses `RainDrops(FT)`. Its lookup fields are materialized by
  `read_lookup_tables`; every supplied parameter is preserved.
- `precipitation_boundary_condition`: Boundary condition for surface precipitation.
  `nothing` (default) is an open surface: the diagnosed fall speed is retained at the
  bottom face, so all sedimenting species leave the domain. `ImpenetrableBoundaryCondition()`
  zeroes the fall speed there instead, so precipitation accumulates in the lowest cell.
- `negative_moisture_correction`: Repair of negative densities left by the advection
  operator, applied at the top of `update_state!`. Defaults to `SpeciesBorrowing()`,
  which borrows along the chain ``ρqʷⁱ ← ρqⁱ ← ρqʳ ← ρqᶜˡ ← ρqᵛ``, zeroes number and
  rime fields orphaned by a vanishing ice mass, and clamps negative number and rime
  densities. Pass `SpeciesBorrowing(vertical_borrowing = VerticalBorrowing())`
  to additionally redistribute leftover vapor deficits within each column, or `nothing`
  to disable the repair (P3's process rates then see zero-clamped values while the
  prognostic fields keep their negative mass).

# Prognostic CCN Activation

Pass `aerosol = AerosolActivation(AerosolMode())` to enable prognostic cloud
droplet number from aerosol activation physics (Morrison & Grabowski 2007).
When `aerosol = nothing` (default), cloud droplet number uses the prescribed
`CloudDroplets.number_concentration`.

# Configuring the empirical warm-phase parameters

The cloud-width, rain fall-speed, and rain-ventilation fits are each owned by a small
parameter container that is visible from this constructor. Custom values are threaded
through the startup quadrature and every runtime kernel:

```jldoctest
using Breeze
using Breeze.Microphysics.PredictedParticleProperties:
    CloudDroplets, CloudShape,
    RainDrops, RainFallSpeed, RainVentilation

cloud = CloudDroplets(Float64;
    shape = CloudShape(Float64; maximum_shape_parameter = 12))

rain = RainDrops(Float64;
    fall_speed = RainFallSpeed(Float64; plateau_velocity = 9.5),
    ventilation = RainVentilation(Float64; reynolds_coefficient = 0.35))

p3 = P3Microphysics(Float64; cloud, rain)
p3.rain.ventilation

# output
RainVentilation(ℂᵛᵉⁿᵗ₁=0.78, ℂᵛᵉⁿᵗ₂=0.35)
```

# Example

```jldoctest
using Breeze

# The `P3_lookup_tables` artifact is lazy: Pkg downloads it on first use
microphysics = PredictedParticlePropertiesMicrophysics()

# output
PredictedParticlePropertiesMicrophysics
├── ρʷ: 1000.0 kg/m³
├── qmin: 1.0e-14 kg/kg
├── ice: IceParticles
├── rain: RainDrops
├── cloud: CloudDroplets
├── process_rates: ProcessRate
├── negative_moisture_correction: SpeciesBorrowing(vertical_borrowing = nothing)
├── aerosol: nothing (prescribed CCN)
└── warm_rain_scheme: KhairoutdinovKogan2000
```

# References

This implementation follows P3 v5.5 from the
[P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

Key papers describing P3:
- [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization): Original scheme
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Predicted liquid fraction

See also the [P3 documentation](@ref p3_overview) for detailed physics.
"""
function PredictedParticlePropertiesMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                                 lookup_tables = artifact"P3_lookup_tables",
                                                 thermodynamic_constants = ThermodynamicConstants(FT),
                                                 minimum_mass_mixing_ratio = 1e-14,
                                                 minimum_number_mixing_ratio = 1e-16,
                                                 precipitation_boundary_condition = nothing,
                                                 negative_moisture_correction = SpeciesBorrowing(),
                                                 aerosol = nothing,
                                                 cloud = nothing,
                                                 rain = nothing,
                                                 process_rates = nothing,
                                                 predict_supersaturation = false,
                                                 warm_rain_scheme = KhairoutdinovKogan2000())
    if isnothing(process_rates)
        process_rates = ProcessRate(FT; thermodynamic_constants,
                                    predict_supersaturation)
    end
    return read_lookup_tables(lookup_tables; FT,
                              thermodynamic_constants,
                              minimum_mass_mixing_ratio, minimum_number_mixing_ratio,
                              precipitation_boundary_condition,
                              negative_moisture_correction,
                              aerosol, cloud, rain, process_rates, warm_rain_scheme)
end

# Shorthand alias
const P3Microphysics = PredictedParticlePropertiesMicrophysics

Base.summary(::PredictedParticlePropertiesMicrophysics) = "PredictedParticlePropertiesMicrophysics"

function Base.show(io::IO, p3::PredictedParticlePropertiesMicrophysics)
    print(io, summary(p3), '\n')
    print(io, "├── ρʷ: ", p3.process_rates.liquid_water_density, " kg/m³\n")
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

# Note: prognostic_field_names is implemented in p3_microphysical_state.jl to extend
# AtmosphereModels.prognostic_field_names
