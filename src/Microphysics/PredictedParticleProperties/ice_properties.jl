#####
##### Ice properties
#####
##### The concept containers that hold the ice-side integrals of the P3 scheme —
##### fall speed, deposition, bulk properties, collection, the λ limiter, and
##### ice-rain collection — the lookup-table containers they are read from, and
##### the `Ice` container that gathers them.
#####
##### Every container follows the materialization pattern: the user-facing
##### constructor builds a skeleton whose integral fields are `nothing`, and
##### [`read_lookup_tables`](@ref) replaces each one with a tabulated integral.
#####

#####
##### Ice fall speed
#####
##### Terminal velocity integrals over the ice particle size distribution.
##### P3 computes number- and mass-weighted fall speeds.
#####

struct IceFallSpeed{FT, N, M}
    reference_air_density :: FT
    number_weighted :: N
    mass_weighted :: M
end

"""
$(TYPEDSIGNATURES)

Construct `IceFallSpeed` with parameters and quadrature-based integrals.

Ice particle terminal velocity uses the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best-number formulation with air density correction exponent 0.54 from
[Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007). The reference density
``ρ₀ = p₀ / (Rᵈ T₀)`` is the dry-air density at the reference conditions at which
the P3 lookup tables are computed, ``T₀ = 253.15`` K and ``p₀ = 600`` hPa, and
comes out at ≈0.825 kg/m³. It is *not* the surface reference density ≈1.275 kg/m³
that corrects rain fall speeds.

Two weighted fall speeds are computed by integrating over the size distribution:

- **Number-weighted** ``V_n``: For number flux (sedimentation of particle count)
- **Mass-weighted** ``V_m``: For mass flux (precipitation rate)

# Keyword Arguments

- `thermodynamic_constants`: Source of the dry-air gas constant used to diagnose
  the default reference-air density.
- `reference_pressure`: Pressure ``p₀`` [Pa] of the lookup-table reference state.
- `reference_temperature`: Temperature ``T₀`` [K] of the lookup-table reference state.
- `reference_air_density`: Reference ``ρ₀`` [kg/m³], by default diagnosed from
  `reference_pressure` and `reference_temperature`.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 20.
"""
function IceFallSpeed(FT::DataType = Oceananigans.defaults.FloatType;
                      thermodynamic_constants = ThermodynamicConstants(FT),
                      reference_pressure = 60000,     # 600 hPa
                      reference_temperature = 253.15, # -20 °C
                      reference_air_density = reference_pressure /
                      (dry_air_gas_constant(thermodynamic_constants) * reference_temperature))
    return IceFallSpeed(FT(reference_air_density), nothing, nothing)
end

Base.summary(::IceFallSpeed) = "IceFallSpeed"

function Base.show(io::IO, fs::IceFallSpeed)
    print(io, summary(fs), "(")
    print(io, "ρ₀=", fs.reference_air_density, ")")
end

#####
##### Ice deposition
#####
##### Vapor deposition/sublimation integrals for ice particles, including the
##### ventilation factors that account for enhanced vapor transport due to
##### particle motion through air.
#####

struct IceDeposition{V, V1, SC, SR, LC, LR}
    ventilation :: V
    ventilation_enhanced :: V1
    small_ice_ventilation_constant :: SC
    small_ice_ventilation_reynolds :: SR
    large_ice_ventilation_constant :: LC
    large_ice_ventilation_reynolds :: LR
end

"""
$(TYPEDSIGNATURES)

Construct `IceDeposition` with quadrature-based ventilation integrals.

Ice growth/decay by vapor deposition/sublimation follows the diffusion equation
with ventilation enhancement. The ventilation factor ``fᵛᵉ`` accounts for
enhanced vapor transport due to particle motion through air:

```math
fᵛᵉ = a + b \\cdot Sc^{1/3} Re^{1/2}
```

where ``Sc`` is the Schmidt number and ``Re`` is the Reynolds number.
[Hall and Pruppacher (1976)](@cite HallPruppacher1976) showed that falling
particles have significantly enhanced vapor exchange compared to stationary
particles.

Thermal conductivity ``κ`` and vapor diffusivity ``Dᵥ`` are computed at runtime
from temperature, pressure, and the model thermodynamic constants via
`air_transport_properties(T, P, constants)`. They are not stored on
`IceDeposition`.

**Basic ventilation integrals:**
- `ventilation`: Integrated over full size spectrum
- `ventilation_enhanced`: For larger particles (D > 100 μm)

**Size-regime ventilation** (for melting with liquid fraction):
- `small_ice_ventilation_*`: D ≤ Dcrit, meltwater → rain
- `large_ice_ventilation_*`: D > Dcrit, meltwater → liquid on ice

# References

[Hall and Pruppacher (1976)](@cite HallPruppacher1976),
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 34.
"""
function IceDeposition(::DataType = Oceananigans.defaults.FloatType)
    return IceDeposition(nothing, nothing, nothing, nothing, nothing, nothing)
end

Base.summary(::IceDeposition) = "IceDeposition"

Base.show(io::IO, d::IceDeposition) = print(io, summary(d), "()")

#####
##### Ice bulk properties
#####
##### Population-averaged properties computed by integrating over the ice
##### particle size distribution.
#####

struct IceBulk{FT, EF, DM, RH, RF, LA, MU, SH}
    maximum_mean_diameter :: FT
    minimum_mean_diameter :: FT
    effective_radius :: EF
    mean_diameter :: DM
    mean_density :: RH
    reflectivity :: RF
    slope :: LA
    shape :: MU
    shedding :: SH
end

"""
$(TYPEDSIGNATURES)

Construct `IceBulk` with parameters and quadrature-based integrals.

These integrals compute bulk properties by averaging over the particle
size distribution. They are used for radiation, radar, and diagnostics.

**Diagnostic integrals:**

- `effective_radius`: Radiation-weighted radius ``r_e = ∫A·N'dD / ∫N'dD``
- `mean_diameter`: Mass-weighted diameter ``D_m = ∫D·m·N'dD / ∫m·N'dD``
- `mean_density`: Mass-weighted density ``ρ̄ = ∫ρ·m·N'dD / ∫m·N'dD``
- `reflectivity`: Radar reflectivity ``Z = ∫D^6·N'dD``

**Distribution parameters (for λ-limiting):**

- `slope`: Slope parameter λ from prognostic constraints
- `shape`: Shape parameter μⁱ from empirical μⁱ-λ relationship

**Process integrals:**

- `shedding`: Rate at which meltwater sheds from large particles

# Keyword Arguments

- `maximum_mean_diameter`: Upper Dm limit [m], default 0.02 (2 cm)
- `minimum_mean_diameter`: Lower Dm limit [m], default 2×10⁻⁶ (2 μm)

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Field et al. (2007)](@cite FieldEtAl2007) for μⁱ-λ relationship.
"""
function IceBulk(FT::DataType = Oceananigans.defaults.FloatType;
                 maximum_mean_diameter = 20e-3,
                 minimum_mean_diameter = 2e-6)
    return IceBulk(
        FT(maximum_mean_diameter),
        FT(minimum_mean_diameter),
        nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

Base.summary(::IceBulk) = "IceBulk"

function Base.show(io::IO, bp::IceBulk)
    print(io, summary(bp), "(")
    print(io, "Dmax=", bp.maximum_mean_diameter, ", ")
    print(io, "Dmin=", bp.minimum_mean_diameter, ")")
end

#####
##### Ice collection
#####
##### Single-ice-PSD collision-collection integrals: ice-ice aggregation,
##### ice-cloud-water collection, and aerosol scavenging.
#####

struct IceCollection{AG, CW, WA, IA}
    aggregation :: AG
    cloud_collection :: CW
    cloud_aerosol_collection :: WA
    ice_aerosol_collection :: IA
end

"""
$(TYPEDSIGNATURES)

Construct `IceCollection` with placeholder (`nothing`) integrals, following the
materialization pattern: [`read_lookup_tables`](@ref) replaces each field
with the corresponding tabulated integral.

Collection processes describe ice particles sweeping up other hydrometeors
through gravitational settling. The integrals held here are the ones that depend
on the ice size distribution alone:

**Aggregation** (ice + ice → larger ice):
Ice particles collide and stick together to form larger aggregates. This is the
dominant growth mechanism for snow, and depends on the differential fall speeds of
particles of different sizes. Consumed by [`ice_aggregation_rate`](@ref).

**Cloud collection** (ice + cloud droplets → rime on ice):
The PSD-integrated sweep-out kernel ``\\int V(D) A(D) N'(D) \\, dD`` [m³/s] per
particle, with the collision kernel set to zero for ice diameters below 100 μm.
Cloud droplets are small enough relative to ice that their own size distribution
does not enter the collision geometry, so a single ice-PSD integral suffices.
Consumed by [`cloud_riming_rate`](@ref) below freezing and by
[`cloud_warm_collection_rate`](@ref) above it.

**Aerosol scavenging** (`cloud_aerosol_collection`, `ice_aerosol_collection`):
Collection by ice particles of water-friendly and ice-friendly interstitial
aerosol, respectively.

Ice-rain collection is handled separately, by [`IceRainCollection`](@ref) and the
5D rain-ice block embedded in Lookup Table 1, because its kernel needs the
rain slope parameter ``λ_r`` in addition to the ice PSD.

Collection efficiencies are not stored here. They live in
[`ProcessRate`](@ref) alongside the other rate parameters, as
`cloud_ice_collection_efficiency` (``E^{ci}``) and
`rain_ice_collection_efficiency` (``E^{ri}``).

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Sections 2d-e,
[Milbrandt and Yau (2005)](@cite MilbrandtYau2005).
"""
IceCollection() = IceCollection(nothing, nothing, nothing, nothing)

Base.summary(::IceCollection) = "IceCollection"

Base.show(io::IO, ::IceCollection) = print(io, "IceCollection(4 integrals)")

#####
##### Ice lambda limiter
#####
##### Integrals used to limit the slope parameter λ of the gamma size
##### distribution to physically reasonable values.
#####

struct IceLambdaLimiter{S, L}
    small_q :: S
    large_q :: L
end

"""
$(TYPEDSIGNATURES)

Construct `IceLambdaLimiter` with quadrature-based integrals.

The slope parameter λ of the gamma size distribution can become
unrealistically large or small as prognostic moments evolve. This
happens at edges of mixed-phase regions or during rapid microphysical
adjustments.

**Physical interpretation:**
- Very large λ → all particles tiny (mean size → 0)
- Very small λ → all particles huge (mean size → ∞)

These integrals compute the limiting values:
- `small_q`: λ limit when q is small (prevents vanishingly tiny particles)
- `large_q`: λ limit when q is large (prevents unrealistically huge particles)

The limiter ensures the diagnosed size distribution remains physically
sensible even when the prognostic constraints become degenerate.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function IceLambdaLimiter()
    return IceLambdaLimiter(nothing, nothing)
end

Base.summary(::IceLambdaLimiter) = "IceLambdaLimiter"
Base.show(io::IO, ::IceLambdaLimiter) = print(io, "IceLambdaLimiter(2 integrals)")

#####
##### Ice-rain collection
#####
##### Collection integrals for ice particles collecting rain drops. These are
##### computed for multiple rain size bins in the P3 scheme.
#####

struct IceRainCollection{QR, NR}
    mass :: QR
    number :: NR
end

"""
$(TYPEDSIGNATURES)

Construct a placeholder `IceRainCollection` with `nothing` fields.

The actual ice-rain collection integrals are double integrals over both
the ice and rain size distributions, tabulated offline in the P3 lookup tables.
This placeholder is overwritten when tables are loaded via `read_lookup_tables`.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
function IceRainCollection()
    return IceRainCollection(nothing, nothing)
end

Base.summary(::IceRainCollection) = "IceRainCollection"
Base.show(io::IO, ::IceRainCollection) = print(io, "IceRainCollection(2 integrals)")

#####
##### Ice: the container combining all ice particle property concepts
#####

struct Ice{FT, FS, DP, BP, CL, LL, IR}
    # Top-level parameters
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    maximum_shape_parameter :: FT
    # Concept containers. Each one owns the lookup arrays for its block of Lookup
    # Table 1, so `on_architecture` transfers every table to the device exactly once.
    fall_speed :: FS
    deposition :: DP
    bulk :: BP
    collection :: CL
    lambda_limiter :: LL
    ice_rain :: IR
end

"""
$(TYPEDSIGNATURES)

Construct ice particle properties with parameters and integrals for the P3 scheme.

Ice particles in P3 span a continuum from small pristine crystals to large
heavily-rimed graupel. The particle mass ``m(D)`` follows a piecewise power
law depending on size ``D``, rime fraction ``Fᶠ``, and rime density ``ρᶠ``.

# Physical Concepts

This container organizes all ice-related computations:

- **Fall speed**: Terminal velocity integrals for sedimentation
  (number-weighted, mass-weighted)
- **Deposition**: Ventilation integrals for vapor diffusion growth
- **Bulk properties**: Population-averaged diameter, density, reflectivity
- **Collection**: Integrals for aggregation and riming rates
- **Lambda limiter**: Constraints on size distribution slope

# Keyword Arguments

- `thermodynamic_constants`: Source of constants used by the ice-property defaults.
- `minimum_rime_density`: Lower bound for ρᶠ [kg/m³], default 50
- `maximum_rime_density`: Upper bound for ρᶠ [kg/m³], default 900 (pure ice)
- `maximum_shape_parameter`: Upper limit on μⁱ [-], default 20

# References

The mass-diameter relationship is from
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
function Ice(FT::DataType = Oceananigans.defaults.FloatType;
             thermodynamic_constants = ThermodynamicConstants(FT),
             minimum_rime_density = 50,
             maximum_rime_density = 900,
             maximum_shape_parameter = 20)
    return Ice(
        FT(minimum_rime_density),
        FT(maximum_rime_density),
        FT(maximum_shape_parameter),
        IceFallSpeed(FT; thermodynamic_constants),
        IceDeposition(FT),
        IceBulk(FT),
        IceCollection(),
        IceLambdaLimiter(),
        IceRainCollection())
end

Base.summary(::Ice) = "Ice"

function Base.show(io::IO, ice::Ice)
    print(io, summary(ice), '\n')
    print(io, "├── ρᶠ: [", ice.minimum_rime_density, ", ", ice.maximum_rime_density, "] kg/m³\n")
    print(io, "├── μmax: ", ice.maximum_shape_parameter, "\n")
    print(io, "├── ", ice.fall_speed, "\n")
    print(io, "├── ", ice.deposition, "\n")
    print(io, "├── ", ice.bulk, "\n")
    print(io, "├── ", ice.collection, "\n")
    print(io, "├── ", ice.lambda_limiter, "\n")
    print(io, "└── ", ice.ice_rain)
end
