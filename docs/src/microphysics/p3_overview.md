# Predicted Particle Properties (P3) Microphysics

The Predicted Particle Properties (P3) scheme represents a paradigm shift in bulk microphysics parameterization.
Rather than using discrete hydrometeor categories (cloud ice, snow, graupel, hail), P3 uses a **single ice category**
with continuously predicted properties that evolve naturally as particles grow, rime, and melt.

## Motivation

Traditional bulk microphysics schemes partition frozen hydrometeors into separate categories:

| Category | Typical Properties |
|----------|-------------------|
| Cloud ice | Small, pristine crystals |
| Snow | Aggregated crystals, low density |
| Graupel | Heavily rimed, moderate density |
| Hail | Fully frozen, ice density |

This categorical approach creates artificial boundaries. A growing ice particle must "convert" from
one category to another through ad-hoc transfer terms, leading to:

- **Discontinuous property changes** when particles cross category thresholds
- **Arbitrary conversion parameters** that are difficult to constrain observationally
- **Loss of information** about particle history and evolution

P3 solves these problems by tracking the **physical properties** of ice particles directly:

- **Rime mass fraction** ``Fᶠ``: What fraction of particle mass is rime?
- **Rime density** ``ρᶠ``: How dense is the rime layer?
- **Liquid fraction** ``Fˡ``: How much unfrozen water coats the particle?

These properties evolve continuously through microphysical processes, and particle characteristics
(mass, fall speed, collection efficiency) are diagnosed from them.

## Key Features of P3

### Single Ice Category with Predicted Properties

Instead of discrete categories, P3 tracks a population of ice particles with a gamma size distribution:

```math
N'(D) = N₀ D^μ e^{-λD}
```

where ``D`` is particle maximum dimension. The **mass-diameter relationship** ``m(D)`` depends on
the predicted rime properties, allowing particles to transition smoothly from pristine crystals
to heavily rimed graupel.

### Three-Moment Ice

P3 v5.5 uses three prognostic moments for ice:
1. **Mass** (``ρqⁱ``): Total ice mass concentration
2. **Number** (``ρnⁱ``): Ice particle number concentration  
3. **Reflectivity** (``ρzⁱ``): Sixth moment, proportional to radar reflectivity

The third moment provides additional constraint on the size distribution, improving
representation of precipitation-sized particles.

### Predicted Liquid Fraction

[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) extended P3 to track liquid water on ice particles.
This is crucial for:
- **Wet growth**: Melting particles with liquid coatings
- **Shedding**: Liquid water dripping from large ice
- **Refreezing**: Coating that freezes into rime

## Scheme Evolution

| Version | Reference | Key Addition |
|---------|-----------|--------------|
| P3 v1 | [Morrison & Milbrandt (2015)](@cite Morrison2015parameterization) | Single ice category, predicted rime |
| P3 v2 | [Milbrandt & Morrison (2016)](@cite MilbrandtMorrison2016) | Three-moment ice (reflectivity) |
| P3 v5.5 | [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) | Predicted liquid fraction |

Our implementation follows **P3 v5.5** from the official [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

## Prognostic Variables

P3 tracks 9 prognostic variables for the hydrometeor population:

**Cloud liquid** (1 variable):
- ``ρqᶜˡ``: Cloud droplet mass concentration [kg/m³]

**Rain** (2 variables):
- ``ρqʳ``: Rain mass concentration [kg/m³]
- ``ρnʳ``: Raindrop number concentration [1/m³]

**Ice** (6 variables):
- ``ρqⁱ``: Total ice mass concentration [kg/m³]
- ``ρnⁱ``: Ice particle number concentration [1/m³]
- ``ρqᶠ``: Rime/frost mass concentration [kg/m³]
- ``ρbᶠ``: Rime volume concentration [m³/m³]
- ``ρzⁱ``: Ice 6th moment (reflectivity proxy) [m⁶/m³]
- ``ρqʷⁱ``: Liquid water on ice [kg/m³]

From these, diagnostic properties are computed:
- **Rime fraction**: ``Fᶠ = ρqᶠ / ρqⁱ``
- **Rime density**: ``ρᶠ = ρqᶠ / ρbᶠ``
- **Liquid fraction**: ``Fˡ = ρqʷⁱ / ρqⁱ``

## Quick Start

```@example p3_overview
using Breeze.Microphysics.PredictedParticleProperties

# Create P3 scheme with default parameters
microphysics = PredictedParticlePropertiesMicrophysics()
```

```@example p3_overview
# Access ice properties
microphysics.ice
```

```@example p3_overview
# Get prognostic field names
prognostic_field_names(microphysics)
```

## Documentation Outline

The following sections provide detailed documentation of the P3 scheme:

1. **[Particle Properties](@ref p3_particle_properties)**: Mass-diameter and area-diameter relationships
2. **[Size Distribution](@ref p3_size_distribution)**: Gamma PSD and parameter determination
3. **[Integral Properties](@ref p3_integral_properties)**: Bulk properties from PSD integrals
4. **[Microphysical Processes](@ref p3_processes)**: Process rate formulations
5. **[Prognostic Equations](@ref p3_prognostics)**: Tendency equations and model coupling

## References

The P3 scheme is described in detail in:

- [Morrison2015parameterization](@cite): Original P3 formulation with predicted rime
- [MilbrandtMorrison2016](@cite): Extension to three-moment ice  
- [MilbrandtEtAl2024](@cite): Predicted liquid fraction on ice

