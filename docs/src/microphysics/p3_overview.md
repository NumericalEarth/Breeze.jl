# [Predicted Particle Properties (P3) Microphysics](@id p3_overview)

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
representation of precipitation-sized particles. This was introduced in
[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) and further refined in
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) and
[Morrison et al. (2025)](@cite Morrison2025complete3moment).

### Predicted Liquid Fraction

[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) extended P3 to track liquid water on ice particles.
This is crucial for:
- **Wet growth**: Melting particles with liquid coatings
- **Shedding**: Liquid water dripping from large ice
- **Refreezing**: Coating that freezes into rime

## Scheme Evolution and Citation Guide

The P3 scheme has evolved through multiple papers. Here we document what each paper contributes
and which equations from each paper are implemented:

| Version | Reference | Key Contributions | Status |
|---------|-----------|-------------------|--------|
| P3 v1.0 | [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) | Single ice category, predicted rime, m(D) relationships | ✓ Implemented |
| P3 v1.0 | [Morrison et al. (2015b)](@cite Morrison2015part2) | Case study validation | For reference only |
| P3 v2.0 | [Milbrandt & Morrison (2016)](@cite MilbrandtMorrison2016) | Multiple free ice categories | ⚠ Not implemented |
| P3 v3.0 | [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) | Three-moment ice (Z prognostic) | ✓ Implemented |
| P3 v4.0 | [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) | Updated triple-moment formulation | ✓ Implemented |
| P3 v5.0 | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) | Predicted liquid fraction | ✓ Implemented |
| P3 v5.5 | [Morrison et al. (2025)](@cite Morrison2025complete3moment) | Complete three-moment implementation | ✓ Reference implementation |

Our implementation follows **P3 v5.5** from the official [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

### What We Implement

From [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization):
- Mass-diameter relationship with four regimes (Equations 1-5)
- Area-diameter relationship (Equations 6-8)
- Terminal velocity parameterization (Equations 9-11)
- Rime density parameterization
- μ-λ relationship for size distribution closure

From [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) and [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024):
- Sixth moment (reflectivity) as prognostic variable
- Reflectivity-weighted fall speed
- Z-tendency from each microphysical process

From [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction):
- Liquid fraction prognostic variable (``ρqʷⁱ``)
- Shedding process
- Refreezing process

### What We Do NOT Implement (Future Work)

!!! note "Multiple Ice Categories"
    [Milbrandt & Morrison (2016)](@cite MilbrandtMorrison2016) introduced **multiple free ice categories**
    that can coexist and interact. Our implementation uses a single ice category. Multiple categories
    may be added in a future version to better represent environments with distinct ice populations
    (e.g., anvil ice vs. convective ice).

!!! note "Full Process Rate Parameterizations"
    The full process rate formulations from the P3 papers are documented in [Microphysical Processes](@ref p3_processes)
    but are not yet all implemented as tendency functions. Current implementation provides the
    integral infrastructure for computing bulk rates; the complete tendency equations are a TODO.

!!! note "Saturation Adjustment-Free Approach"
    The E3SM implementation includes modifications for saturation adjustment-free supersaturation
    evolution. Our implementation currently uses saturation adjustment for cloud liquid.

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

## Complete References

The P3 scheme is described in detail in the following papers:

### Core P3 Papers

- [Morrison2015parameterization](@cite): Original P3 formulation with predicted rime (Part I)
- [Morrison2015part2](@cite): Case study comparisons with observations (Part II)
- [MilbrandtMorrison2016](@cite): Extension to multiple free ice categories (Part III)
- [MilbrandtEtAl2021](@cite): Original three-moment ice in JAS
- [MilbrandtEtAl2024](@cite): Updated triple-moment formulation in JAMES
- [MilbrandtEtAl2025liquidfraction](@cite): Predicted liquid fraction on ice
- [Morrison2025complete3moment](@cite): Complete three-moment implementation

### Related Papers

- [MilbrandtYau2005](@cite): Multimoment microphysics and spectral shape parameter
- [SeifertBeheng2006](@cite): Two-moment cloud microphysics for mixed-phase clouds
- [KhairoutdinovKogan2000](@cite): Warm rain autoconversion parameterization
- [pruppacher2010microphysics](@cite): Microphysics of clouds and precipitation (textbook)

