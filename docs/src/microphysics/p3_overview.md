# [Predicted Particle Properties (P3) Microphysics](@id p3_overview)

The Predicted Particle Properties (P3) scheme represents a paradigm shift in bulk microphysics
parameterization. Rather than using discrete hydrometeor categories (cloud ice, snow, graupel,
hail), P3 uses a **single ice category** with continuously predicted properties that evolve
naturally as particles grow, rime, and melt.

This implementation tracks Fortran [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)
([Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) — the predicted-liquid-fraction extension —
and [Morrison et al. (2025)](@cite Morrison2025complete3moment) for full triple moment).

## Motivation

Traditional bulk microphysics schemes partition frozen hydrometeors into separate categories:

| Category | Typical Properties |
|----------|-------------------|
| Cloud ice | Small, pristine crystals |
| Snow | Aggregated crystals, low density |
| Graupel | Heavily rimed, moderate density |
| Hail | Fully frozen, ice density |

This categorical approach creates artificial boundaries. A growing ice particle must "convert"
from one category to another through ad-hoc transfer terms, leading to:

- **Discontinuous property changes** when particles cross category thresholds
- **Arbitrary conversion parameters** that are difficult to constrain observationally
- **Loss of information** about particle history and evolution

P3 solves these problems by tracking the **physical properties** of ice particles directly:

- **Rime mass fraction** ``F^f``: What fraction of particle mass is rime?
- **Rime density** ``ρ^f``: How dense is the rime layer?
- **Liquid fraction** ``F^l``: How much unfrozen water coats the particle?

These properties evolve continuously through microphysical processes, and particle
characteristics (mass, fall speed, collection efficiency) are diagnosed from them.

## Architectural choice: Breeze P3 updates tendencies, instead of prognostic variables

The Fortran reference is structured as a subcycle module that updates prognostic variables
in place over its internal Δt: it can hard-clamp ``N_i ≤ N_{i,\max}`` after each step,
zero out small-mass species and add a compensating ``θ`` correction, and use ``1/Δt``
relaxation rates for nucleation and saturation adjustment.

Breeze's P3 (`_p3_scalar_compute` in `p3_interface.jl`) returns a `P3CacheResult`
of *tendencies*, which Breeze sums with advection and diffusion before
time-stepping. P3 has no write access to the prognostic state and no awareness
of host Δt. This produces several deliberate, documented differences from
Fortran:

- **Hard prognostic clamps are replaced by read-time caps.** For example,
  `impose_max_Ni` becomes a 10-second relaxation sink toward
  ``N_{i,\max}/ρ`` rather than an instantaneous cap.
- **Per-Δt depletion rates use a fixed timescale.** Cooper nucleation,
  immersion/homogeneous freezing, and CCN activation use a 10 s relaxation
  in place of Fortran's ``1/Δt``.
- **Latent heating is delegated to the thermodynamics formulation.** The Anelastic
  and compressible formulations carry energy through their prognostic
  thermodynamic variable ``θ_{li}``.

These choices are noted in context throughout the documentation.

## Key Features of P3

### Single Ice Category with Predicted Properties

Instead of discrete categories, P3 tracks a population of ice particles with a gamma
size distribution

```math
N'(D) = N_0\, D^μ\, e^{-λD},
```

where ``D`` is the maximum particle dimension. The mass-diameter relationship ``m(D)``
depends on the predicted rime properties, allowing particles to transition smoothly from
pristine crystals to heavily rimed graupel. See [Particle Properties](@ref p3_particle_properties)
for the four-regime piecewise ``m(D)`` and ``A(D)`` laws and
[Size Distribution](@ref p3_size_distribution) for the closure that determines
``(N_0, λ, μ)`` from prognostic moments.

### Two- and Three-Moment Ice

P3 v5.5 supports both two-moment and three-moment ice. Breeze defaults to the
**two-moment** path; three-moment ice is opt-in (`three_moment_ice = true` in
the Breeze constructor).

Two-moment ice tracks:

1. **Mass** (``ρq^i``): Ice mass concentration (dry component; see prognostic table below).
2. **Number** (``ρn^i``): Ice particle number concentration.

Three-moment ice additionally tracks:

3. **Reflectivity** (``ρz^i``): Sixth moment, proportional to radar reflectivity.

The three-moment scheme provides additional constraint on the size distribution, improving
representation of precipitation-sized particles
([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024),
[Morrison et al. (2025)](@cite Morrison2025complete3moment)).

When three-moment ice is enabled, Breeze uses the active "hybrid" ``Z_i`` update
path: between processes, the shape parameter ``μ_i`` and the third statistical
moment of the PSD ``M_3 = ∫ D^3 N'(D) \, dD`` are recomputed from updated
``q_i`` and the bulk ice density (using ``q_i ≈ ρ̄_i \, (π/6) \, M_3`` for
spherical particles), then ``Z_i`` is reconstructed via ``G(μ_i)\, M_3^2 / N_i``.
Note that ``M_3`` is the third statistical moment of the size distribution
(mass-related), *not* the third prognostic moment of the scheme, which is
``ρz^i ∝ M_6``. Initiation processes (nucleation,
immersion freezing, splintering, homogeneous freezing) add explicit ``Z_i``
increments using the source PSD's ``μ`` (``μ_c`` for cloud water, ``μ_r`` for
rain — held at 0 at runtime — and 0 for all other source types).

### Predicted Liquid Fraction

[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) extended P3 to
track liquid water on ice particles. This is crucial for:

- **Wet growth**: Melting particles with liquid coatings.
- **Shedding**: Liquid water dripping from large ice.
- **Refreezing**: Coating that freezes into rime.

Breeze implements liquid-fraction wet growth, refreezing, and shedding.
Shedding uses the PSD integral over particles
with ``D \ge 9`` mm (tabulated as `f1pr28`); see
[Microphysical Processes](@ref p3_processes) for details.

## What is implemented

| Feature | Source |
|---------|--------|
| Four-regime piecewise mass–diameter and matching area–diameter relationships | [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) |
| Best-number terminal velocity with air-density correction ``(ρ_s/ρ)^{0.54}`` | [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005) |
| Cober–List rime density | [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) |
| Two-moment μ–λ closure (Heymsfield 2003 fit for small particles; rime-/density-weighted relation from the Fortran lookup-table generator for larger particles) | [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) |
| Sixth moment (reflectivity) as a prognostic variable | [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021), [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024), [Morrison et al. (2025)](@cite Morrison2025complete3moment) |
| Reflectivity-weighted fall speed | [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021), [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024), [Morrison et al. (2025)](@cite Morrison2025complete3moment) |
| Active hybrid ``Z_i`` update via ``G(μ_i)\, M_3^2/N_i`` after the continuous "group 1" processes, plus explicit increments for the initiation "group 2" processes via ``G(μ)\, ΔM_3^2/ΔN`` | [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021), [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024), [Morrison et al. (2025)](@cite Morrison2025complete3moment) |
| Liquid fraction prognostic variable (``ρq^{wi}``) | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |
| Wet growth and refreezing | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |
| Tabulated, size-thresholded (``D \ge 9`` mm) shedding | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |

## What is *not* implemented

!!! note "Multiple free ice categories"
    [Milbrandt & Morrison (2016)](@cite MilbrandtMorrison2016) introduced
    multiple free ice categories. Breeze defaults to a single ice
    category. Multi-category scaffolding (`MultiIceCategory`,
    `inter_category_collection`) exists in the source but is currently a
    placeholder that is not called from the tendency assembly. Adding
    multi-category support requires implementing both
    `compute_inter_category_collection` and the destination/merge logic.

!!! note "Subgrid cloud and precipitation fractions (SCPF)"
    Breeze runs permanently in the ``\text{SCF}=\text{SPF}=1`` limit. Fortran's SCPF
    diagnostic, which calls `compute_SCPF` three times per step to
    diagnose subgrid cloud cover from a bounded total-water PDF, is
    not ported.

!!! note "Adaptive sedimentation substepping"
    Sedimentation is routed through tracer transport rather than the
    Fortran's adaptive `dt_left` substepping based on the maximum Courant
    number. Tabulated reflectivity-weighted fall speed ``V_Z`` is
    computed but not used to set a Courant constraint inside P3.

!!! note "Alternative warm-rain options"
    The Fortran scheme exposes three autoconversion / accretion / rain
    self-collection options
    (``\mathtt{autoAccr\_param} \in \{\text{SB2001},\, \text{KK2000},\, \text{Kogan2013}\}``,
    default KK2000). Breeze hard-codes the KK2000 option; the SB2001 and
    Kogan 2013 alternatives are not exposed.

!!! note "Variable rain shape parameter"
    Both Breeze and Fortran v5.5.0 hold the rain shape parameter at
    ``μ_r = 0`` at runtime (the Cao-2008 variable-``μ_r`` block is
    commented out in the Fortran source). The closures used by Breeze
    are therefore identical to Fortran's runtime behaviour.

!!! note "Lookup-table I/O scope"
    Breeze reads the same Fortran ASCII ice lookup tables as the reference
    implementation (`p3_lookupTable_1`, `p3_lookupTable_2`,
    `p3_lookupTable_3`); the ice tables are not regenerated. The rain 1D
    tables (mass- and number-weighted fall speed, evaporation ventilation)
    *are* tabulated at startup from Chebyshev–Gauss quadrature via
    `tabulate_rain_from_quadrature`.

## Prognostic Variables

P3 evolves eleven prognostic densities:

**Cloud liquid** (2 variables):

- ``ρq^{cl}``: Cloud droplet mass concentration [kg/m³].
- ``ρn^{cl}``: Cloud droplet number concentration [1/m³] (always carried; values are taken
  from aerosol activation when prognostic activation is enabled, or held at the
  configured ``N_c`` constant otherwise).

**Rain** (2 variables):

- ``ρq^r``: Rain mass concentration [kg/m³].
- ``ρn^r``: Raindrop number concentration [1/m³].

**Ice** (6 variables):

- ``ρq^i``: Dry ice mass concentration [kg/m³] (rime + deposited mass; excludes ``ρq^{wi}``).
- ``ρn^i``: Ice particle number concentration [1/m³].
- ``ρq^f``: Rime mass concentration [kg/m³].
- ``ρb^f``: Rime volume concentration [m³/m³].
- ``ρz^i``: Ice 6th moment (reflectivity proxy) [m⁶/m³] (only updated when `three_moment_ice = true`).
- ``ρq^{wi}``: Liquid water on ice [kg/m³].

**Saturation diagnostic** (1 variable):

- ``ρs^{sat}``: Predicted supersaturation [kg/m³]
  ([Grabowski and Morrison (2008)](@cite GrabowskiMorrison2008)).
  Breeze exposes a `predict_supersaturation` flag on `ProcessRateParameters`,
  defaulting to `false`. When `false`, the prognostic field is inactive and
  has zero microphysical tendency; diagnostics that need local saturation use
  ``q^v - q^{v,s}(T)`` directly. When `true`, the bounded G&M (2008)
  adjustment fires before the M&G rates, shifting the local ``q^v``,
  ``q^{cl}``, and ``T`` (and thus ``q^{v,s}(T)``) so that
  ``q^v - q^{v,s}`` matches the advected ``s^{sat}``. The M&G semi-analytic
  rates then run on this post-G&M state — the "diagnostic supersaturation"
  they see is ``q^v_{\text{post-GM}} - q^{v,s}(T_{\text{post-GM}})``, not the
  host's ``s^{sat}`` field. The G&M adjustment and the end-of-step
  ``s^{sat}`` reset both relax over `sink_limiting_timescale`, so they land
  exactly when the host integrates with
  ``\Delta t = \text{sink\_limiting\_timescale}``.

From these, diagnostic properties are computed:

- **Rime fraction**: ``F^f = ρq^f / ρq^i`` (the prognostic ``ρq^i`` is dry ice,
  matching Fortran's ``Fr = qirim / (qitot - qiliq)``).
- **Rime density**: ``ρ^f = ρq^f / ρb^f``.
- **Liquid fraction**: ``F^l = ρq^{wi} / (ρq^i + ρq^{wi})`` (denominator is total
  ice mass, matching Fortran's ``Fl = qiliq / qitot``).

## Quick Start

```@example p3_overview
using Breeze

# Create a P3 scheme with default parameters
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

1. **[Particle Properties](@ref p3_particle_properties)**: Mass-diameter and area-diameter relationships.
2. **[Size Distribution](@ref p3_size_distribution)**: Gamma PSD and parameter determination.
3. **[Integral Properties](@ref p3_integral_properties)**: Bulk properties from PSD integrals.
4. **[Microphysical Processes](@ref p3_processes)**: Process rate formulations.
5. **[Prognostic Equations](@ref p3_prognostics)**: Tendency equations and model coupling.
6. **[Examples](@ref p3_examples)**: Worked examples and visualizations of P3 microphysics concepts.

## Complete References

### Core P3 Papers

- [Morrison2015parameterization](@citet): Original P3 formulation with predicted rime (Part I).
- [Morrison2015part2](@citet): Case study comparisons with observations (Part II).
- [MilbrandtMorrison2016](@citet): Extension to multiple free ice categories (Part III).
- [MilbrandtEtAl2021](@citet): Original three-moment ice in JAS.
- [MilbrandtEtAl2024](@citet): Updated triple-moment formulation in JAMES.
- [MilbrandtEtAl2025liquidfraction](@citet): Predicted liquid fraction on ice.
- [Morrison2025complete3moment](@citet): Complete three-moment implementation.

### Related Papers

- [MilbrandtYau2005](@citet): Multimoment microphysics and spectral shape parameter.
- [SeifertBeheng2006](@citet): Two-moment cloud microphysics for mixed-phase clouds.
- [KhairoutdinovKogan2000](@citet): Warm rain autoconversion parameterization.
- [pruppacher2010microphysics](@citet): Microphysics of clouds and precipitation (textbook).
