# [Predicted Particle Properties (P3) Microphysics](@id p3_overview)

The Predicted Particle Properties (P3) scheme represents a paradigm shift in bulk microphysics
parameterization. Rather than using discrete hydrometeor categories (cloud ice, snow, graupel,
hail), P3 uses a **single ice category** with continuously predicted properties that evolve
naturally as particles grow, rime, and melt.

This implementation tracks Fortran [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)
([Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) and
[Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) — the predicted-liquid-fraction extension).

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
in place over its internal Δt: it can hard-clamp ``n^i ≤ N^i_\text{max}/ρ`` after each step,
zero out small-mass species and add a compensating ``θ`` correction, and use ``1/Δt``
relaxation rates for nucleation and saturation adjustment.

Breeze's P3 returns *tendencies*, which Breeze sums with advection and
diffusion before time-stepping. On a grid, `compute_microphysical_tendencies!`
(`p3_driver.jl`) launches one kernel that writes a per-field tendency cache and
a second that adds it to ``G^n``; gridless callers (`ParcelModels`) go through
the `microphysical_tendency` methods in `p3_microphysical_tendencies.jl`. Both
paths funnel into `p3_tendency_compute` (`p3_microphysical_state.jl`), which
assembles the per-field tendencies from `prognostic_tendencies.jl`. P3 has no
write access to the prognostic state and no awareness of host Δt. This produces
several deliberate, documented differences from Fortran:

- **Hard prognostic clamps are replaced by tendency-form relaxations.** For
  example, `impose_max_Ni` becomes a relaxation sink toward ``N_{i,\max}/ρ``
  over `sink_limiting_timescale` (default 10 s) rather than an instantaneous cap.
- **Per-Δt depletion rates use a fixed timescale.** Cooper nucleation and
  homogeneous freezing relax over `ice_nucleation_timescale` /
  `homogeneous_freezing_timescale` (both 10 s by default) in place of Fortran's
  ``1/Δt``; CCN activation uses its own `aerosol.activation_timescale`
  (default 1 s). Every per-species sink budget is likewise sized against
  `sink_limiting_timescale`. For a single forward update no longer than that
  interval, the limited P3 sinks cannot remove more than their donor reservoir.
  This is a rate-budget guarantee, not an exact equivalence between Breeze's RK
  tendency update and Fortran's in-place one-shot operator.
- **Latent heating is delegated to the thermodynamics formulation.** The Anelastic
  and compressible formulations carry energy through their prognostic
  thermodynamic variable ``θ_{li}``.
- **Negative densities are repaired by the host, not by P3.** The advection
  operator is not positive-definite, so `update_state!` applies P3's
  `negative_moisture_correction` (a `SpeciesBorrowing` by default) before the
  rates are evaluated; see [Prognostic Equations](@ref p3_prognostics).

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

### Two-Moment Ice

Breeze runs the **two-moment** ice path, which tracks:

1. **Mass** (``ρq^i``): Ice mass concentration (dry component; see prognostic table below).
2. **Number** (``ρn^i``): Ice particle number concentration.

The shape parameter ``μ^i`` is diagnosed from the ``μ``–``λ`` closure tabulated in
Fortran Lookup Table 1.

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
| Liquid fraction prognostic variable (``ρq^{wi}``) | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |
| Wet growth and refreezing | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |
| Tabulated, size-thresholded (``D \ge 9`` mm) shedding | [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction) |

## What is *not* implemented

!!! note "Three-moment ice"
    [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) added the sixth moment
    (radar reflectivity) as a third prognostic ice moment. Breeze runs
    two-moment ice only; the reflectivity prognostic, its reflectivity-weighted
    fall speed, and the Table-3 ``μ`` closure are not implemented.

!!! note "Multiple free ice categories"
    [Milbrandt & Morrison (2016)](@cite MilbrandtMorrison2016) introduced
    multiple free ice categories. Breeze runs a single ice category.
    Adding multi-category support requires an inter-category collection
    kernel plus the destination/merge logic; neither is present.

!!! note "Subgrid cloud and precipitation fractions (SCPF)"
    Breeze runs permanently in the `SCF = SPF = 1` limit. Fortran's SCPF
    diagnostic, which calls `compute_SCPF` three times per step to
    diagnose subgrid cloud cover from a bounded total-water PDF, is
    not ported.

!!! note "Adaptive sedimentation substepping"
    Sedimentation is routed through tracer transport rather than the
    Fortran's adaptive `dt_left` substepping based on the maximum Courant
    number.

!!! note "Lookup-table I/O scope"
    Breeze reads the same Fortran ASCII ice lookup table as the reference
    implementation: `p3_lookupTable_1.dat-v6.9-2momI`, which carries
    both the 5-D ice-only integrals and the embedded 6-D ice–rain collection
    block. The ice tables are not regenerated. The rain 1D tables (mass- and number-weighted
    fall speed, evaporation ventilation) *are* tabulated at startup from
    Chebyshev–Gauss quadrature via `tabulate_rain_from_quadrature`.

## Equivalences with the Fortran runtime

These are options where Fortran and Breeze differ in *form* but agree on
what actually runs.

!!! note "Alternative warm-rain options"
    The Fortran scheme exposes three autoconversion / accretion / rain
    self-collection options
    (``\mathtt{autoAccr\_param} \in \{\text{SB2001},\, \text{KK2000},\, \text{Kogan2013}\}``,
    default KK2000). Breeze implements the default only:
    [Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000), selected
    through the `warm_rain_scheme` keyword as `KhairoutdinovKogan2000()`. The
    scheme also sets the seed-drop mass used to convert the autoconversion mass
    rate into a rain number source.

!!! note "Variable rain shape parameter"
    Both Breeze and Fortran v5.5.0 hold the rain shape parameter at
    ``μ^r = 0`` at runtime (the Cao-2008 variable-``μ^r`` block is
    commented out in the Fortran source). The closures used by Breeze
    are therefore identical to Fortran's runtime behaviour.

!!! note "Prescribed vs. prognostic droplet number"
    Fortran v5.5.0 runs with `log_predictNc = .false.`, taking cloud droplet
    number from a scheme constant. That is Breeze's default too
    (`cloud.number_concentration`). Passing
    `aerosol = AerosolActivation(AerosolMode())` switches on the prognostic
    path, which adds ``ρn^{cl}`` and an unactivated-aerosol reservoir
    ``ρn^a`` to the prognostic set.

## Prognostic Variables

P3 evolves eight prognostic densities by default, and up to twelve with every option
enabled. Each optional group is gated on a type, so a configuration that does not use one
neither allocates nor advects it.

**Cloud liquid** (1–2 variables):

- ``ρq^{cl}``: Cloud droplet mass concentration [kg/m³].
- ``ρn^{cl}``: Cloud droplet number concentration [1/m³], prognostic only when aerosol
  activation is enabled. Otherwise droplet number is the scheme parameter
  `cloud.number_concentration` and this field does not exist.

**Aerosol** (0–1 variables):

- ``ρn^a``: Unactivated aerosol number concentration [1/m³], allocated together
  with ``ρn^{cl}`` when `aerosol isa AerosolActivation`. Each activated droplet
  removes one unit from this reservoir.

**Rain** (2 variables):

- ``ρq^r``: Rain mass concentration [kg/m³].
- ``ρn^r``: Raindrop number concentration [1/m³].

**Ice** (5 variables):

- ``ρq^i``: Dry ice mass concentration [kg/m³] (rime + deposited mass; excludes ``ρq^{wi}``).
- ``ρn^i``: Ice particle number concentration [1/m³].
- ``ρq^f``: Rime mass concentration [kg/m³].
- ``ρb^f``: Rime volume concentration [m³/m³].
- ``ρq^{wi}``: Liquid water on ice [kg/m³].

**Saturation diagnostic** (0–1 variables):

- ``ρs^{sat}``: Predicted supersaturation [kg/m³]
  ([Grabowski and Morrison (2008)](@cite GrabowskiMorrison2008)).
  Breeze exposes a `predict_supersaturation` flag on `ProcessRateParameters`,
  defaulting to `false`. When `false`, the field is not allocated and is
  absent from `prognostic_field_names`; diagnostics that need local saturation use
  ``q^v - q^{v+l}(T)`` directly. When `true`, the bounded G&M (2008)
  adjustment fires before the M&G rates, shifting the local ``q^v``,
  ``q^{cl}``, and ``T`` (and thus ``q^{v+l}(T)``) so that
  ``q^v - q^{v+l}`` matches the advected ``s^{sat}``. The M&G semi-analytic
  rates then run on this post-G&M state — the "diagnostic supersaturation"
  they see is ``q^v_{\text{post-GM}} - q^{v+l}(T_{\text{post-GM}})``, not the
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
- [MilbrandtEtAl2021](@citet): Original three-moment ice in JAS (not implemented).
- [MilbrandtEtAl2025liquidfraction](@citet): Predicted liquid fraction on ice.

### Related Papers

- [MilbrandtYau2005](@citet): Multimoment microphysics and spectral shape parameter.
- [SeifertBeheng2006](@citet): Two-moment cloud microphysics for mixed-phase clouds.
- [KhairoutdinovKogan2000](@citet): Warm rain autoconversion parameterization.
- [pruppacher2010microphysics](@citet): Microphysics of clouds and precipitation (textbook).
