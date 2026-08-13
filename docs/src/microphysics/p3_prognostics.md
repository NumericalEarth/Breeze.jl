# [Prognostic Variables and Tendencies](@id p3_prognostics)

P3 tracks eight prognostic densities by default,

```math
ρq^{cl}, \quad ρq^r, \quad ρn^r, \quad ρq^i, \quad ρn^i, \quad ρq^f, \quad ρb^f, \quad ρq^{wi},
```

that is, the cloud liquid mass, the rain mass and number, the dry ice mass and
number, the rime mass and rime volume, and the liquid coating on ice — alongside
the host's vapor density ``ρq^v``. Three more appear when the corresponding option
is enabled: the cloud droplet number ``ρn^{cl}`` and unactivated aerosol number
``ρn^a`` with aerosol activation, and the supersaturation ``ρs^{v+l}`` with predicted
supersaturation. Together they describe the complete microphysical state.

This section documents each variable, its physical meaning, and the source-term
assembly used in `tendency_ρ*` (`prognostic_tendencies.jl`) to build the
microphysical tendency for each prognostic field. The rates those functions consume
are assembled by `compute_p3_process_rates` in `process_rates.jl`. The optional
groups are gated on a type, so a configuration that does not use one neither
allocates nor advects it.

The prognostic variable formulation has evolved through the P3 papers:

- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Original 4 ice variables.
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Added ``ρq^{wi}`` for liquid fraction.

Our implementation follows P3 v5.5, carrying five ice prognostics. Sign
convention used throughout the per-field tendencies: rate functions return
*positive magnitudes*, and the tendency assembly takes ``\text{gain} - \text{loss}``.
Bidirectional rates (condensation, deposition) keep their natural sign and
appear as gains; their negative branches contribute as losses elsewhere.

!!! note "Convention: prognostic ``ρq^i`` is dry ice"
    In Breeze the prognostic ice-mass density ``ρq^i`` stores **dry ice only**
    (rime + deposited mass; excludes ``ρq^{wi}``). The total ice mass, used
    wherever a lookup table is indexed by particle mass, is the sum
    ``ρq^i + ρq^{wi}``. Formulations that carry the total as the prognostic and
    recover the dry mass by subtraction are equivalent.

## Variable Definitions

### Cloud Liquid and Aerosol

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^{cl}`` | Cloud liquid mass density | kg/m³ | Mass of cloud droplets per unit volume |
| ``ρn^{cl}`` | Cloud droplet number density | m⁻³ | Number of cloud droplets per unit volume |
| ``ρn^a`` | Unactivated aerosol number density | m⁻³ | Aerosol not yet activated into droplets |

``ρn^{cl}`` and ``ρn^a`` are prognostic only when the optional aerosol-activation path
(`AerosolActivation` in `aerosol_activation.jl`) is enabled, where CCN-activation source
terms drive them. Otherwise droplet number is
the scheme parameter `cloud.number_concentration` (typical continental ``\sim 100`` cm⁻³
or marine ``\sim 50`` cm⁻³): every rate reads that constant, and neither field is
allocated or advected.

### Rain

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^r`` | Rain mass density | kg/m³ | Mass of raindrops per unit volume |
| ``ρn^r`` | Rain number density | m⁻³ | Number of raindrops per unit volume |

Rain follows a gamma size distribution with parameters diagnosed from the
mass / number ratio. Breeze runs with ``μ^r = 0``.

### Ice

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^i`` | Dry ice mass density | kg/m³ | Rime + deposited ice mass (excludes ``ρq^{wi}``) |
| ``ρn^i`` | Ice number density | m⁻³ | Number of ice particles |
| ``ρq^f`` | Rime mass density | kg/m³ | Mass of rime (frost) on ice |
| ``ρb^f`` | Rime volume density | m³/m³ | Volume of rime per unit volume |
| ``ρq^{wi}`` | Water on ice | kg/m³ | Liquid water coating ice particles |

### Vapor and Saturation Diagnostic

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^v`` | Water vapor density | kg/m³ | The host-coupled moisture variable |
| ``ρs^{v+l}`` | Supersaturation density, ``s^{v+l} = q^v - q^{v+l}`` | kg/m³ | Predicted-supersaturation path, controlled by the `predict_supersaturation` flag, which defaults to `false`. When `false`, the field is not allocated and is absent from `prognostic_field_names`; diagnostics use ``q^v - q^{v+l}(T)`` directly. When `true`, the bounded G&M (2008) adjustment is active. |

## Derived Quantities

From the prognostic variables, key diagnostic properties are computed:

**Rime fraction** (mass fraction of rime, *of dry ice*):

```math
F^f = \frac{ρq^f}{ρq^i}.
```

The denominator is the prognostic dry-ice mass, which excludes the liquid coating.

**Rime density**:

```math
ρ^f = \frac{ρq^f}{ρb^f}.
```

**Liquid fraction** (mass fraction of liquid coating, *of total ice mass*):

```math
F^l = \frac{ρq^{wi}}{ρq^i + ρq^{wi}}.
```

The denominator is the total ice mass, dry ice plus liquid coating.

**Mean particle mass** (per total ice mass):

```math
\bar{m} = \frac{ρq^i + ρq^{wi}}{ρn^i}.
```

## Tendency Equations

Each prognostic variable evolves according to:

```math
\frac{\partial (ρX)}{\partial t}
= - ∇ \cdot (\boldsymbol{u}\, ρX)
  - ∇ \cdot \boldsymbol{J}_X
  - \frac{\partial \mathcal{F}_X}{\partial z}
  + G_{ρX},
```

with, in order: advection by the resolved flow, subgrid turbulent transport
through the closure flux ``\boldsymbol{J}_X``, sedimentation of the flux
``\mathcal{F}_X`` at the field-specific fall speed, and the microphysical source
term ``G_{ρX}``. Only the last is P3's responsibility, and it is what the rest of
this section assembles.

The rate symbols used below are collected in [P3 notation](@ref p3_notation): a
dot marks a process rate per unit mass of air, the subscript names the process,
and a superscript names the species it acts on when a process acts on more than
one. The assembly mirrors the per-field ``\rho \cdot (\text{gain} - \text{loss})``
calls in `prognostic_tendencies.jl`.

### Cloud Liquid Tendency

```math
G_{ρq^{cl}}
=\rho\big[\dot{q}^{cl}_\text{cond} + \dot{q}_\text{act}
- \dot{q}_\text{aut} - \dot{q}_\text{acc} - \dot{q}^{cl}_\text{rim}
- \dot{q}^{cl}_\text{frz} - \dot{q}^{cl}_\text{hom}
- \dot{q}^{cl}_\text{col} - \dot{q}^{cl}_\text{wet} - \dot{q}_\text{wsh} \big].
```

| Term | Meaning |
|------|--------|
| ``\dot{q}^{cl}_\text{cond}`` | Condensation (positive) / evaporation (negative) — bidirectional. Includes the G&M alignment when `predict_supersaturation = true`. |
| ``\dot{q}_\text{act}`` | CCN-activation mass source (when prognostic ``N^{cl}`` enabled). |
| ``\dot{q}_\text{aut}`` | Autoconversion to rain. |
| ``\dot{q}_\text{acc}`` | Accretion by rain. |
| ``\dot{q}^{cl}_\text{rim}`` | Cloud riming by ice. |
| ``\dot{q}^{cl}_\text{frz}`` | Immersion freezing of cloud droplets. |
| ``\dot{q}^{cl}_\text{hom}`` | Homogeneous freezing (``T < -40°``C). |
| ``\dot{q}^{cl}_\text{col}`` | Cloud collection by ice above ``T_0`` (routes to ``q^{wi}`` or shedding). |
| ``\dot{q}^{cl}_\text{wet}`` | Wet-growth re-routing of cloud collection into ``q^{wi}`` (liquid-fraction branch). |
| ``\dot{q}_\text{wsh}`` | Wet-growth excess cloud collection shed to rain (non-liquid-fraction branch). |

``\dot{q}^{cl}_\text{wet}`` and ``\dot{q}_\text{wsh}`` are mutually exclusive: exactly one of the
two branches is active for a given `liquid_fraction_active` setting.

### Cloud Number Tendency

Only assembled when aerosol activation is enabled; in the prescribed-``N^{cl}`` path
``ρn^{cl}`` does not exist.

```math
G_{ρn^{cl}}
=\rho\big[\dot{n}_\text{act}
- \dot{n}^{cl}_\text{aut} - \tfrac{n^{cl}}{q^{cl}}\,\dot{q}_\text{acc} - \dot{n}^{cl}_\text{slf}
- \dot{n}^{cl}_\text{rim} - \dot{n}^{cl}_\text{frz} - \dot{n}^{cl}_\text{hom}
- \dot{n}^{cl}_\text{col} + \dot{n}^{cl}_\text{corr}\big].
```

- ``\dot{n}^{cl}_\text{aut}`` is scheme-aware: KK2000 scales by the in-cloud
  ``n^{cl}/q^{cl}`` ratio.
- ``\dot{n}^{cl}_\text{slf}`` is cloud self-collection, zero for KK2000.
- ``\dot{n}^{cl}_\text{corr}`` is the cloud-DSD ``λ``-bound number correction,
  applied as a relaxation over `sink_limiting_timescale` rather than as an
  instantaneous write-back.

### Rain Mass Tendency

```math
G_{ρq^r}
=\rho\big[\dot{q}_\text{aut} + \dot{q}_\text{acc} + \dot{q}^{r}_\text{cond} + \dot{q}_{\text{mlt},f}
+ \dot{q}_\text{shed} + \dot{q}_\text{wsh} + \dot{q}^{cl}_\text{col}\,\big[\text{no } F^l\big]
- \dot{q}^{r}_\text{evap} - \dot{q}^{r}_\text{rim} - \dot{q}^{r}_\text{frz} - \dot{q}^{r}_\text{hom}
- \dot{q}^{r}_\text{col} - \dot{q}^{r}_\text{wet} \big].
```

| Term | Meaning |
|------|--------|
| ``\dot{q}^{r}_\text{cond}`` | Coupled rain condensation (vapor → rain). |
| ``\dot{q}_{\text{mlt},f}`` | "Complete" melting flux from ice → rain, including the whole-particle clips. |
| ``\dot{q}_\text{shed}`` | Liquid coating shed from ice. |
| ``\dot{q}_\text{wsh}`` | Wet-growth shedding of excess cloud collection. |
| ``\dot{q}^{cl}_\text{col}`` | Above-freezing collected cloud, shed straight back to rain — only when liquid fraction is *off*. |
| ``\dot{q}^{r}_\text{evap}`` | Rain evaporation. |
| ``\dot{q}^{r}_\text{rim}`` | Rain riming by ice. |
| ``\dot{q}^{r}_\text{frz}`` | Immersion freezing of rain. |
| ``\dot{q}^{r}_\text{hom}`` | Homogeneous freezing of rain. |
| ``\dot{q}^{r}_\text{col}`` | Rain collection by ice above ``T_0``, zeroed at rate-assembly time unless liquid fraction is on. |
| ``\dot{q}^{r}_\text{wet}`` | Wet-growth re-routing of rain collection into ``q^{wi}``. |

### Rain Number Tendency

```math
G_{ρn^r}
=\rho\big[\dot{n}^{r}_\text{aut} + \dot{n}_\text{mlt} + \dot{n}^{r}_\text{brk}
+ \dot{n}_\text{shed} + \dot{n}^{cl}_\text{col}\, \big[\text{no } F^l\big]
+ \dot{n}_\text{wsh}
- \dot{n}^{r}_\text{evap} - \dot{n}^{r}_\text{slf} - \dot{n}^{r}_\text{rim}
- \dot{n}^{r}_\text{frz} - \dot{n}^{r}_\text{hom} - \dot{n}^{r}_\text{col}
+ \dot{n}^{r}_\text{corr}\big].
```

- ``\dot{n}^{r}_\text{aut} = \dot{q}_\text{aut} / m_\text{seed}``, with the seed-drop mass set by
  `warm_rain_scheme`: a 25 μm-radius drop for KK2000 (`initial_rain_drop_mass`).
- ``\dot{n}_\text{mlt}`` is the number companion the process operator budgets alongside
  ``\dot{q}_{\text{mlt},f}``. It is carried explicitly rather than recomputed as
  ``(n^i/q^i)\,\dot{q}_{\text{mlt},f}``, because a whole-particle clip transfers the
  remaining population even when the dry-ice mass has already gone to zero.
- ``\dot{n}^{r}_\text{slf}`` and ``\dot{n}^{r}_\text{brk}`` are the *netted* self-collection / breakup pair:
  physically one signed rate, so Breeze collapses the two directions before
  the number limiter runs and at most one of them is nonzero.
- ``\dot{n}_\text{shed} = \dot{q}_\text{shed} / m_{\text{shed},F^l}``, where
  ``m_{\text{shed},F^l}`` is `shed_drop_mass_liqfrac`.
- ``\dot{n}^{cl}_\text{col} = \dot{q}^{cl}_\text{col} / m_\text{shed}``, where
  ``m_\text{shed}`` is `shed_drop_mass`, and only when liquid fraction is *off*.
  Both masses default to a 1 mm drop, ``π/6\, ρ^L D^3 ≈ 5.24 \times 10^{-7}`` kg.
- ``\dot{n}^{r}_\text{evap}`` is the evaporation number sink the rain-number limiter budgeted
  (formed from the DSD-bounded ``n^r`` and rescaled by the same factor as the other
  rain-number sinks), not a fresh ``(n^r/q^r)\,\dot{q}^{r}_\text{evap}`` product.
- ``\dot{n}^{r}_\text{corr}`` is the diagnosed PSD ``λ``-bound number correction
  (the rain PSD diagnosis produces a clipped ``n^r``; Breeze adds a matching
  relaxation tendency rather than mutating the prognostic state).

### Ice Mass Tendency

```math
G_{ρq^i}
=\rho\big[\dot{q}_\text{dep} + \dot{q}^{cl}_\text{rim} + \dot{q}^{r}_\text{rim} + \dot{q}_\text{refr}
+ \dot{q}_\text{nuc} + \dot{q}^{cl}_\text{frz} + \dot{q}^{r}_\text{frz} + \dot{q}^{cl}_\text{hom} + \dot{q}^{r}_\text{hom}
- \dot{q}_{\text{mlt},p} - \dot{q}_{\text{mlt},f}\big].
```

Splintering mass does *not* appear separately in the ice mass tendency:
splinters are fragments of rime the particle already collected, and Breeze
carries the *full* (unreduced) riming rates, so the splintered mass is already
inside ``\dot{q}^{cl}_\text{rim} + \dot{q}^{r}_\text{rim}``. Adding it again would double count.
Wet growth also contributes nothing here in either branch: with liquid fraction
active the collected mass raises total ice and ``q^{wi}`` by equal amounts,
leaving the dry-ice mass unchanged, and without it the retained collection
already arrives through the reduced riming rates. The deposition term is
bidirectional; sublimation is its negative branch.

### Ice Number Tendency

```math
G_{ρn^i}
=\rho\big[\dot{n}_\text{nuc} + \dot{n}^{cl}_\text{frz} + \dot{n}^{r}_\text{frz} + \dot{n}^{cl}_\text{hom} + \dot{n}^{r}_\text{hom}
+ \dot{n}_\text{HM}
- \dot{n}_\text{mlt} - \dot{n}_\text{sub} - \dot{n}_\text{agg} - \dot{n}_\text{cap}
+ \dot{n}^{i}_\text{corr}\big].
```

- ``\dot{n}_\text{HM}`` is the Hallett–Mossop number source.
- ``\dot{n}_\text{sub}`` is the sublimation number sink, plus the number companion of
  liquid-coating evaporation.
- ``\dot{n}_\text{agg}`` is the aggregation magnitude.
- ``\dot{n}_\text{cap}`` is the soft-relaxation analog of a hard global ice-number
  cap. When ``n^i`` exceeds ``N^i_\text{max}/ρ``, a relaxation sink over
  `sink_limiting_timescale` is added to push it back toward the cap.
- ``\dot{n}^{i}_\text{corr}`` is the ice ``λ``-limiter correction: ``n^i`` is bounded
  against the tabulated mean-size limits, and Breeze adds the difference between
  the bounded and the globally capped number as a relaxation tendency. It is
  suppressed when a whole-particle clip fires, since that path drains the raw
  population directly.

The three number sinks are additionally projected onto the population that
actually exists: melting takes priority, then sublimation, then number-only
aggregation is limited to whatever remains.

### Rime Mass Tendency

```math
G_{ρq^f}
=\rho\big[\dot{q}^{cl}_\text{rim} + \dot{q}^{r}_\text{rim} + \dot{q}_\text{refr}
+ \dot{q}^{cl}_\text{frz} + \dot{q}^{r}_\text{frz} + \dot{q}^{cl}_\text{hom} + \dot{q}^{r}_\text{hom}
+ \dot{q}_\text{wdn}
- F^f\,(\dot{q}_{\text{mlt},p} + \dot{q}_{\text{mlt},f,\text{ord}} + \dot{q}_\text{sub})
- \dot{q}^f_\text{clip}\big].
```

``\dot{q}_\text{wdn}`` is the wet-growth densification mass term: when wet-growth
shedding fires (without active liquid fraction), the rime is set to its
maximum density. ``\dot{q}_\text{sub}`` is the sublimation mass magnitude
(``\dot{q}_\text{sub} = \max(0, -\dot{q}_\text{dep})``).

Ordinary melting removes the beginning-of-stage rime fraction ``F^f``, but a
*whole-particle clip* must remove the rime exactly, including any change the
same step made to it. So the melting term uses only the ordinary part,

```math
\dot{q}_{\text{mlt},f,\text{ord}} = \max\!\big(0,\; \dot{q}_{\text{mlt},f} - \dot{q}^i_\text{clip}\big),
```

and the clipped particles' rime is drained through the explicitly reconstructed
companion ``\dot{q}^f_\text{clip}``.

### Rime Volume Tendency

```math
G_{ρb^f}
=\rho\!\Bigg[\frac{\dot{q}^{cl}_\text{rim}}{ρ^f_\text{new}}
+ \frac{\dot{q}^{r}_\text{rim} + \dot{q}_\text{refr} + \dot{q}^{cl}_\text{frz} + \dot{q}^{r}_\text{frz} + \dot{q}^{cl}_\text{hom} + \dot{q}^{r}_\text{hom}}{ρ^f_\text{max}}
+ \dot{b}_\text{wdn}
- \frac{F^f\,(\dot{q}_{\text{mlt},p} + \dot{q}_{\text{mlt},f,\text{ord}} + \dot{q}_\text{sub})}{ρ^f}
- \dot{b}_\text{clip}
- \dot{b}_\text{dens}\Bigg].
```

The rime-density denominators differ by process:
fresh cloud rime uses the Cober–List density ``ρ^f_\text{new}``; rain
riming, refreezing, immersion freezing, and homogeneous freezing all
deposit at the maximum rime density ``ρ^f_\text{max} = 900`` kg/m³. ``\dot{b}_\text{clip}``
is the whole-particle volume companion, and ``\dot{b}_\text{dens}``
is the melt-densification correction that drives the remaining rime
toward the configured solid-ice density (`pure_ice_density`, 917 kg/m³ by
default) when ``ρ^f`` is below that density and liquid fraction is *not* active.

### Liquid on Ice Tendency

```math
G_{ρq^{wi}}
=\rho\big[\dot{q}_{\text{mlt},p} + \dot{q}^{cl}_\text{col} + \dot{q}^{r}_\text{col}
+ \dot{q}^{cl}_\text{wet} + \dot{q}^{r}_\text{wet} + \dot{q}^{wi}_\text{cond}
- \dot{q}_\text{shed} - \dot{q}_\text{refr} - \dot{q}^{wi}_\text{evap}\big],
```

valid in the active liquid-fraction branch. Above-freezing
collection of cloud and rain feeds the liquid coating; when wet growth is
diagnosed below freezing, ``\dot{q}^{cl}_\text{wet}`` and ``\dot{q}^{r}_\text{wet}`` route the collected
mass there too. ``\dot{q}_\text{wsh}`` does not appear in this budget: it is
nonzero only without liquid fraction and routes excess cloud water directly to
rain. ``\dot{q}^{wi}_\text{cond}`` and ``\dot{q}^{wi}_\text{evap}``
are the coupled liquid-coated-ice condensation / evaporation rates, which are
active when ``F^l \ge`` `liquid_fraction_clipping_threshold` (0.01) — below it,
the dry-ice deposition branch runs instead. With liquid fraction *off*, the
ordinary liquid-fraction terms are zero and any leftover coating is drained to
rain through ``\dot{q}_\text{shed}`` over `sink_limiting_timescale`.

### Vapor and Aerosol Tendencies

```math
G_{ρq^v}
=\rho\big[\dot{q}^{r}_\text{evap} + \dot{q}^{wi}_\text{evap}
- \dot{q}^{cl}_\text{cond} - \dot{q}_\text{dep} - \dot{q}_\text{nuc} - \dot{q}_\text{act} - \dot{q}^{r}_\text{cond} - \dot{q}^{wi}_\text{cond}\big],
```

with the bidirectional ``\dot{q}^{cl}_\text{cond}`` and ``\dot{q}_\text{dep}`` supplying their own
evaporation / sublimation branches through their negative values.

```math
G_{ρn^a} = -\rho\,\dot{n}_\text{act},
```

one aerosol removed per activated droplet; zero in the prescribed-``N^{cl}`` path.

## Sedimentation

Each quantity sediments at its characteristic velocity. The velocities are
diagnosed once per RK stage by `prepare_microphysical_tendencies!` into z-Face
fields, because the scalar flux divergence consumes them as advecting velocities
at ``(\text{Center}, \text{Center}, \text{Face})``.

| Variable | Sedimentation Velocity | Flux |
|----------|----------------------|------|
| ``ρq^{cl}`` | ``V_m^{cl}`` (mass-weighted Stokes) | ``\mathcal{F}_{ρq^{cl}} = -V_m^{cl} ρq^{cl}`` |
| ``ρn^{cl}`` | ``V_n^{cl}`` (number-weighted Stokes) | ``\mathcal{F}_{ρn^{cl}} = -V_n^{cl} ρn^{cl}`` |
| ``ρq^r`` | ``V_m^r`` | ``\mathcal{F}_{ρq^r} = -V_m^r ρq^r`` |
| ``ρn^r`` | ``V_n^r`` | ``\mathcal{F}_{ρn^r} = -V_n^r ρn^r`` |
| ``ρq^i`` | ``V_m^i`` | ``\mathcal{F}_{ρq^i} = -V_m^i ρq^i`` |
| ``ρn^i`` | ``V_n^i`` | ``\mathcal{F}_{ρn^i} = -V_n^i ρn^i`` |
| ``ρq^f`` | ``V_m^i`` | ``\mathcal{F}_{ρq^f} = -V_m^i ρq^f`` |
| ``ρb^f`` | ``V_m^i`` | ``\mathcal{F}_{ρb^f} = -V_m^i ρb^f`` |
| ``ρq^{wi}`` | ``V_m^i`` | ``\mathcal{F}_{ρq^{wi}} = -V_m^i ρq^{wi}`` |

``ρs^{v+l}`` and ``ρn^a`` do not sediment. Cloud droplets do: cloud mass and
number settle with DSD-integrated Stokes velocities.

The sedimentation tendency is

```math
\frac{\partial ρX}{\partial t}\bigg|_\text{sed} = -\frac{\partial \mathcal{F}_X}{\partial z}.
```

At the bottom face, `precipitation_boundary_condition = nothing` (the default)
keeps the diagnosed fall speed, so precipitation leaves the domain through an
open surface; an `ImpenetrableBoundaryCondition()` zeroes it instead, so
precipitation accumulates in the lowest cell. The top face is held at zero, so
nothing sediments in from above the model top.

Breeze does not subcycle sedimentation inside P3; Oceananigans is responsible
for stability in transport.

## Coupling to AtmosphereModel

In Breeze, P3 microphysics couples to `AtmosphereModel` through the
microphysics interfaces implemented in `p3_microphysical_state.jl` and
`p3_driver.jl`. The default configuration uses prescribed cloud droplet number,
two-moment ice, and diagnostic supersaturation:

```jldoctest
using Breeze

microphysics = PredictedParticlePropertiesMicrophysics()
prognostic_field_names(microphysics)

# output
(:ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
```

``ρnᶜˡ`` and ``ρnᵃ`` appear only when `aerosol`
is an `AerosolActivation`: the default prescribed-Nᶜˡ path takes droplet number
from `cloud.number_concentration`, so neither field is allocated or advected there. ``ρsᵛ⁺ˡ`` appears only when
`predict_supersaturation = true`.

P3's aerosol distribution is specified **per unit mass of air**: `AerosolMode.number_mixing_ratio`
is in kg⁻¹, and so are the activated numbers it produces and the ``n^{cl}`` and ``n^a`` that the
activation cap compares them against. The prognostic reservoir ``ρn^a`` therefore holds the
``ρ``-weighted count in m⁻³.

Nothing needs to be initialized by hand. `AtmosphereModel` construction and every `set!` write
``ρn^a`` from [`initial_aerosol_number_density`](@ref), which for P3 is the air density times
`AerosolMode.number_mixing_ratio` summed over all modes, so a multi-mode population is seeded from
its own parameters. Because that weighting needs a density, the value is written against whichever
density is established at the time: the reference density for anelastic dynamics, a prescribed
density for the kinematic driver, the reconciled total density for compressible dynamics. Only
compressible dynamics has no density at construction, so there the reservoir stays zero until the
first `set!` carrying `ρ`, `ρᵈ`, or a `HydrostaticallyBalancedDensity`. The two-moment scheme's
`CloudMicrophysics` modes are volumetric to begin with, so the generic density-aware hook forwards
their concentration without applying the density argument.

Pass `nᵃ` [kg⁻¹] or `ρnᵃ` [m⁻³] to `set!` to choose the value instead. That is also how a partly
depleted reservoir survives a `set!`, since an unqualified `set!` rewrites it to the default.

Host-facing entry points:

1. **`prepare_microphysical_tendencies!`**: Diagnoses the terminal velocities for the
   current RK stage into the z-Face fall-speed fields, then refreshes their halos.
2. **`compute_microphysical_tendencies!`**: Fills the per-field process-rate cache from
   the current state in one kernel and adds it to ``G^n`` in a second. Optional
   prognostic groups get their own kernel, launched only when that group exists.
3. **`microphysical_tendency`**: The gridless per-field fallback used by `ParcelModels`.
4. **`moisture_fractions`**: Converts prognostic densities to mass fractions
   (liquid = cloud + rain + liquid-on-ice; ice = dry ice).
5. **`update_microphysical_fields!`**: Refreshes diagnostic fields after a state update.
6. **`negative_moisture_correction`**: The repair applied at the top of `update_state!`
   (see *Positivity* below).

The tendency-only architecture is described in
[Architectural choice: Breeze P3 updates tendencies, instead of prognostic variables](@ref p3_overview).

## Conservation Properties

P3 conserves total water in a closed system:

```math
\frac{d}{dt}\left( q^v + q^{cl} + q^r + q^i + q^{wi} \right) = 0.
```

(The liquid coating ``q^{wi}`` is included because shedding moves it to rain,
and refreezing converts it to rime — both internal to the ice mass.)

Within P3 the limiting happens in two stages. First
`limit_vapor_rates` (`process_rate_helpers.jl`) applies the
saturation-adjustment caps (see the saturation adjustment limits in
[Microphysical Processes](@ref p3_processes)). Then the per-species conservation
budgets run: for each donor reservoir, `sink_limiting_factor` compares the total
sink against what is available over `sink_limiting_timescale` and rescales every
sink for that species proportionally. Because rain, dry ice, total ice, and
coating water exchange mass with one another, one sequential pass can credit a
source that a later donor limiter then reduces, so those four budgets are
re-projected `coupled_sink_limiting_iterations` times (default 4). Every
projection only reduces rates, so the loop converges monotonically while staying
allocation-free and GPU-safe.

Energy conservation is delegated to the host: the Anelastic and
compressible formulations carry latent heating implicitly through their
prognostic thermodynamic variable. P3 itself does not assemble a ``θ``
tendency.

## Numerical Considerations

### Positivity

The saturation-adjustment caps and per-species sink-limiting factors bound P3
sinks against the donor reservoirs available over `sink_limiting_timescale`.
Thus, for a single forward update no longer than that interval, limited P3 sinks
alone cannot make the corresponding mass or number reservoirs negative. This is
not an unconditional positivity guarantee for an arbitrary host timestep or RK
stage. The optional ``ρs^{v+l}`` prognostic is also excluded: subsaturation is
legitimately negative. Breeze does not implement a post-step "return small mass
to vapor" cleanup, because that requires state mutation with a paired ``θ``
correction.

The advection operator is a separate matter: it is not positive-definite, so a
stage update can return any density negative. `AtmosphereModel` therefore applies
P3's `negative_moisture_correction` at the top of `update_state!`, before the
rates see the state. The default `SpeciesBorrowing`:

- borrows along the chain
  ``ρq^{wi} \leftarrow ρq^i \leftarrow ρq^r \leftarrow ρq^{cl} \leftarrow ρq^v``,
  so a negative coating deficit is covered
  by the ice mass carrying it (implied refreezing), negative ice by rain
  (implied freezing), and the warm-phase tail as in the 1- and 2-moment schemes.
  Borrowing searches the whole lighter-species tail, so an empty immediate donor
  does not block a deficit from reaching water further down;
- zeroes the ice-population fields (``ρn^i``, ``ρq^f``, ``ρb^f``) orphaned by a
  vanished ``ρq^i``, and ``ρn^r`` / ``ρn^{cl}``
  orphaned by their masses. ``ρq^{wi}`` is deliberately *not* paired with
  ``ρq^i``: it is real water, and the whole-particle clip already sheds it to
  rain when the dry ice is gone;
- clamps the remaining non-water fields (number moments, rime properties,
  aerosol count). ``ρs^{v+l}`` is excluded, since subsaturation is
  legitimately negative.

Passing `SpeciesBorrowing(vertical_borrowing = VerticalBorrowing())` additionally
redistributes leftover vapor deficits within each column;
`negative_moisture_correction = nothing` disables the repair, in which case the
process rates still `clamp_positive` what they read but the prognostic fields keep
their negative mass.

### Consistency

The rime fraction must satisfy ``0 \le F^f \le 1`` (so ``ρq^f \le ρq^i``) and
the liquid fraction ``0 \le F^l \le 1``. `consistent_rime_state` caps the
diagnosed fractions at read time, and `p3_ice_properties`
(`p3_microphysical_state.jl`) carries the capped values into every rate so the
whole step sees one consistent state.

### Threshold Handling

Small values below numerical thresholds are treated as zero in the source
assembly:

```jldoctest thresholds
using Breeze

microphysics = PredictedParticlePropertiesMicrophysics()

(microphysics.minimum_mass_mixing_ratio,     # [kg/kg]
 microphysics.minimum_number_mixing_ratio)   # [1/kg]

# output
(1.0e-14, 1.0e-16)
```

Two further thresholds on `ProcessRateParameters` control whole-particle
handling — `liquid_fraction_clipping_threshold` [-] and
`tiny_ice_to_rain_threshold` [kg/kg]:

```jldoctest thresholds
parameters = microphysics.process_rates

(parameters.liquid_fraction_clipping_threshold,
 parameters.tiny_ice_to_rain_threshold)

# output
(0.01, 1.0e-12)
```

``F^l`` below `liquid_fraction_clipping_threshold` freezes the residual coating to
rime; above ``1 -`` that value (or above 0.99) the particle is transferred whole to
rain, as is warm ice with total mass below `tiny_ice_to_rain_threshold`. Both are
implemented as relaxation drains over `refreezing_timescale`, and both suppress
every process that would need the clipped particle.

## Code Example

```@example p3_prognostics
using Breeze

p3 = PredictedParticlePropertiesMicrophysics()

# Get all prognostic field names
names = prognostic_field_names(p3)
println("Prognostic fields:")
for name in names
    println("  ", name)
end
```

```@example p3_prognostics
# Access thresholds
println("\nNumerical thresholds:")
println("  Minimum mass mixing ratio: ", p3.minimum_mass_mixing_ratio, " kg/kg")
println("  Minimum number mixing ratio: ", p3.minimum_number_mixing_ratio, " 1/kg")
```

## References for This Section

- [Morrison2015parameterization](@cite): Original prognostic variables and tendencies (Section 2).
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid fraction prognostic (``ρq^{wi}``).
- [MilbrandtYau2005](@cite): Multi-moment microphysics and sedimentation.
