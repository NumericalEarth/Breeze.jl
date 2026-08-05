# [Prognostic Variables and Tendencies](@id p3_prognostics)

P3 tracks 8 prognostic densities by default, and up to 12 with every option enabled;
together they describe the complete microphysical state of the atmosphere. This section
documents each variable, its physical meaning, and the source-term assembly used in
`tendency_ρ*` (`prognostic_tendencies.jl`, plus `sixth_moment_tendencies.jl` for
``ρ\tilde z^i``) to build the microphysical tendency for each prognostic field. The rates
those functions consume are assembled by `compute_p3_process_rates` in
`process_rates.jl`. Optional groups (``ρ\tilde z^i``, ``ρs^{sat}``, ``ρn^{cl}``/``ρn^a``)
are gated on a type, so a configuration that does not use one neither allocates nor
advects it.

The prognostic variable formulation has evolved through the P3 papers:

- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Original 4 ice variables.
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Added the transformed
  sixth-moment prognostic ``ρ\tilde z^i`` for 3-moment ice, where
  ``\tilde z^i = \sqrt{z^i n^i}``.
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Added ``ρq^{wi}`` for liquid fraction.

Our implementation follows P3 v5.5, carrying five ice prognostics by default and
the sixth (``ρ\tilde z^i``) when three-moment ice is enabled. Sign
convention used throughout the per-field tendencies: rate functions return
*positive magnitudes*, and the tendency assembly takes ``\text{gain} - \text{loss}``.
Bidirectional rates (condensation, deposition) keep their natural sign and
appear as gains; their negative branches contribute as losses elsewhere.

!!! note "Convention: prognostic ``ρq^i`` is dry ice"
    In Breeze the prognostic ice-mass density ``ρq^i`` stores **dry ice only**
    (rime + deposited mass; excludes ``ρq^{wi}``). The Fortran reference uses
    the opposite convention: `qitot` is the *total* (ice + liquid coating)
    and the dry-ice mass is recovered as `qitot - qiliq`. The two formulations
    are equivalent — Breeze's ``ρq^i + ρq^{wi}`` equals Fortran's `qitot`.

## Variable Definitions

### Cloud Liquid and Aerosol

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^{cl}`` | Cloud liquid mass density | kg/m³ | Mass of cloud droplets per unit volume |
| ``ρn^{cl}`` | Cloud droplet number density | m⁻³ | Number of cloud droplets per unit volume |
| ``ρn^a`` | Unactivated aerosol number density | m⁻³ | Aerosol not yet activated into droplets |

``ρn^{cl}`` and ``ρn^a`` are prognostic only when the optional aerosol-activation path
(`AerosolActivation` in `aerosol_activation.jl`) is enabled, where CCN-activation source
terms drive them. Otherwise, matching Fortran `log_predictNc = .false.`, droplet number is
the scheme parameter `cloud.number_concentration` (typical continental ``\sim 100`` cm⁻³
or marine ``\sim 50`` cm⁻³): every rate reads that constant, and neither field is
allocated or advected.

### Rain

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^r`` | Rain mass density | kg/m³ | Mass of raindrops per unit volume |
| ``ρn^r`` | Rain number density | m⁻³ | Number of raindrops per unit volume |

Rain follows a gamma size distribution with parameters diagnosed from the
mass / number ratio. Both Fortran and Breeze run with ``μ_r = 0`` at runtime.

### Ice

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^i`` | Dry ice mass density | kg/m³ | Rime + deposited ice mass (excludes ``ρq^{wi}``) |
| ``ρn^i`` | Ice number density | m⁻³ | Number of ice particles |
| ``ρq^f`` | Rime mass density | kg/m³ | Mass of rime (frost) on ice |
| ``ρb^f`` | Rime volume density | m³/m³ | Volume of rime per unit volume |
| ``ρ\tilde z^i`` | Advected ice reflectivity variable | (kg m⁻³)(m³ kg⁻¹) | Square-root sixth moment ``ρ\sqrt{z^i n^i}``, from ``z^i`` [m⁶/kg] and ``n^i`` [kg⁻¹]; present only when `three_moment_ice = true` |
| ``ρq^{wi}`` | Water on ice | kg/m³ | Liquid water coating ice particles |

### Vapor and Saturation Diagnostic

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^v`` | Water vapor density | kg/m³ | The host-coupled moisture variable |
| ``ρs^{sat}`` | Predicted supersaturation | kg/m³ | Predicted-supersaturation path. Fortran v5.5 hard-codes `log_predictSsat = .false.`; Breeze's `predict_supersaturation` flag defaults to `false` to match. When `false`, the field is not allocated and is absent from `prognostic_field_names`; diagnostics use ``q^v - q^{v,s}(T)`` directly. When `true`, the bounded G&M (2008) adjustment is active. |

## Derived Quantities

From the prognostic variables, key diagnostic properties are computed:

**Rime fraction** (mass fraction of rime, *of dry ice*):

```math
F^f = \frac{ρq^f}{ρq^i}.
```

The denominator is the prognostic dry-ice mass — equivalent to Fortran's
`qirim / (qitot - qiliq)`.

**Rime density**:

```math
ρ^f = \frac{ρq^f}{ρb^f}.
```

**Liquid fraction** (mass fraction of liquid coating, *of total ice mass*):

```math
F^l = \frac{ρq^{wi}}{ρq^i + ρq^{wi}}.
```

The denominator is the total ice mass — equivalent to Fortran's
`qiliq / qitot`.

**Mean particle mass** (per total ice mass):

```math
\bar{m} = \frac{ρq^i + ρq^{wi}}{ρn^i}.
```

## Tendency Equations

Each prognostic variable evolves according to:

```math
\frac{\partial (ρX)}{\partial t} = \text{ADV} + \text{TURB} + \text{SED} + \text{SRC},
```

where:

- **ADV**: Advection by resolved flow.
- **TURB**: Subgrid turbulent transport.
- **SED**: Sedimentation (gravitational settling at the field-specific tabulated velocity).
- **SRC**: Microphysical source/sink terms.

The microphysical source assembly below mirrors the per-field
``\rho \cdot (\text{gain} - \text{loss})`` calls in `prognostic_tendencies.jl`.

### Cloud Liquid Tendency

```math
\partial_t (ρq^{cl})\big|_\text{src}
=\rho\big[\text{COND} + \text{CCN}_q
- \text{AUTO} - \text{ACCR} - \text{RIM}_c
- \text{IMMF}_c - \text{HOM}_c
- \text{COL}_{c,\text{warm}} - \text{WG}_c - \text{WG}_\text{shed} \big].
```

| Term | Meaning |
|------|--------|
| COND | Condensation (positive) / evaporation (negative) — bidirectional. Includes the G&M alignment when `predict_supersaturation = true`. |
| CCN``_q`` | CCN-activation mass source (when prognostic ``N_c`` enabled). |
| AUTO | Autoconversion to rain. |
| ACCR | Accretion by rain. |
| RIM``_c`` | Cloud riming by ice. |
| IMMF``_c`` | Immersion freezing of cloud droplets. |
| HOM``_c`` | Homogeneous freezing (``T < -40°``C). |
| COL``_{c,\text{warm}}`` | Cloud collection by ice above ``T_0`` (routes to ``q^{wi}`` or shedding). |
| WG``_c`` | Wet-growth re-routing of cloud collection into ``q^{wi}`` (liquid-fraction branch). |
| WG``_\text{shed}`` | Wet-growth excess cloud collection shed to rain (non-liquid-fraction branch). |

``\text{WG}_c`` and ``\text{WG}_\text{shed}`` are mutually exclusive: exactly one of the
two branches is active for a given `liquid_fraction_active` setting.

### Cloud Number Tendency

Only assembled when aerosol activation is enabled; in the prescribed-``N_c`` path
``ρn^{cl}`` does not exist.

```math
\partial_t (ρn^{cl})\big|_\text{src}
=\rho\big[\text{CCN}_n
- \text{AUTO}_{c,n} - \tfrac{n^{cl}}{q^{cl}}\,\text{ACCR} - \text{SCOL}_c
- \text{RIM}_{c,n} - \text{IMMF}_{c,n} - \text{HOM}_{c,n}
- \text{COL}_{c,\text{warm},n} + \text{N-CORR}_{cl}\big].
```

- ``\text{AUTO}_{c,n}`` is scheme-aware: KK2000 and Kogan2013 scale by the in-cloud
  ``n^{cl}/q^{cl}`` ratio (Fortran `ncautc = qcaut × nc/qc`), SB2001 removes a
  fixed-mass drizzle drop per unit converted mass.
- ``\text{SCOL}_c`` is cloud self-collection, nonzero only for SB2001.
- ``\text{N-CORR}_{cl}`` is the cloud-DSD ``λ``-bound number correction
  (Fortran `get_cloud_dsd2` write-back), applied as a relaxation over
  `sink_limiting_timescale`.

### Rain Mass Tendency

```math
\partial_t (ρq^r)\big|_\text{src}
=\rho\big[\text{AUTO} + \text{ACCR} + \text{RAIN-COND} + \text{MELT}_\text{full}
+ \text{SHED} + \text{WG}_\text{shed} + \text{COL}_{c,\text{warm}}\,\big[\text{no } F^l\big]
- \text{REVP} - \text{RIM}_r - \text{IMMF}_r - \text{HOM}_r
- \text{COL}_{r,\text{warm}} - \text{WG}_r \big].
```

| Term | Meaning |
|------|--------|
| RAIN-COND | Coupled rain condensation (vapor → rain). |
| MELT``_\text{full}`` | "Complete" melting flux from ice → rain, including the whole-particle clips. |
| SHED | Liquid coating shed from ice. |
| WG``_\text{shed}`` | Wet-growth shedding of excess cloud collection. |
| COL``_{c,\text{warm}}`` | Above-freezing collected cloud, shed straight back to rain — only when liquid fraction is *off*. |
| REVP | Rain evaporation. |
| RIM``_r`` | Rain riming by ice. |
| IMMF``_r`` | Immersion freezing of rain. |
| HOM``_r`` | Homogeneous freezing of rain. |
| COL``_{r,\text{warm}}`` | Rain collection by ice above ``T_0``, zeroed at rate-assembly time unless liquid fraction is on. |
| WG``_r`` | Wet-growth re-routing of rain collection into ``q^{wi}``. |

### Rain Number Tendency

```math
\partial_t (ρn^r)\big|_\text{src}
=\rho\big[\text{AUTO}_n + \text{MELT}_n + \text{BR}
+ \text{SHED}_n + \text{COL}_{c,\text{warm},n}\, \big[\text{no } F^l\big]
+ \text{WG}_{\text{shed},n}
- \text{REVP}_n - \text{SCBK} - \text{RIM}_{r,n}
- \text{IMMF}_{r,n} - \text{HOM}_{r,n} - \text{COL}_{r,\text{warm},n}
+ \text{N-CORR}_r\big].
```

- ``\text{AUTO}_n = \text{AUTO} / m_\text{seed}``, with the seed-drop mass set by
  `warm_rain_scheme`: a 25 μm-radius drop for KK2000 (`initial_rain_drop_mass`),
  40 μm for Kogan2013, and ``2/7.6923\times10^{9}`` kg for SB2001.
- ``\text{MELT}_n`` is the number companion the process operator budgets alongside
  ``\text{MELT}_\text{full}``. It is carried explicitly rather than recomputed as
  ``(n^i/q^i)\,\text{MELT}_\text{full}``, because a whole-particle clip transfers the
  remaining population even when the dry-ice mass has already gone to zero.
- ``\text{SCBK}`` and ``\text{BR}`` are the *netted* self-collection / breakup pair:
  Fortran carries one signed `nrslf`, so Breeze collapses the two directions before
  the number limiter runs and at most one of them is nonzero.
- ``\text{SHED}_n = \text{SHED} / m_\text{shed}^{F^l}`` with
  ``1/m_\text{shed}^{F^l} = 1.928 \times 10^6`` kg⁻¹ (Fortran `nlshd`).
- ``\text{COL}_{c,\text{warm},n} = \text{COL}_{c,\text{warm}} / m_\text{shed}``
  with ``1/m_\text{shed} = 1.923 \times 10^6`` kg⁻¹ (Fortran `ncshdc`), and only when
  liquid fraction is *off*.
- ``\text{REVP}_n`` is the evaporation number sink the rain-number limiter budgeted
  (formed from the DSD-bounded ``n^r`` and rescaled by the same factor as the other
  rain-number sinks), not a fresh ``(n^r/q^r)\,\text{REVP}`` product.
- ``\text{N-CORR}_r`` is the diagnosed PSD ``λ``-bound number correction
  (Fortran `get_rain_dsd2` writes back a clipped ``n_r``; Breeze adds a
  matching relaxation tendency rather than mutating the prognostic state).

### Ice Mass Tendency

```math
\partial_t (ρq^i)\big|_\text{src}
=\rho\big[\text{DEP} + \text{RIM}_c + \text{RIM}_r + \text{REFR}
+ \text{NUC} + \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r
- \text{MELT}_\text{partial} - \text{MELT}_\text{full}\big].
```

Splintering mass does *not* appear separately in the ice mass tendency:
splinters are fragments of rime the particle already collected, and Breeze
carries the *full* (unreduced) riming rates, so the splintered mass is already
inside ``\text{RIM}_c + \text{RIM}_r``. Adding it again would double count.
Wet growth also contributes nothing here in either branch: with liquid fraction
active the collected mass raises total ice and ``q^{wi}`` by equal amounts,
leaving the dry-ice mass unchanged, and without it the retained collection
already arrives through the reduced riming rates. The deposition term is
bidirectional; sublimation is its negative branch.

### Ice Number Tendency

```math
\partial_t (ρn^i)\big|_\text{src}
=\rho\big[\text{NUC}_n + \text{IMMF}_{c,n} + \text{IMMF}_{r,n} + \text{HOM}_{c,n} + \text{HOM}_{r,n}
+ \text{HM}_n
- \text{MELT}_n - \text{SUB}_n - \text{AGG}_n - \text{NLIM}
+ \text{N-CORR}_i\big].
```

- ``\text{HM}_n`` is the Hallett–Mossop number source.
- ``\text{SUB}_n`` is the sublimation number sink, plus the number companion of
  liquid-coating evaporation (Fortran `nisub + nlevp`).
- ``\text{AGG}_n`` is the aggregation magnitude.
- ``\text{NLIM}`` is the soft-relaxation analog of Fortran's `impose_max_Ni`
  hard cap. When ``n^i`` exceeds ``N_{i,\max}/ρ``, a relaxation sink over
  `sink_limiting_timescale` is added to push it back toward the cap.
- ``\text{N-CORR}_i`` is the ice ``λ``-limiter correction: Fortran clamps `nitot`
  against the tabulated mean-size bounds (`f1pr09`/`f1pr10`), and Breeze adds
  the difference between the bounded and the globally capped number as a
  relaxation tendency. It is suppressed when a whole-particle clip fires, since
  that path drains the raw population directly.

The three number sinks are additionally projected onto the population that
actually exists: melting takes priority, then sublimation, then number-only
aggregation is limited to whatever remains.

### Rime Mass Tendency

```math
\partial_t (ρq^f)\big|_\text{src}
=\rho\big[\text{RIM}_c + \text{RIM}_r + \text{REFR}
+ \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r
+ \text{WG-DENS}_q
- F^f\,(\text{MELT}_\text{partial} + \text{MELT}_\text{full}^\text{ord} + \text{SUB})
- \text{CLIP}_q\big].
```

``\text{WG-DENS}_q`` is the wet-growth densification mass term: when wet-growth
shedding fires (without active liquid fraction), the rime is set to its
maximum density. ``\text{SUB}`` is the sublimation mass magnitude
(``\text{SUB} = \max(0, -\text{DEP})``).

Ordinary melting removes the beginning-of-stage rime fraction ``F^f``, but a
*whole-particle clip* must remove the rime exactly, including any change the
same step made to it. So the melting term uses only the ordinary part,

```math
\text{MELT}_\text{full}^\text{ord} = \max\!\big(0,\; \text{MELT}_\text{full} - \text{CLIP}_\text{dry}\big),
```

and the clipped particles' rime is drained through the explicitly reconstructed
companion ``\text{CLIP}_q``.

### Rime Volume Tendency

```math
\partial_t (ρb^f)\big|_\text{src}
=\rho\!\Bigg[\frac{\text{RIM}_c}{ρ^f_\text{new}}
+ \frac{\text{RIM}_r + \text{REFR} + \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r}{ρ_{r,\max}}
+ \text{WG-DENS}_b
- \frac{F^f\,(\text{MELT}_\text{partial} + \text{MELT}_\text{full}^\text{ord} + \text{SUB})}{ρ^f}
- \text{CLIP}_b
- \mathcal{D}\Bigg].
```

The various rime-density denominators reflect the Fortran convention:
fresh cloud rime uses the Cober–List density ``ρ^f_\text{new}``; rain
riming, refreezing, immersion freezing, and homogeneous freezing all
deposit at the maximum rime density ``ρ_{r,\max} = 900`` kg/m³. ``\text{CLIP}_b``
is the whole-particle volume companion, and ``\mathcal{D}``
is the melt-densification correction that drives the remaining rime
toward solid ice density (`pure_ice_density`, 917 kg/m³) when ``ρ^f < 917`` and
liquid fraction is *not* active.

### Reflectivity Tendency (3-moment)

At runtime the sixth moment follows Fortran's active hybrid path
(`active_ice_sixth_moment_tendency`; see the sixth-moment update in
[Microphysical Processes](@ref p3_processes)). The shape parameter ``μ_i`` is
held at its pre-process Table-3 value, the continuous "group 1" tendencies are
integrated over ``τ`` (`sink_limiting_timescale`) to obtain
``(q^i, q^{wi}, n^i, q^f, b^f)_\text{new}``, and the reflectivity is rebuilt from
that state:

```math
\partial_t (ρz^i_\text{physical})\big|_\text{src}
=\rho\Bigg[\frac{Z_\text{new} - \max(0, z^i)}{τ}
+ \sum_{p\in\text{group2}} G(μ_{\text{src},p})\,\frac{\dot{M}_{3,p}^2}{\dot{n}_p}\Bigg],
\qquad
Z_\text{new} = G(μ_i)\,\frac{M_{3,\text{new}}^2}{n^i_\text{new}},
```

with ``M_{3,\text{new}} = 6\, q^i_\text{total,new} / (π\, \bar{ρ}_i(μ_i))`` and
``\bar{ρ}_i(μ_i)`` the Table-1 bulk density read at the same fixed ``μ_i``. Every
group-1 tendency is formed by subtracting the group-2 sources from the
corresponding per-field tendency, so no process is counted twice. The group-2
sum runs over deposition nucleation, immersion freezing of cloud / rain, both
splintering branches, and homogeneous freezing of cloud / rain, with
``\dot{M}_3 = 6\,\dot{q}_\text{src}/(π\,ρ_i)`` at ``ρ_i = 900`` kg/m³. All group-2
sources use ``μ_\text{src} = μ_r = 0`` except homogeneous freezing of *cloud*
water, which uses the cloud shape ``μ_c`` diagnosed from the residual cloud
reservoir immediately before that process fires.

The prognostic field stores ``ρ\tilde z^i``, so Breeze converts this physical
sixth-moment source to a ``ρ\tilde z^i`` source (from
``d\sqrt{z n} = (n\,\dot z + z\,\dot n)/(2\sqrt{zn})``, with the
``\sqrt{\dot z\, \dot n}`` limit at initiation) and bounds the result so the sink
can never exceed ``ρ\tilde z^i / τ``.

Two other overloads exist. `tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, ...)` is a
simplified proportional-scaling form, ``(z^i/q^i)\,\dot q^i_\text{group1}`` plus the
same group-2 sum, used only where no ice integral table is available. The fully
tabulated `tendency_ρzⁱ(rates, ρ, ..., p3, nu, D_v, μ, μ_cloud)` overload
corresponds to Fortran's *inactive* `log_full3mom` branch.

### Liquid on Ice Tendency

```math
\partial_t (ρq^{wi})\big|_\text{src}
=\rho\big[\text{MELT}_\text{partial} + \text{COL}_{c,\text{warm}} + \text{COL}_{r,\text{warm}}
+ \text{WG}_c + \text{WG}_r + \text{COAT-COND}
- \text{SHED} - \text{REFR} - \text{COAT-EVAP}\big],
```

valid in the active liquid-fraction branch. Above-freezing
collection of cloud and rain feeds the liquid coating; when wet growth is
diagnosed below freezing, ``\text{WG}_c`` and ``\text{WG}_r`` route the collected
mass there too. ``\text{WG}_\text{shed}`` does not appear in this budget: it is
nonzero only without liquid fraction and routes excess cloud water directly to
rain. ``\text{COAT-COND}`` and ``\text{COAT-EVAP}``
are the coupled liquid-coated-ice condensation / evaporation rates, which are
active when ``F^l \ge`` `liquid_fraction_clipping_threshold` (0.01) — below it,
the dry-ice deposition branch runs instead. With liquid fraction *off*, the
ordinary liquid-fraction terms are zero and any leftover coating is drained to
rain through ``\text{SHED}`` over `sink_limiting_timescale`.

### Vapor and Aerosol Tendencies

```math
\partial_t (ρq^v)\big|_\text{src}
=\rho\big[\text{REVP} + \text{COAT-EVAP}
- \text{COND} - \text{DEP} - \text{NUC} - \text{CCN}_q - \text{RAIN-COND} - \text{COAT-COND}\big],
```

with the bidirectional ``\text{COND}`` and ``\text{DEP}`` supplying their own
evaporation / sublimation branches through their negative values.

```math
\partial_t (ρn^a)\big|_\text{src} = -\rho\,\text{CCN}_n,
```

one aerosol removed per activated droplet; zero in the prescribed-``N_c`` path.

## Sedimentation

Each quantity sediments at its characteristic velocity. The velocities are
diagnosed once per RK stage by `prepare_microphysical_tendencies!` into z-Face
fields, because the scalar flux divergence consumes them as advecting velocities
at ``(\text{Center}, \text{Center}, \text{Face})``.

| Variable | Sedimentation Velocity | Flux |
|----------|----------------------|------|
| ``ρq^{cl}`` | ``V_m^{cl}`` (mass-weighted Stokes) | ``F_q^{cl} = -V_m^{cl} ρq^{cl}`` |
| ``ρn^{cl}`` | ``V_n^{cl}`` (number-weighted Stokes) | ``F_n^{cl} = -V_n^{cl} ρn^{cl}`` |
| ``ρq^r`` | ``V_m^r`` | ``F_q^r = -V_m^r ρq^r`` |
| ``ρn^r`` | ``V_n^r`` | ``F_n^r = -V_n^r ρn^r`` |
| ``ρq^i`` | ``V_m^i`` | ``F_q^i = -V_m^i ρq^i`` |
| ``ρn^i`` | ``V_n^i`` | ``F_n^i = -V_n^i ρn^i`` |
| ``ρq^f`` | ``V_m^i`` | ``F_q^f = -V_m^i ρq^f`` |
| ``ρb^f`` | ``V_m^i`` | ``F_b^f = -V_m^i ρb^f`` |
| ``ρ\tilde z^i`` | ``\tfrac{1}{2}(V_z^i + V_n^i)`` | ``F_{\tilde z}^i = -\tfrac{1}{2}(V_z^i + V_n^i)\, ρ\tilde z^i`` |
| ``ρq^{wi}`` | ``V_m^i`` | ``F_q^{wi} = -V_m^i ρq^{wi}`` |

``ρs^{sat}`` and ``ρn^a`` do not sediment. Cloud droplets do: Fortran settles
cloud mass and number with DSD-integrated Stokes velocities, and Breeze mirrors
that.

The advected ``ρ\tilde z^i`` uses the mean of the ``Z``- and ``N``-weighted
particle speeds, which is the sedimentation characteristic implied by
``d\sqrt{ZN} = \tfrac{1}{2}\sqrt{N/Z}\,dZ + \tfrac{1}{2}\sqrt{Z/N}\,dN``. The
purely reflectivity-weighted ``V_z^i`` is still tabulated and stored as a
diagnostic.

The sedimentation tendency is

```math
\frac{\partial ρX}{\partial t}\bigg|_\text{sed} = -\frac{\partial F_X}{\partial z}.
```

At the bottom face, `precipitation_boundary_condition = nothing` (the default)
keeps the diagnosed fall speed, so precipitation leaves the domain through an
open surface; an `ImpenetrableBoundaryCondition()` zeroes it instead, so
precipitation accumulates in the lowest cell. The top face is held at zero, so
nothing sediments in from above the model top.

Breeze does not subcycle sedimentation inside P3 (Fortran's `dt_left` loop is
not ported); Oceananigans is responsible for stability in transport.

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

``ρz̃ⁱ`` appears only in 3-moment mode. ``ρnᶜˡ`` and ``ρnᵃ`` appear only when `aerosol`
is an `AerosolActivation`: the default prescribed-Nᶜ path (Fortran
`log_predictNc = .false.`) takes droplet number from `cloud.number_concentration`, so
neither field is allocated or advected there. ``ρsˢᵃᵗ`` appears only when
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
\frac{d}{dt}\left( q_v + q^{cl} + q^r + q^i + q^{wi} \right) = 0.
```

(The liquid coating ``q^{wi}`` is included because shedding moves it to rain,
and refreezing converts it to rime — both internal to the ice mass.)

Within P3 the limiting happens in two stages, in Fortran's order. First
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
stage. The optional ``ρs^{sat}`` prognostic is also excluded: subsaturation is
legitimately negative. Breeze does not implement a Fortran-style post-step
"return small mass to vapor" cleanup, because that requires state mutation with
a paired ``θ`` correction.

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
- zeroes the ice-population fields (``ρn^i``, ``ρq^f``, ``ρb^f``,
  ``ρ\tilde z^i``) orphaned by a vanished ``ρq^i``, and ``ρn^r`` / ``ρn^{cl}``
  orphaned by their masses. ``ρq^{wi}`` is deliberately *not* paired with
  ``ρq^i``: it is real water, and the whole-particle clip already sheds it to
  rain when the dry ice is gone;
- clamps the remaining non-water fields (number moments, rime properties, sixth
  moment, aerosol count). ``ρs^{sat}`` is excluded, since subsaturation is
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
handling — `liquid_fraction_clipping_threshold` (Fortran `liqfracsmall`) and
`tiny_ice_to_rain_threshold` (Fortran `qsmall_dry`, in kg/kg):

```jldoctest thresholds
prp = microphysics.process_rates

(prp.liquid_fraction_clipping_threshold,
 prp.tiny_ice_to_rain_threshold)

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
- [MilbrandtEtAl2021](@cite): Transformed sixth-moment prognostic
  (``ρ\tilde z^i``) for three-moment ice.
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid fraction prognostic (``ρq^{wi}``).
- [Morrison2025complete3moment](@cite): Complete tendency equations with all six ice variables.
- [MilbrandtYau2005](@cite): Multi-moment microphysics and sedimentation.
