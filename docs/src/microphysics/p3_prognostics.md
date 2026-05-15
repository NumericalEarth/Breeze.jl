# [Prognostic Variables and Tendencies](@id p3_prognostics)

P3 tracks 11 prognostic densities that together describe the complete microphysical
state of the atmosphere. This section documents each variable, its physical
meaning, and the source-term assembly used in `tendency_ρ*` (`process_rates.jl`)
to build the microphysical tendency for each prognostic field.

The prognostic variable formulation has evolved through the P3 papers:

- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Original 4 ice variables.
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Added ``ρz^i`` for 3-moment ice.
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Added ``ρq^{wi}`` for liquid fraction.

Our implementation follows P3 v5.5 with all 6 ice prognostic variables. Sign
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

### Cloud Liquid

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^{cl}`` | Cloud liquid mass density | kg/m³ | Mass of cloud droplets per unit volume |
| ``ρn^{cl}`` | Cloud droplet number density | m⁻³ | Number of cloud droplets per unit volume |

Breeze always carries ``ρn^{cl}`` as a prognostic field. When the optional
aerosol-activation path (`AerosolActivation` in `aerosol_activation.jl`) is
enabled, CCN-activation source terms drive ``ρn^{cl}``. Otherwise the field
is held at the configured constant cloud-droplet number (typical continental
``\sim 100`` cm⁻³ or marine ``\sim 50`` cm⁻³).

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
| ``ρz^i`` | Ice reflectivity | m⁶/m³ | 6th moment of size distribution (only updated when `three_moment_ice = true`) |
| ``ρq^{wi}`` | Water on ice | kg/m³ | Liquid water coating ice particles |

### Vapor and Saturation Diagnostic

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^v`` | Water vapor density | kg/m³ | The host-coupled moisture variable |
| ``ρs^{sat}`` | Predicted supersaturation | kg/m³ | Predicted-supersaturation path. Fortran v5.5 hard-codes `log_predictSsat = .false.`; Breeze's `predict_supersaturation` flag defaults to `false` to match. When `false`, the prognostic field is inactive and has zero microphysical tendency; diagnostics use ``q^v - q^{v,s}(T)`` directly. When `true`, the bounded G&M (2008) adjustment is active. |

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
``\rho \cdot (\text{gain} - \text{loss})`` calls in `process_rates.jl`.

### Cloud Liquid Tendency

```math
\partial_t (ρq^{cl})\big|_\text{src}
=\rho\big[\text{COND} + \text{CCN}_q
- \text{AUTO} - \text{ACCR} - \text{RIM}_c
- \text{IMMF}_c - \text{HOM}_c
- \text{COL}_{c,\text{warm}} - \text{WG}_c \big].
```

| Term | Meaning |
|------|--------|
| COND | Condensation (positive) / evaporation (negative) — bidirectional. |
| CCN``_q`` | CCN-activation mass source (when prognostic ``N_c`` enabled). |
| AUTO | Autoconversion to rain. |
| ACCR | Accretion by rain. |
| RIM``_c`` | Cloud riming by ice. |
| IMMF``_c`` | Immersion freezing of cloud droplets. |
| HOM``_c`` | Homogeneous freezing (``T < -40°``C). |
| COL``_{c,\text{warm}}`` | Cloud collection by ice above ``T_0`` (routes to ``q^{wi}`` or shedding). |
| WG``_c`` | Wet-growth re-routing of cloud collection (excess into ``q^{wi}``). |

### Rain Mass Tendency

```math
\partial_t (ρq^r)\big|_\text{src}
=\rho\big[\text{AUTO} + \text{ACCR} + \text{RAIN-COND} + \text{MELT}_\text{full}
+ \text{SHED} + \text{WG}_\text{shed}
- \text{REVP} - \text{RIM}_r - \text{IMMF}_r - \text{HOM}_r
- \text{COL}_{r,\text{warm}} - \text{WG}_r \big].
```

| Term | Meaning |
|------|--------|
| RAIN-COND | Coupled rain condensation (bidirectional). |
| MELT``_\text{full}`` | "Complete" melting flux from ice → rain. |
| SHED | Liquid coating shed from ice. |
| WG``_\text{shed}`` | Wet-growth shedding (Fortran `nrshdr`). |
| REVP | Rain evaporation. |
| RIM``_r`` | Rain riming by ice. |
| IMMF``_r`` | Immersion freezing of rain. |
| HOM``_r`` | Homogeneous freezing of rain. |
| COL``_{r,\text{warm}}`` | Rain collection by ice above ``T_0`` (routes to ``q^{wi}`` when liquid fraction is on). |
| WG``_r`` | Wet-growth re-routing of rain collection. |

### Rain Number Tendency

```math
\partial_t (ρn^r)\big|_\text{src}
=\rho\big[\text{AUTO}_n + \text{MELT}_n + \text{BR}
+ \text{SHED}_n + \text{COL}_{c,\text{warm},n}\, [F^l\!=\!0]
+ \text{WG}_{\text{shed},n}
- \text{REVP}_n - \text{SCBK} - \text{RIM}_{r,n}
- \text{IMMF}_{r,n} - \text{HOM}_{r,n} - \text{COL}_{r,\text{warm},n}\big]
+ \text{N-CORR}.
```

- ``\text{AUTO}_n = \text{AUTO} / m_\text{drop,init}`` (drops produced from autoconversion).
- ``\text{MELT}_n = (n^i / q^i) \cdot \text{MELT}_\text{full}`` (drops produced from melting).
- ``\text{BR}`` is the rain-breakup number source.
- ``\text{SHED}_n \approx 1.928 \times 10^6 \cdot \text{SHED}`` (1 mm shed drops).
- ``\text{COL}_{c,\text{warm},n}`` only contributes when liquid fraction is *off*
  (above-freezing collected cloud is then shed as rain rather than added to ``q^{wi}``).
- ``\text{REVP}_n = (n^r / q^r) \cdot \text{REVP}`` (proportional removal).
- ``\text{SCBK}`` is the rain self-collection magnitude.
- ``\text{N-CORR}`` is the diagnosed PSD ``λ``-bound number correction
  (Fortran `get_rain_dsd2` writes back a clipped ``n_r``; Breeze adds a
  matching tendency rather than mutating the prognostic state).

### Ice Mass Tendency

```math
\partial_t (ρq^i)\big|_\text{src}
=\rho\big[\text{DEP} + \text{RIM}_c + \text{RIM}_r + \text{REFR}
+ \text{NUC} + \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r
- \text{MELT}_\text{partial} - \text{MELT}_\text{full}\big].
```

Splintering does *not* appear in the ice mass tendency:
splinters are fragments of existing rime mass, so the splintered mass
is internally subtracted from rime in `tendency_ρqᶠ` and added back as
splinter mass — netting to zero in ``q^i``. The deposition term is
bidirectional; sublimation is its negative branch.

### Ice Number Tendency

```math
\partial_t (ρn^i)\big|_\text{src}
=\rho\big[\text{NUC}_n + \text{IMMF}_{c,n} + \text{IMMF}_{r,n} + \text{HOM}_{c,n} + \text{HOM}_{r,n}
+ \text{HM}_n
- \text{MELT}_n - \text{SUB}_n - \text{AGG}_n - \text{NLIM}\big].
```

- ``\text{HM}_n`` is the Hallett–Mossop number source.
- ``\text{SUB}_n`` is the sublimation number sink.
- ``\text{AGG}_n`` is the aggregation magnitude.
- ``\text{NLIM}`` is the soft-relaxation analog of Fortran's `impose_max_Ni`
  hard cap. When ``n^i`` exceeds ``N_{i,\max}/ρ``, a 10 s relaxation sink is
  added to push it back toward the cap.

### Rime Mass Tendency

```math
\partial_t (ρq^f)\big|_\text{src}
=\rho\big[\text{RIM}_c + \text{RIM}_r + \text{REFR}
+ \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r
+ \text{WG-DENS}_q
- F^f\,(\text{MELT}_\text{partial} + \text{MELT}_\text{full} + \text{SUB})\big].
```

``\text{WG-DENS}_q`` is the wet-growth densification mass term: when wet-growth
shedding fires (without active liquid fraction), the rime is set to its
maximum density. ``\text{SUB}`` is the sublimation mass magnitude
(``\text{SUB} = \max(0, -\text{DEP})``).

### Rime Volume Tendency

```math
\partial_t (ρb^f)\big|_\text{src}
=\rho\!\Bigg[\frac{\text{RIM}_c}{ρ^f_\text{new}}
+ \frac{\text{RIM}_r + \text{REFR} + \text{IMMF}_c + \text{IMMF}_r + \text{HOM}_c + \text{HOM}_r}{ρ_{r,\max}}
+ \text{WG-DENS}_b
- \frac{F^f\,(\text{MELT}_\text{total} + \text{SUB})}{ρ^f}
- \mathcal{D}\Bigg].
```

The various rime-density denominators reflect the Fortran convention:
fresh cloud rime uses the Cober–List density ``ρ^f_\text{new}``; rain
riming, refreezing, immersion freezing, and homogeneous freezing all
deposit at the maximum rime density ``ρ_{r,\max} = 900`` kg/m³. ``\mathcal{D}``
is the melt-densification correction that drives the remaining rime
toward solid ice density (917 kg/m³) when ``ρ^f < 917`` and liquid
fraction is *not* active.

### Reflectivity Tendency (3-moment)

The simplified path used by `tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ)` follows
the active hybrid path (see the sixth-moment update in [Microphysical Processes](@ref p3_processes)):

```math
\partial_t (ρz^i)\big|_\text{src}
=\rho\Big[\frac{z^i}{q^i}\,\dot{q}^{i}_\text{group1}
+ \sum_{p\in\text{group2}} G(μ_\text{src,p})\,\frac{(ΔM_3)_p^2}{Δn_p}\Big],
```

with the group-1 mass change including deposition (signed), riming (cloud +
rain), refreezing, partial+complete melting, and ``q^{wi}`` condensation /
evaporation; and the group-2 sum running over deposition nucleation,
immersion freezing of cloud / rain, splintering, and homogeneous freezing
of cloud / rain.

The fully tabulated `tendency_ρzⁱ(rates, ρ, ..., p3, nu, D_v, μ, μ_cloud)`
overload exists for completeness but corresponds to Fortran's *inactive*
`log_full3mom` branch.

### Liquid on Ice Tendency

```math
\partial_t (ρq^{wi})\big|_\text{src}
=\rho\big[\text{MELT}_\text{partial} + \text{COL}_{c,\text{warm}} + \text{COL}_{r,\text{warm}}
+ \text{WG}_c + \text{WG}_r + \text{COAT-COND}
- \text{SHED} - \text{REFR} - \text{COAT-EVAP}\big],
```

valid in the active liquid-fraction branch (``F^l \ge 0.01``). Above-freezing
collection of cloud and rain feeds the liquid coating; below-freezing
wet-growth excess does too. ``\text{COAT-COND}`` and ``\text{COAT-EVAP}``
are the coupled liquid-coated-ice condensation / evaporation rates.

## Sedimentation

Each quantity sediments at its characteristic velocity:

| Variable | Sedimentation Velocity | Flux |
|----------|----------------------|------|
| ``ρq^r`` | ``V_m^r`` | ``F_q^r = -V_m^r ρq^r`` |
| ``ρn^r`` | ``V_n^r`` | ``F_n^r = -V_n^r ρn^r`` |
| ``ρq^i`` | ``V_m^i`` | ``F_q^i = -V_m^i ρq^i`` |
| ``ρn^i`` | ``V_n^i`` | ``F_n^i = -V_n^i ρn^i`` |
| ``ρq^f`` | ``V_m^i`` | ``F_q^f = -V_m^i ρq^f`` |
| ``ρb^f`` | ``V_m^i`` | ``F_b^f = -V_m^i ρb^f`` |
| ``ρz^i`` | ``V_z^i`` | ``F_z^i = -V_z^i ρz^i`` |
| ``ρq^{wi}`` | ``V_m^i`` | ``F_q^{wi} = -V_m^i ρq^{wi}`` |

The sedimentation tendency is

```math
\frac{\partial ρX}{\partial t}\bigg|_\text{sed} = -\frac{\partial F_X}{\partial z}.
```

Breeze does not subcycle sedimentation inside P3 (Fortran's `dt_left` loop is
not ported); Oceananigans is responsible for stability in transport.

## Coupling to AtmosphereModel

In Breeze, P3 microphysics couples to `AtmosphereModel` through the
microphysics interface in `p3_interface.jl`. Under the hood,
``\_p3\_scalar\_compute`` returns a `P3CacheResult` that the field-by-field
`microphysical_tendency` overloads consume:

```julia
# Prognostic field names
names = prognostic_field_names(microphysics)
# (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρzⁱ, :ρqʷⁱ, :ρsˢᵃᵗ)
```

Three host-facing entry points:

1. **`microphysical_tendency`**: Computes source terms for all prognostic variables.
2. **`compute_moisture_fractions`**: Converts prognostic densities to mixing ratios.
3. **`update_microphysical_fields!`**: Refreshes diagnostic fields after a state update.

The tendency-only architecture is described in
[Architectural choice: Breeze P3 is tendency-only](@ref p3_overview).

## Conservation Properties

P3 conserves total water in a closed system:

```math
\frac{d}{dt}\left( q_v + q^{cl} + q^r + q^i + q^{wi} \right) = 0.
```

(The liquid coating ``q^{wi}`` is included because shedding moves it to rain,
and refreezing converts it to rime — both internal to the ice mass.)

Within P3, the limiter `limit_vapor_rates` enforces the saturation-adjustment
caps (see the saturation adjustment limits in [Microphysical Processes](@ref p3_processes)) and the
phase-conservation caps in `phase2_conservation_limit!` ensure that
sources do not exceed the available reservoir over the host time step.

Energy conservation is delegated to the host: the Anelastic and
compressible formulations carry latent heating implicitly through their
prognostic thermodynamic variable. P3 itself does not assemble a ``θ``
tendency.

## Numerical Considerations

### Positivity

All prognostic variables remain non-negative through the saturation-adjustment
caps and per-species sink-limiting factors in `process_rates.jl`. Breeze does
not implement a Fortran-style post-step "return small mass to vapor" cleanup
because that requires state mutation with a paired ``θ`` correction.

### Consistency

The rime fraction must satisfy ``0 \le F^f \le 1`` (so ``ρq^f \le ρq^i``) and
the liquid fraction ``0 \le F^l \le 1``. When the diagnosed fractions exceed
bounds, the read-time interface caps them and flags the state for special
handling (see `p3_interface.jl`).

### Threshold Handling

Small values below numerical thresholds are treated as zero in the source
assembly:

```julia
q_min = microphysics.minimum_mass_mixing_ratio  # default: 1e-14 kg/kg
n_min = microphysics.minimum_number_mixing_ratio  # default: 1e-16 1/kg
```

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
- [MilbrandtEtAl2021](@cite): Sixth moment prognostic (``ρz^i``) for three-moment ice.
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid fraction prognostic (``ρq^{wi}``).
- [Morrison2025complete3moment](@cite): Complete tendency equations with all six ice variables.
- [MilbrandtYau2005](@cite): Multi-moment microphysics and sedimentation.
