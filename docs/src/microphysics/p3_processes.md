# [Microphysical Processes](@id p3_processes)

This section documents the process rate formulations as they are implemented
in Breeze, with explicit notes wherever Breeze diverges from the Fortran v5.5.0
reference in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

The bulk of the implementation lives in:

- `process_rates.jl` — top-level rate assembly and per-field tendencies.
- `rain_process_rates.jl` — KK2000 warm-rain rates.
- `ice_nucleation_rates.jl` — Cooper deposition nucleation, immersion freezing, homogeneous freezing.
- `melting_rates.jl` — heat-balance melting (with optional Fˡ split).
- `collection_rates.jl` — riming and aggregation.
- `ice_rain_collection.jl` — ice–rain collection.

## Process Map

The following block diagram summarises the active mass-flow paths between
species in a single ice category. Number-only paths (self-collection,
breakup, aggregation, splintering) are noted in the per-section text.

```
          ┌─────────────┐                ┌─────────────┐
          │   Vapor q_v │                │  Liquid on  │
          └──────┬──────┘                │   ice qʷⁱ   │
                 │                       └──┬───┬───┬──┘
   condensation │  deposition / sublimation │   │   │
                ▼                       ▲   │   │   │ partial melt
         ┌──────────┐                   │   │   │   │ wet growth
         │  Cloud   │     riming        │   │   │   │
         │  q_cl    ├──────────────────►│   │   │   │  shedding
         └────┬─────┘                   │   │   │   ▼
   accretion │ autoconversion           │   │   │ ┌──────────┐
             ▼                           │   │   │ │  Ice qⁱ  │
        ┌──────────┐  ice–rain collect.  │   │   │ │  rime qᶠ │
        │  Rain    ├────────────────────►│   │   │ │  vol bᶠ  │
        │  q_r,n_r │      complete melt  │   │   │ │  Z   zⁱ  │
        └────┬─────┘ ◄───────────────────┘   │   │ └─┬────┬───┘
             │ rain evaporation             │   │   │    │
             └──── self-collection / breakup◄┘   │   │    │ refreeze
                                                ▲   │    │
                                                └───┴────┘
```

## Warm-Rain Microphysics

### Autoconversion (KK2000)

Cloud droplets coalesce to form rain following [Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000):

```math
\dot{q}_\text{aut} = k_1\, q_{cl}^{\alpha}\, \left(\frac{N_c}{N_{c,\text{ref}}}\right)^{\beta},
```

with the runtime defaults ``k_1 = 1350``, ``α = 2.47``, ``β = -1.79``, and the
in-cloud cloud-water threshold ``q_\text{small,1} = 10^{-8}`` kg/kg below which
the rate is gated to zero. ``N_c`` is the cloud-droplet number concentration
in m⁻³ and ``N_{c,\text{ref}} = 10^6`` m⁻³ is a reference concentration that
absorbs the ``× 10^{-6}`` unit conversion in the original KK2000 form.

!!! note "Single warm-rain option"
    The reference Fortran offers Seifert–Beheng 2001 and Kogan 2013 as
    alternatives via the `autoAccr_param` switch. Breeze hard-codes
    KK2000.

### Accretion (KK2000)

```math
\dot{q}_\text{acc} = k_2\, (q_{cl}\, q_r)^{\alpha},
```

with ``k_2 = 67`` and ``α = 1.15``.

### Rain self-collection and breakup

Number-only term, modeling the balance between large drops collecting smaller
ones and very large drops breaking up (modified from Verlinde and Cotton 1993):

```math
\dot{N}_{r,\text{slf}} = k_{r,\text{slf}}\, ρ\, q_r\, N_r,
```

with ``k_{r,\text{slf}} = 5.78`` m³ kg⁻¹ s⁻¹.
A breakup multiplier modifies this rate by ``f_\text{brk}``:

```math
f_\text{brk} = \begin{cases}
1 & D_r < D_\text{th} \\
2 - \exp\!\left[κ_\text{br}\,(D_r - D_\text{th})\right] & D_r \ge D_\text{th},
\end{cases}
```

where ``D_r = 1/λ_r`` (the Fortran convention; for an exponential PSD this is
proportional to but not equal to the mass-mean diameter), ``D_\text{th} = 280``
μm, and ``κ_\text{br} = 2300`` m⁻¹. Above the threshold the multiplier becomes
negative, i.e. breakup outweighs self-collection.

### Rain evaporation

Below cloud base, rain evaporates into subsaturated air following the
ventilation-enhanced vapor diffusion equation
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)
appendix C, section b; [Pruppacher and Klett (1997)](@cite pruppacher2010microphysics)):

```math
\frac{dq_r}{dt}\bigg|_\text{evp} = 2π\,\frac{N_r}{Γ(μ_r+1)}\,ρ\,D_v\,(S - 1)\,
                                   \left[\frac{f_{1r}\, Γ(μ_r+2)}{λ_r}
                                       + f_{2r}\,\sqrt{ρ/η}\,\text{Sc}^{1/3}\,I_\text{vent}\right],
```

with ``f_{1r} = 0.78``, ``f_{2r} = 0.32``, and ``I_\text{vent}`` the
ventilation integral computed from the rain DSD (`RainEvaporation` integral).
The number tendency follows the proportionality
``\dot{N}_{r,\text{evp}} = (N_r/q_r)\, \dot{q}_{r,\text{evp}}`` consistent with the Fortran
implementation.

## Ice Nucleation

### Deposition / condensation-freezing nucleation (Cooper)

Active when ``T < T_\text{nuc} = 258.15`` K (``-15°``C) and the ice
supersaturation ``S_i \ge S_{i,\text{nuc}}`` (default 5%).
[Cooper (1986)](@cite Cooper1986):

```math
N_\text{Cooper} = c_\text{nuc}\, \exp\!\left[0.304\,(T_0 - T)\right]\, \rho^{-1}\quad [\text{kg}^{-1}],
```

with ``c_\text{nuc} = 5\,\text{m}^{-3}`` (i.e. ``0.005`` L⁻¹). The equilibrium
ice number is capped at the global maximum:

```math
N_\text{eq} = \min\!\left(N_\text{Cooper},\; N_\text{max}/ρ\right),\qquad N_\text{max} = 10^5\,\text{m}^{-3}.
```

The instantaneous Fortran rate ``(N_\text{eq} - n_i)/Δt`` is replaced by a
fixed-timescale relaxation toward ``N_\text{eq}``:

```math
\dot{N}_\text{nuc} = \max\!\left(0,\, \frac{N_\text{eq} - n_i}{τ_\text{nuc}}\right),
\qquad τ_\text{nuc} = 10\;\text{s}.
```

The mass rate is ``\dot{q}_\text{nuc} = m_{i0}\, \dot{N}_\text{nuc}`` with
``m_{i0} = (4π/3)\, ρ_i\, (1\,μ\text{m})^3`` and ``ρ_i = 900`` kg/m³.

!!! note "Tendency-only relaxation timescale"
    Fortran uses ``1/Δt`` because P3 has access to its own subcycle Δt;
    Breeze's tendency-only P3 does not see the host Δt and falls back to a
    fixed 10 s relaxation. For ``Δt \ll 10`` s this under-produces and for
    ``Δt \gg 10`` s it over-produces relative to Fortran.

### Immersion freezing (Barklie–Gokhale)

Active when ``T \le T_\text{imm} = 269.15`` K (``-4°``C), applied to both
cloud droplets and rain via the cloud / rain DSD integrals from
[Barklie and Gokhale (1959)](@cite BarklieGokhale1959):

```math
\dot{q}_\text{het,c} = \frac{π^2}{36}\, ρ_w\, b_\text{imm}\,
                      \frac{N_c}{Γ(μ_c+1)}\, Γ(7+μ_c)\,
                      \exp[a_\text{imm}\,(T_0-T)]\, λ_c^{-6},
```

```math
\dot{N}_\text{het,c} = \frac{π}{6}\, b_\text{imm}\,
                      \frac{N_c}{Γ(μ_c+1)}\, Γ(μ_c+4)\,
                      \exp[a_\text{imm}\,(T_0-T)]\, λ_c^{-3},
```

with ``a_\text{imm} = 0.65`` and ``b_\text{imm} = 2`` m⁻³ s⁻¹. The same form is
applied to rain with ``μ_r = 0`` (matching the Fortran runtime, where the
Cao-2008 variable-``μ_r`` path is disabled). In Breeze the cloud ``μ_c``
is diagnosed dynamically from the local ``N_c`` via the Liu and Daum (2000)
relation in `psd_corrections.jl`.

### Contact freezing

Disabled in both the Fortran reference and Breeze.

### Homogeneous freezing

Active when ``T < T_\text{hom} = 233.15`` K (``-40°``C). All remaining cloud
liquid and rain are converted to ice on a timescale ``τ_\text{hom}``:

```math
\dot{q}_{c,\text{hom}} = q_{cl}/τ_\text{hom},\qquad
\dot{q}_{r,\text{hom}} = q_r/τ_\text{hom},
```

with the matching number rates. The frozen mass is added to ice as fully
rimed material at the maximum rime density (``ρ_{r,\max} = 900`` kg/m³).
Fortran's homogeneous-freezing block runs after sedimentation as an
instantaneous ``Δt``-paced cleanup; Breeze's tendency-only equivalent uses
the fixed relaxation timescale.

### Hallett–Mossop rime splintering

Active for ``-8°\text{C} < T < -3°\text{C}`` and ice with diameter
``D \ge D_\text{HM} = 250\;μ``m and liquid fraction ``< 0.1``:

```math
f_\text{HM} = \begin{cases}
(T_2 - T)\, \kappa_1 & T_1 < T < T_2 \\
(T - T_0)\, \kappa_2 & T_0 \le T \le T_1
\end{cases},
```

with ``T_0 = 265.15``, ``T_1 = 268.15``, ``T_2 = 270.15`` K. The number rate
is ``\dot{N}_\text{HM} = 3.5 \times 10^5\, \dot{q}_\text{rim}\, 10^3\, f_\text{HM}``
(splinters per gram of rime). The mass rate uses an initial diameter
``D_\text{init,HM} = 10\;μ``m at ``ρ_i = 900`` kg/m³.

!!! warning "Surface-temperature shutoff diverges"
    Fortran shuts off Hallett–Mossop when the *column-bottom* temperature
    exceeds 282 K (a proxy for surface conditions); Breeze uses the local
    cell air temperature, which for tall columns is far below 282 K and
    therefore never triggers the shutoff. Fortran also raises ``D_\text{HM}``
    to 1000 μm for ``n_\text{cat} > 1``; Breeze uses the single-category
    threshold (250 μm) regardless.

## Droplet Activation (CCN)

Cloud droplet number is prognostic when CCN activation is enabled. Aerosol
activation follows the equilibrium Köhler-theory approach of
[Morrison and Grabowski (2007)](@cite MorrisonGrabowski2007), with
multi-mode lognormal aerosol distributions and a ``\sigma_g`` width parameter.
The activated number is:

```math
N_\text{act} = N_a\,\frac{1}{2}\left[1 - \text{erf}\!\left(\frac{2\,\ln(s_m/S)}{4.242\,\ln σ_g}\right)\right],
```

where ``s_m`` is the mean activation supersaturation (function of aerosol
size and chemistry), and ``S`` is the environmental supersaturation. The
resulting rate is gated by ``\max(0, N_\text{act} - N_c)`` divided by
the same fixed nucleation timescale as Cooper.

## Ice Collection and Riming

### Cloud–ice collection (riming)

Ice particles collect cloud droplets at ``T \le T_0``:

```math
\frac{dq_f}{dt} = ρ\, E_{ic}\, ρ_\text{corr}\, \mathcal{K}_{ic}\, q_{cl}\, N_i,
```

where ``\mathcal{K}_{ic}`` is the PSD-integrated cloud-collection kernel
``\int A(D)\, V(D)\, N'(D)\, dD`` (referred to as ``f_{1\text{pr04}}``
in the Fortran lookup tables). ``E_{ic} = 0.5``,
``ρ_\text{corr} = (ρ_s/ρ)^{0.54}`` is the air-density fall-speed correction.
Cloud number is collected proportionally:
``\dot{N}_\text{ccol} = ρ\, E_{ic}\, ρ_\text{corr}\, \mathcal{K}_{ic}\, N_c\, N_i``.

The rime volume increases as ``\dot{b}_f = \dot{q}_f / ρ_f``, with the
rime density ``ρ_f`` computed from the Cober–List parameterization
described in [Particle Properties](@ref p3_particle_properties).

### Above-freezing collection

For ``T > T_0``, when liquid fraction is active the collected mass goes to
``q^{wi}`` (liquid coating); otherwise it is shed as 1 mm rain drops with
``\dot{N}_{r,\text{shed}} = 1.923 \times 10^6\, \dot{q}_{c,\text{shed}}``.

### Ice–rain collection

Rain collected by ice uses the ice–rain double integral
(`IceRainCollection` family, ``f_{1\text{pr07}}``, ``f_{1\text{pr08}}``):

```math
\dot{q}_\text{rcol} = 10^{f_{1\text{pr08}} + \log_{10} N_{0r}}\, ρ\, ρ_\text{corr}\, E_{ri}\, N_i,
```

with ``E_{ri} = 1.0``. The corresponding number rate uses
``f_{1\text{pr07}}`` analogously.

### Aggregation

Ice particles aggregate to form larger ice. The number sink integral is
``\mathcal{K}_\text{agg}``:

```math
\frac{dN_i}{dt}\bigg|_\text{agg} = -E_{ii}(T)\, E_{ii,\text{fact}}(F^f)\,
                                   \mathcal{K}_\text{agg}\, ρ\, ρ_\text{corr}\, N_i^2.
```

The temperature-dependent efficiency follows
[Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization):

```math
E_{ii}(T) = \begin{cases}
0.001 & T < 253.15\;\text{K} \\
\text{linear ramp from } 0.001 \to 0.3 & 253.15 \le T < 273.15\;\text{K} \\
0.3 & T \ge 273.15\;\text{K}
\end{cases}.
```

The rime ramp ``E_{ii,\text{fact}}`` shuts off aggregation for heavily rimed
particles: 1 for ``F^f < 0.6`` and ramping linearly to 0 at ``F^f = 0.9``.

!!! note "Inter-category collection"
    The single-category aggregation kernel above is fully wired. The
    multi-category `inter_category_collection` function exists in
    `multi_ice_category.jl` but is currently a placeholder that is not
    invoked from the tendency assembly.

## Vapor Deposition and Sublimation

P3's deposition step uses a coupled semi-analytic vapor balance: cloud,
rain, ice, and liquid-coated ice all draw from (or release to) a common
vapor reservoir. Each species ``i`` contributes an inverse relaxation
timescale ``ε_i = 2π\, ρ\, D_v\, \mathcal{C}_i\, N_i`` (where
``\mathcal{C}_i`` is the relevant ventilation-enhanced capacitance integral
from the lookup tables), and the total ``X = ∑_i ε_i\,(\text{Bergeron-corrected})``.

The deposition rate for ice category ``i`` is then
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)):

```math
\dot{q}_\text{dep,i} = \left[\frac{A\,ε_i}{X}
                            + \frac{(S_l - A/X)\,ε_i}{X\,Δt}\,(1 - e^{-X\,Δt})\right]
                      \frac{1}{1 + \frac{L_s}{c_p}\frac{dq_{v,i}}{dT}}
                      + \frac{(q_{v,s}-q_{v,i})\,ε_i}{1 + \frac{L_s}{c_p}\frac{dq_{v,i}}{dT}},
```

where ``S_l = q_v - q_{v,s}`` is the saturation deficit w.r.t. liquid and
``A`` is a forcing term that, in Fortran, includes a dynamical vapor tendency,
an adiabatic-cooling proxy, and the Bergeron offset. Breeze currently
includes only the Bergeron term — the dynamical and adiabatic terms
are assumed to be carried by the host formulation (which sees them in
its own thermodynamic equation).

Sublimation is the negative branch (``\dot{q}_\text{dep} < 0``); the corresponding
number rate scales with the dry-ice number-to-mass ratio:

```math
\dot{N}_\text{sub} = -\dot{q}_\text{dep}\,\frac{N_i}{q_i - q^{wi}}.
```

Coupled liquid-coated ice (``F^l \ge 0.01``) uses the saturation
factor ``1/a_b = 1/(1 + L_v^2 q_{v,s}/(c_p R_v T^2))`` instead of ``1/a_{bi}``,
matching the coupled liquid-ice branch in Fortran.

!!! note "SCF=1 limit"
    Breeze evaluates ``S_l`` and the saturation-adjustment caps without an
    SCF / SPF weighting; the in-cloud and clear-sky vapor fields collapse
    to the grid-mean ``q_v`` (i.e. the ``\text{SCF}=\text{SPF}=1`` limit
    of Fortran).

## Melting

Above ``T_0``, ice melts via the heat balance of Mason 1971, implemented
following the simplified diffusion approximation in
[Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)
appendix C, section i:

```math
\frac{dm}{dt} = -\frac{2π\, \mathcal{C}}{L_f}\,
                \big[K_a\,(T - T_0) + ρ\, L_v\, D_v\, (q_v - q_{v,s,0})\big]\, f_v,
```

where ``\mathcal{C}`` is the (lookup-table) capacitance, ``f_v`` is the
ventilation factor, ``q_{v,s,0}`` is the saturation mixing ratio at ``T_0``,
and the prefactor ``2π`` (rather than ``4π``) reflects the Fortran convention
where the tabulated capacitance integral stores ``\text{capm} = \mathcal{C}\, D``
(twice the physical capacitance ``C/2``).

When the liquid fraction is active, melting is split into two destinations
using a size threshold internal to the lookup tables:

- Small particles (``D \le D_\text{th}``): meltwater goes straight to rain as
  ``\dot{q}_\text{rmlt}``.
- Large particles (``D > D_\text{th}``): meltwater stays as a liquid coating
  on ice, contributing to ``q^{wi}`` as ``\dot{q}_\text{imlt}``.

The number melt rate ``\dot{N}_\text{mlt}`` is proportional to the rain-side
mass rate through the dry-ice number ratio ``N_i / (q_i - q^{wi})`` so that the
mean particle mass after melting is preserved.

When liquid fraction is inactive, the full melt rate is routed to rain
(``\dot{q}_\text{imlt} = 0``).

After tendency application, particles whose liquid fraction exceeds 0.99
("complete melting" diagnostic) transfer the remaining ice mass and number
to rain — see [Prognostic Equations](@ref p3_prognostics).

## Wet Growth and Refreezing

When the latent-heat release rate from collection exceeds what conduction
plus evaporative cooling can dissipate, ice enters wet growth.

The wet-growth capacity rate (Musil 1970):

```math
\dot{q}_\text{wg} = \big[K_a\,(T_0 - T) + 2π\,ρ\,L_s\,D_v\,(q_{v,s,0} - q_{v,\text{cld}})/L_f\big]\, f_v\, N_i.
```

Without liquid fraction, the excess
``\dot{q}_\text{wg,excess} = \dot{q}_\text{ccol} + \dot{q}_\text{rcol} - \dot{q}_\text{wg}``
is shed as 1 mm rain drops, and the rime is set to maximum density
(``q^f = q^i``, ``b^f = q^f / ρ_{r,\max}``).

With liquid fraction active, the excess is retained as ``q^{wi}``; no
densification flag is set. Refreezing then transfers ``q^{wi}`` back to
rime when ``T < T_0``:

```math
\dot{q}_\text{frz} = \dot{q}_\text{wg}|_{T < T_0},
```

bounded by ``q^{wi} / τ`` (Breeze uses the same fixed nucleation timescale
in place of Fortran's ``1/Δt``).

### Shedding

```math
\dot{q}_\text{shed} = -k_\text{shed}\, \max(0,\, F^l - F^l_\text{max})\, q^i,
```

with ``F^l_\text{max} = 0.3`` and ``k_\text{shed}`` chosen so the bulk
relaxation occurs over a few seconds. The shed mass is added to rain; the
shed number uses the ``\sim 1.928 \times 10^6`` per-kg conversion (1 mm
drops, identical to Fortran's `nlshd` factor).

!!! warning "Shedding form differs from Fortran"
    The Fortran reference computes shedding by a tabulated PSD integral
    restricted to particles with ``D \ge 9`` mm. Breeze instead applies
    a bulk relaxation toward an upper threshold liquid fraction. Both
    deplete ``q^{wi}`` and feed rain, but the size-resolved mass partition
    differs.

## Rime Density

Rime density from cloud-water collection is parameterized following
Cober and List (1993) — see [Particle Properties](@ref p3_particle_properties).
For collected rain the rime is assumed to be at the maximum density
``ρ_{r,\max} = 900`` kg/m³.

Without liquid fraction, melting drives the remaining rime toward solid ice
density (917 kg/m³) via a melt-densification term in the rime-volume tendency.
With liquid fraction active, this densification is skipped because the
liquid is tracked explicitly in ``q^{wi}``.

## Saturation adjustment limits

After all liquid- and ice-phase rates are assembled, Breeze applies four caps
matching the Fortran limits:

- Liquid condensation against
  ``\max(0,\, q_v - q_{v,s})/(1 + L_v^2 q_{v,s}/(c_p R_v T^2))``.
- Liquid evaporation against
  ``\max(0,\, q_{v,s} - q_v)/(\dots)``, plus a hard zero when supersaturated.
- Ice deposition against
  ``\max(0,\, q_v^{*} - q_{v,i}^{*})/(1 + L_s^2 q_{v,i}^{*}/(c_p R_v {T^*}^2))``,
  where ``T^* = T + \text{net liquid}\,\cdot\, L_v / c_p \cdot Δt`` and
  the saturation field is recomputed at ``T^*``.
- Ice sublimation against the negative analog.

These caps follow the saturation-adjustment limits in
[Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)
appendix C, section b (the Morrison–Grabowski 2008b semi-analytic
condensation/evaporation framework, extended to the ice phase).
`limit_vapor_rates` in `process_rates.jl` applies them in the
tendency interface.

## Sedimentation

Sedimentation is delegated to Oceananigans transport. Each prognostic field
falls at its tabulated, density-corrected velocity:

| Variable | Velocity | Reference |
|----------|---------|-----------|
| Rain mass / number | mass-weighted ``V_m^r``, number-weighted ``V_n^r`` | Gunn–Kinzer 1949 lookup tables |
| Ice mass / rime mass / rime volume / liquid coating | mass-weighted ``V_m^i`` | Mitchell–Heymsfield 2005 |
| Ice number | number-weighted ``V_n^i`` | Mitchell–Heymsfield 2005 |
| Ice 6th moment ``z^i`` | reflectivity-weighted ``V_z^i`` | Mitchell–Heymsfield 2005 |

All ice fall speeds are corrected by the air-density factor
``(ρ_s/ρ)^{0.54}`` with the 600 hPa, 253.15 K reference ``ρ_s`` for ice
and the surface ``ρ_s = p_0/(R_d\, T_0)`` for rain (matching the Fortran
`rhosur` / `rhosui`).

The Fortran adaptive `dt_left` Courant substepping is *not* part of P3
in Breeze; the host transport scheme is responsible for stability. ``V_z``
is computed for transport but does not feed back into a Courant constraint.

## Sixth-moment (``Z_i``) update

Breeze follows Fortran v5.5.0's active hybrid path. After each "group 1"
process (deposition, melting, riming, shedding, sublimation, ``q^{wi}``
condensation/evaporation), ``μ_i`` is held fixed for the step, ``M_3`` is
re-estimated from the updated ice mass and bulk density, and:

```math
Z_i = G(μ_i)\,\frac{M_3^2}{N_i},\qquad
G(μ) = \frac{(6+μ)(5+μ)(4+μ)}{(3+μ)(2+μ)(1+μ)}.
```

For "group 2" initiation processes (deposition nucleation, immersion freezing
of cloud / rain, splintering, homogeneous freezing of cloud / rain), an
explicit increment is added:

```math
ΔM_3 = \frac{6\,\dot{q}_\text{src}}{π\, ρ_i},\qquad
ΔZ_i = G(μ_\text{src})\,\frac{ΔM_3^2}{\dot{N}_\text{src}}\, Δt,
```

where ``μ_\text{src} = μ_c`` for cloud-source freezing, ``μ_\text{src} = μ_r = 0``
for rain-source freezing, and ``μ_\text{src} = 0`` for nucleation and splintering
(consistent with the Fortran `update_zi_proc2` block, which is identical
once Fortran's `mu_r_constant = 0` runtime is taken into account).

Breeze does not implement the dormant `log_full3mom` Fortran branch (which
computes per-process tabulated ``Z_i`` increments), since `log_full3mom = .false.`
is hardwired in v5.5.0.

## Temperature Dependence

Many processes have strong temperature dependence:

```
T < 233.15 K:  Homogeneous freezing of cloud and rain
233 K – 269 K: Immersion freezing (T ≤ 269.15 K)
T < 258.15 K:  Cooper deposition / condensation-freezing nucleation
265 K – 270 K: Hallett–Mossop ice multiplication (-8 to -3°C)
253 K – 273 K: Aggregation efficiency ramp (0.001 → 0.3)
T > 273.15 K:  Melting, shedding (with Fˡ active), wet growth
```

## Coupling to Thermodynamics

Microphysical processes release or absorb latent heat via the host
thermodynamic equation. The Anelastic and compressible formulations
in Breeze carry latent heating implicitly through their prognostic
``ρθ`` (or ``ρe``) variable; P3 does not assemble an explicit ``θ``
tendency. The relevant latent heats at standard conditions are:

- ``L_v \approx 2.5 \times 10^6`` J/kg (vaporization)
- ``L_s \approx 2.83 \times 10^6`` J/kg (sublimation)
- ``L_f \approx 3.34 \times 10^5`` J/kg (fusion)

## Process Summary

| Process | Affects | Key parameter / form | Reference |
|---------|---------|-----------------------|-----------|
| Condensation / evaporation | ``q^{cl}, q^r, q^{wi}`` | Coupled semi-analytic | [Morrison2015parameterization](@cite) |
| Autoconversion | ``q^{cl} \to q^r`` | KK2000 (only) | [KhairoutdinovKogan2000](@cite) |
| Accretion | ``q^{cl} \to q^r`` | KK2000 | [KhairoutdinovKogan2000](@cite) |
| Rain self-collection / breakup | ``n^r`` | Verlinde–Cotton + KK2000 | [Morrison2015parameterization](@cite) |
| Rain evaporation | ``q^r \to q_v`` | Ventilation integral | [Morrison2015parameterization](@cite) |
| Cooper nucleation | ``q^i, n^i`` | ``T < -15°``C, ``S_i \ge 5\%`` | [Cooper1986](@cite) |
| Immersion freezing | ``q^{cl}/q^r \to q^i`` | Barklie–Gokhale | [BarklieGokhale1959](@cite) |
| Homogeneous freezing | ``q^{cl}/q^r \to q^i`` | ``T < -40°``C | [Morrison2015parameterization](@cite) |
| Deposition / sublimation | ``q^i`` | Coupled semi-analytic | [Morrison2015parameterization](@cite) |
| Cloud riming | ``q^{cl} \to q^f`` | ``E_{ic} = 0.5`` | [Morrison2015parameterization](@cite) |
| Rain riming | ``q^r \to q^f`` | ``E_{ri} = 1.0`` | [Morrison2015parameterization](@cite) |
| Aggregation | ``n^i`` | ``E_{ii}(T)``, ``E_{ii,\text{fact}}(F^f)`` | [Morrison2015parameterization](@cite) |
| Hallett–Mossop | ``n^i`` | 250 μm threshold; ``-8°``C to ``-3°``C | [Morrison2015parameterization](@cite) |
| Melting | ``q^i \to q^{wi} \text{ or } q^r`` | Lookup-split by ``D_\text{th}`` | [MilbrandtEtAl2025liquidfraction](@cite) |
| Wet growth | ``q^i, q^{wi}`` | Musil 1970 | [Morrison2015parameterization](@cite) |
| Shedding (Breeze) | ``q^{wi} \to q^r`` | Bulk relaxation, ``F^l_\text{max} = 0.3`` | (diverges from [MilbrandtEtAl2025liquidfraction](@cite)) |
| Refreezing | ``q^{wi} \to q^f`` | Wet-growth form, ``T < T_0`` | [MilbrandtEtAl2025liquidfraction](@cite) |
| Sedimentation | All | Tabulated; delegated to Oceananigans | [MilbrandtYau2005](@cite) |

## References for This Section

### Core P3 Process References
- [Morrison2015parameterization](@cite): Primary process formulations (Section 2).
- [Morrison2015part2](@cite): Process validation against observations.
- [MilbrandtEtAl2021](@cite): ``Z``-tendencies for three-moment ice.
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid-fraction processes (shedding, refreezing).
- [Morrison2025complete3moment](@cite): Complete three-moment process rates.

### Related References
- [KhairoutdinovKogan2000](@cite): Warm rain autoconversion.
- [MilbrandtYau2005](@cite): Multimoment sedimentation.
- [pruppacher2010microphysics](@cite): Cloud physics fundamentals.
- [rogers1989short](@cite): Cloud physics textbook.
