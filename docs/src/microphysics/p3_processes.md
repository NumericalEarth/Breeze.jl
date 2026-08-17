# [Microphysical Processes](@id p3_processes)

This section documents the process rate formulations as they are implemented
in Breeze, with explicit notes wherever Breeze diverges from the Fortran v5.5.0
reference in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

The bulk of the implementation lives in:

- `process_rates.jl` — top-level rate assembly, sink limiting, and whole-particle clipping.
- `prognostic_tendencies.jl` — per-field `tendency_ρ*` assembly from those rates.
- `coupled_saturation_adjustment.jl` — the shared semi-analytic vapor balance
  (cloud / rain / ice / coated-ice condensation, evaporation, deposition, sublimation).
- `rain_process_rates.jl` and `warm_rain_schemes.jl` — warm-rain rates and the
  KK2000 / SB2001 / Kogan2013 selector.
- `ccn_activation_rates.jl` and `aerosol_activation.jl` — prognostic droplet activation.
- `ice_nucleation_rates.jl` — Cooper deposition nucleation, immersion freezing,
  homogeneous freezing, Hallett–Mossop splintering.
- `melting_rates.jl` — heat-balance melting (with optional Fˡ split).
- `riming_rates.jl`, `ice_collection.jl`, `ice_aggregation_rates.jl` — riming,
  above-freezing collection, and aggregation.
- `ice_rain_collection.jl` — ice–rain collection tables.
- `wet_ice_processes.jl` — Cober–List rime density, shedding, wet growth, refreezing.
- `sixth_moment_tendencies.jl` — the three-moment ``Z_i`` update.

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

Autoconversion, accretion, rain self-collection, and cloud self-collection all
dispatch on `p3.warm_rain_scheme`, mirroring Fortran's `autoAccr_param`. The
equations below are the default `KhairoutdinovKogan2000` branch;
`SeifertBeheng2001` (Long 1974 kernel with a universal function, plus an explicit
cloud self-collection sink) and `Kogan2013` (updated power laws, including a
different rain self-collection form) provide their own.

Breeze applies all warm-rain rates to the grid-mean state. Fortran scales them by
in-cloud / in-precipitation fractions (`iSCF`, `iSPF`, `SPF - SPF_clr`); with no
subgrid fraction prognostics in Breeze those factors are dropped, equivalent to
``\text{SCF} = \text{SPF} = 1``, ``\text{SPF}_\text{clr} = 0``.

### Autoconversion (KK2000)

Cloud droplets coalesce to form rain following [Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000):

```math
\dot{q}_\text{aut} = k_1\, q_{cl}^{\alpha}\, \left(\frac{N_c}{N_{c,\text{ref}}}\right)^{\beta},
```

with the runtime defaults ``k_1 \approx 0.355`` (= ``1350 \cdot 100^{-1.79}``),
``α = 2.47``, ``β = -1.79``, and the in-cloud cloud-water threshold
``q_\text{small,1} = 10^{-8}`` kg/kg below which the rate is gated to zero.
``N_c`` is the cloud-droplet number concentration in m⁻³ and
``N_{c,\text{ref}} = 10^8`` m⁻³ (= 100 cm⁻³). Breeze's ``(k_1, N_{c,\text{ref}})``
pair is a unit-rescaled equivalent of the original KK2000 form
``1350\, q_{cl}^{2.47}\, N_c[\text{cm}^{-3}]^{-1.79}`` used by the Fortran reference.

The autoconversion mass rate also sets the rain *number* source, through a
scheme-dependent seed-drop mass: a 25 μm-radius drop for KK2000
(`initial_rain_drop_mass`, Fortran `cons3⁻¹`), 40 μm for Kogan2013
(`cons8⁻¹`), and ``2/7.6923\times10^{9}`` kg for SB2001. The matching cloud number
sink is ``\text{AUTO}\, N_c/q_{cl}`` for KK2000 and Kogan2013 (Fortran `ncautc`)
and a fixed-mass drizzle drop for SB2001.

### Accretion (KK2000)

```math
\dot{q}_\text{acc} = k_2\, (q_{cl}\, q_r)^{\alpha},
```

with ``k_2 = 67`` and ``α = 1.15``.

### Rain self-collection and breakup

Number-only term, modeling the balance between large drops collecting smaller
ones and very large drops breaking up. The KK2000 self-collection coefficient
is combined with a Verlinde and Cotton (1993)-style breakup multiplier:

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

Fortran carries the result as one signed `nrslf`, so Breeze reports the two
directions separately for diagnostics but nets them back into a single signed
term before the rain-number limiter runs — rescaling only the sink half would
leave breakup at full strength against a limited sink and manufacture rain
number. Fortran likewise excludes `nrslf` from every limiter rescale list.

### Rain condensation and evaporation

The same coupled saturation-adjustment formula handles both signs.
When the rain DSD is supersaturated, vapor condenses *onto* rain
(Fortran `qrcon` positive branch); when subsaturated, rain evaporates
to vapor (Fortran `qrcon` negative branch, written as `qrevp`). Breeze's
`rain_condensation_rate` returns the signed rate; the negative branch
is split out into `rain_evaporation_rate` for the per-field tendency
assembly. Below cloud base, rain evaporates into subsaturated air following the
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

### Global ice-number cap

Independent of the post-nucleation cap ``N_\text{max} = 10^5`` m⁻³
above, Breeze enforces a per-cell global ice-number relaxation
toward ``N_{i,\max} =`` `maximum_ice_number_density` ``= 2 \times 10^6`` m⁻³
(the Fortran `max_Ni` constant in `microphy_p3.f90`):

```math
\text{NLIM} = \frac{\max(0,\; n^i - N_{i,\max}/ρ)}{τ_\text{sink}},
```

with ``τ_\text{sink} =`` `sink_limiting_timescale` (default 10 s). It enters
``\partial_t (ρn^i)`` as a sink, and is the tendency-form analog of Fortran's
`impose_max_Ni` hard clamp, which runs at several driver points in v5.5.0
including after the sedimentation block. The limiter is computed from the *raw*
prognostic ``n^i``, not the locally pre-capped value the rate functions read —
otherwise it would always be dead.

Every other rate does see the capped ``\min(n^i, N_{i,\max}/ρ)``, mirroring the
fact that Fortran caps `nitot` in place so all downstream math (process rates,
terminal velocities, the ``Z`` tendency, reflectivity) sees the same value.

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

Crucially, ``q_{cl}`` and ``q_r`` here are the **post-process residuals**, not the
beginning-of-stage values: Breeze finalizes every ordinary limiter first, then
re-diagnoses the freezing rate from the liquid that remains. That preserves
Fortran's ordering and also captures liquid *created* during the interval by
condensation, melting, or shedding. The number reservoirs are diagnosed the same
way, so frozen liquid carries the number left by collection, breakup, melting,
and activation — and in the prescribed-``N_c`` path, cloud number is reset to its
prescribed value immediately beforehand, as Fortran does. Because
`homogeneous_freezing_timescale` and `sink_limiting_timescale` are independently
configurable, both the mass and number rates are then capped consistently so one
limiter interval can never remove more than the residual.

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
is ``\dot{N}_\text{HM} = c_\text{splinter}\, \dot{q}_\text{rim}\, f_\text{HM}``
with ``c_\text{splinter} = 3.5 \times 10^8`` kg⁻¹ — equivalent to the
literature value of 350 splinters per mg of rime (the Fortran reference
stores this as `35.e+4` per gram and applies a ``\times 10^3`` kg→g
conversion at the call site). The mass rate uses an initial diameter
``D_\text{init,HM} = 10\;μ``m at ``ρ_i = 900`` kg/m³.

The 282 K warm-season shutoff (`splintering_surface_temperature_max`; `Inf`
disables it) needs a surface temperature, which
`compute_p3_surface_temperature!` obtains by scanning each column for its lowest
*active* cell — so it is correct over an immersed bottom, but cannot broadcast
across a vertical domain partition, since Oceananigans' distributed top/bottom
halo fills are currently no-ops. For gridless calculations, where no column
exists, the local air temperature is used. Fortran also raises ``D_\text{HM}`` to
1000 μm for ``n_\text{cat} > 1``; Breeze uses the single-category
threshold (250 μm) regardless, and correspondingly keeps the ``n_\text{cat} = 1``
cloud-riming branch enabled (`splintering_cloud_riming_scale = 1`).

## Droplet Activation (CCN)

Cloud droplet number is prognostic when CCN activation is enabled. Aerosol
activation follows the equilibrium Köhler-theory approach of
[Morrison and Grabowski (2007)](@cite MorrisonGrabowski2007), with
multi-mode lognormal aerosol distributions and a ``\sigma_g`` width parameter.
The activated number of each mode is:

```math
N_\text{act} = N_a\,\frac{1}{2}\left[1 - \text{erf}\!\left(\frac{2\,\ln(s_m/S)}{4.242\,\ln σ_g}\right)\right],
\qquad
s_m = \frac{2}{\sqrt{β_\text{act}}}\left(\frac{A_\text{act}}{3\, r_m}\right)^{3/2},
```

where ``s_m`` is the mode's critical supersaturation (a function of aerosol
size and solute activity, with the Kelvin parameter
``A_\text{act} = 2 M_w σ_v / (ρ_w R T)``), and ``S`` is the environmental
supersaturation. The per-mode counts are summed and capped at the total aerosol
number.

Breeze then tracks the unactivated pool explicitly, so activation cannot exceed
what remains in it:

```math
\dot{N}_\text{act} = \frac{\max\!\big(0,\; \min(N_\text{act}(S),\, n^{cl} + n^a) - n^{cl}\big)}{τ_\text{act}},
```

with ``τ_\text{act}`` = `aerosol.activation_timescale` (default 1 s), *separate*
from the Cooper ``τ_\text{nuc} = 10`` s. The same rate depletes ``ρn^a``, which
prevents the spurious re-activation that occurs when ``S`` rebounds after
autoconversion or partial evaporation has drained ``n^{cl}``. Activation is gated
on ``S > 10^{-6}`` (Fortran's `sup_cld` threshold), and the mass source is
``\dot{N}_\text{act}`` times the mass of a 1 μm-radius droplet.

Aerosol distributions are specified **per unit mass of air**: `AerosolMode`'s
`number_mixing_ratio` is in kg⁻¹, as are ``n^{cl}`` and ``n^a``; the prognostic
``ρn^a`` holds the ``ρ``-weighted count in m⁻³. See
[Prognostic Equations](@ref p3_prognostics) for how the reservoir is seeded.

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

For ``T > T_0`` the path depends on whether liquid fraction is active:

- **Liquid-fraction on** (`log_LiquidFrac = true` in Fortran;
  `cloud_warm_collection_rate` and `rain_warm_collection_rate` in Breeze):
  collected cloud and rain mass enter the liquid-coating reservoir
  ``q^{wi}`` instead of being shed.
- **Liquid-fraction off** (Fortran "original code" path): collected cloud is
  shed instantaneously back to rain as 1 mm drops with
  ``\dot{N}_{r,\text{shed}} = \dot{q}_{c,\text{shed}} / m_\text{shed}``,
  ``1/m_\text{shed} = 1.923 \times 10^6`` kg⁻¹ (Fortran `ncshdc`; Breeze reads the
  configurable `shed_drop_mass` so the rain-number limiter and the
  homogeneous-freezing residual budget the same value). Collected rain *mass* is
  left alone — `rain_warm_collection` is zeroed at rate-assembly time, matching
  the Fortran `qrcol` zero-mass branch — but the rain *number* sink
  (Fortran `nrcoll` / `nrcol`) fires in both branches.

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

P3's deposition step uses a coupled semi-analytic vapor balance
(`coupled_saturation_adjustment_rates`): cloud, rain, dry ice, and
liquid-coated ice all draw from (or release to) a common vapor reservoir. Each
species ``i`` contributes an inverse relaxation
timescale ``ε_i = 2π\, ρ\, D_v\, \mathcal{C}_i\, N_i`` (where
``\mathcal{C}_i`` is the relevant ventilation-enhanced capacitance integral
from the lookup tables), and the total is

```math
X = ε_{cl} + ε_r + ε_i\,\frac{1 + (L_s/c_p^d)\,dq_{v,s}^l/dT}{ξ^i} + ε_{iw}.
```

The dry-ice (``ε_i``, Fortran `epsi`) and coated-ice (``ε_{iw}``, Fortran
`epsiw`) coefficients share the same formula but select mutually exclusive
liquid-fraction regimes, split at `liquid_fraction_clipping_threshold`, so only
one of them is nonzero in any cell.

The deposition rate for ice category ``i`` is then
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)):

```math
\dot{q}_\text{dep,i} = \left[\frac{A\,ε_i}{X}
                            + \frac{(S_l - A/X)\,ε_i}{X\,τ_\text{sink}}\,
                              \left(1 - e^{-X\,τ_\text{sink}}\right)\right]
                      \frac{1}{ξ^i}
                      + \frac{(q_{v,s}^l-q_{v,s}^i)\,ε_i}{ξ^i},
\qquad ξ^i = 1 + \frac{L_s}{c_p^d}\frac{dq_{v,s}^i}{dT},
```

where ``S_l = q_v - q_{v,s}^l`` is the saturation deficit w.r.t. liquid and
``A`` sums two contributions: the Bergeron offset, and the external change in
liquid-relative supersaturation ``∂_t q^v - (dq_{v,s}/dT)\, ∂_t T``. Breeze
retains the Bergeron offset in full, and approximates the external part with
adiabatic cooling alone, ``∂_t T = -g\, w / cᵖᵐ`` and ``∂_t q^v = 0``, where
``w`` is the resolved (or parcel) vertical velocity. Resolved transport,
turbulent mixing, radiation, and user forcing therefore do not enter this
driver, even though they do act on the host thermodynamic equation. Supplying
the complete external tendency, as the Fortran `aaa` term is intended to carry,
remains a possible future improvement.

Sublimation is the negative branch (``\dot{q}_\text{dep} < 0``); the corresponding
number rate scales with the dry-ice number-to-mass ratio (recall that Breeze's
``q^i`` is already dry ice):

```math
\dot{N}_\text{sub} = -\dot{q}_\text{dep}\,\frac{N_i}{q^i}
                     + \dot{q}_\text{coat-evap}\,\frac{N_i}{q^i + q^{wi}},
```

where the second term is the number companion of liquid-coating evaporation
(Fortran `nlevp`), which shares the same ice-number sink.

Coupled liquid-coated ice (``F^l \ge`` `liquid_fraction_clipping_threshold`)
uses the liquid-side psychrometric factor
``1/ξ^l = 1/(1 + L_v^2 q_{v,s}^l/(c_p R_v T^2))`` instead of the ice-side
``1/ξ^i``, and carries no Bergeron contribution because the surface is already at
liquid saturation — matching the coupled liquid-ice branch in Fortran.

Deposition and sublimation are each scaled by an ad-hoc calibration factor
(`calibration_factor_deposition`, `calibration_factor_sublimation`, both 1 by
default), matching Fortran's `clbfact_dep` / `clbfact_sub`. Sublimation is
additionally capped at ``q^i/τ_\text{sink}`` (Fortran limits it to the dry-ice
mass per unit time) and deposition at ``q^v/τ_\text{sink}``.

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

The whole rate is bounded by ``q^i/τ_\text{sink}``; the physical heat-transfer
rate is the real limiter and the timescale is a numerical guard.

## Whole-Particle Clipping

Some particles must be transferred as a whole rather than eroded by a rate.
Breeze diagnoses the union of three predicates and drains each reservoir exactly
once over `refreezing_timescale`:

| Predicate | Condition | Fortran counterpart |
|-----------|-----------|--------------------|
| Warm fully-liquid | ``T \ge T_0`` and ``F^l > 1 - F^l_\text{small}`` | liquid-fraction clip |
| High liquid fraction | ``F^l > 0.99`` | "complete melting" diagnostic |
| Tiny warm ice | ``T \ge T_0`` and ``q^i + q^{wi} <`` `tiny_ice_to_rain_threshold` | `qsmall_dry` clip |

The first two require `liquid_fraction_active`. When any fires, the dry mass and
number go to rain as complete melting, the coating is shed to rain, and every
process that needs the clipped particle — deposition, coating exchange,
aggregation, riming, wet growth, splintering, above-freezing collection, the
number limiter, and both number corrections — is zeroed. Independent new-ice
sources (nucleation and immersion / homogeneous freezing) survive. The rime mass
and volume are drained through explicitly reconstructed companions
(``\text{CLIP}_q``, ``\text{CLIP}_b`` in [Prognostic Equations](@ref p3_prognostics))
so post-process rime and densification changes are removed exactly, rather than
by assuming the beginning-of-stage rime fraction.

Fortran applies its ``F^l > 0.99`` clip a second time *after* the ordinary
process updates, so Breeze reconstructs the post-process reservoirs from the
limited rates and applies the same clip to particles that crossed the threshold
during melting.

There is also a mirror-image clip below: with liquid fraction active, ``T < T_0``
and ``0 < F^l <`` `liquid_fraction_clipping_threshold`, the residual coating is
added to the refreezing rate rather than left as a vanishing ``q^{wi}``.

## Wet Growth and Refreezing

When the latent-heat release rate from collection exceeds what conduction
plus evaporative cooling can dissipate, ice enters wet growth.

The wet-growth capacity rate (Musil 1970):

```math
\dot{q}_\text{wg} = \big[K_a\,(T_0 - T) + 2π\,ρ\,L_s\,D_v\,(q_{v,s,0} - q_{v,\text{cld}})/L_f\big]\, f_v\, N_i,
```

where the ``2π`` factor multiplies *only* the latent (vapor-diffusion) term;
the sensible-conduction term ``K_a (T_0-T)`` carries no ``2π`` (matching the
Fortran `qwgrth` form).

Wet growth fires when the total collection
``\dot{q}_\text{ccol} + \dot{q}_\text{rcol}``
exceeds ``\dot{q}_\text{wg}`` and there is at least
``10^{-6}`` kg/kg of cloud plus rain to collect. The retained fraction is
``\dot{q}_\text{wg} / (\dot{q}_\text{ccol} + \dot{q}_\text{rcol})``.

Without liquid fraction, the retained portion becomes dense rime — the riming
rates are reduced to it and the new rime density is set to ``ρ_{r,\max}`` — while
the excess is shed as 1 mm drops. Only the excess *cloud* water is a new rain
*mass* source; excess collected rain simply stays rain, so it contributes to the
shed *number* only:

```math
\dot{q}_\text{shed,wg} = \dot{q}_\text{ccol,excess},\qquad
\dot{N}_\text{shed,wg} = \frac{\dot{q}_\text{ccol,excess} + \dot{q}_\text{rcol,excess}}{m_\text{shed}}.
```

The existing rime is simultaneously soaked to maximum density over
`rime_densification_timescale`: ``q^f \to q^i`` and ``b^f \to q^i / ρ_{r,\max}``.

With liquid fraction active, *all* collection becomes liquid coating
(Fortran `qwgrth1c` / `qwgrth1r`), the riming rates are zeroed, and no
densification flag is set. Refreezing then transfers ``q^{wi}`` back to
rime when ``T < T_0``, using the same ventilated heat balance as the wet-growth
capacity:

```math
\dot{q}_\text{frz} = N_i\, \max\!\big(0,\; \mathcal{C} f_v [K_a (T_0 - T)
                     + 2π\, ρ\, L_s D_v (q_{v,s,0} - q_v)/L_f]\big),
```

bounded by ``q^{wi} / τ_\text{sink}`` (Breeze uses a fixed
`sink_limiting_timescale`, default 10 s, in place of Fortran's
per-timestep ``q^{wi}/Δt`` cap).

### Shedding

Shedding is computed from a tabulated PSD integral over particles with
``D \ge 9`` mm (Rasmussen et al. 2011 threshold), matching the Fortran
reference:

```math
\dot{q}_\text{shed} = F^f\, \mathcal{I}_\text{shed}(\bar{m}, F^f, F^l, ρ^f, μ)\,
                      N_i\, F^l,
```

where ``\mathcal{I}_\text{shed}`` is the tabulated mass integral
``\int_{D \ge 9\,\text{mm}} m(D)\, N'(D)\, dD / N_i`` (Fortran `f1pr28`)
loaded from `p3_lookupTable_1`. The rate is bounded by
``q^{wi} / τ_\text{sink}`` (default 10 s) for stability. The shed mass is
added to rain; the shed number uses the ``1.928 \times 10^6`` per-kg
conversion (`shed_drop_mass_liqfrac`, 1 mm drops, identical to Fortran's
`nlshd` factor) — slightly different from the ``1.923 \times 10^6``
(`shed_drop_mass`) used by cloud and wet-growth shedding.

Shedding is gated off entirely when `liquid_fraction_active = false`. In that
configuration any coating left on the state (from a restart, say) is drained to
rain over `sink_limiting_timescale` instead, so ``q^{wi}`` cannot strand water.

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
`limit_vapor_rates` in `process_rate_helpers.jl` applies them, and — matching
Fortran's ordering — it runs *before* the per-species conservation budgets, so
those budgets see the already vapor-limited rates. The budgets themselves are
described under [Conservation Properties](@ref p3_prognostics).

## Sedimentation

Sedimentation is delegated to Oceananigans transport. Each prognostic field
falls at its tabulated, density-corrected velocity, diagnosed once per RK stage
into z-Face fields:

| Variable | Velocity | Reference |
|----------|---------|-----------|
| Cloud mass / number | mass-weighted ``V_m^{cl}``, number-weighted ``V_n^{cl}`` | DSD-integrated Stokes velocities |
| Rain mass / number | mass-weighted ``V_m^r``, number-weighted ``V_n^r`` | Gunn–Kinzer 1949 lookup tables |
| Ice mass / rime mass / rime volume / liquid coating | mass-weighted ``V_m^i`` | Mitchell–Heymsfield 2005 |
| Ice number | number-weighted ``V_n^i`` | Mitchell–Heymsfield 2005 |
| Advected sixth moment ``\tilde z^i`` | ``\tfrac{1}{2}(V_z^i + V_n^i)`` | Mitchell–Heymsfield 2005 |

All ice fall speeds are corrected by the air-density factor
``(ρ_s/ρ)^{0.54}`` with the 600 hPa, 253.15 K reference ``ρ_s`` for ice
and the surface ``ρ_s = p_0/(R_d\, T_0)`` for rain (matching the Fortran
`rhosur` / `rhosui`).

The prognostic sixth moment is ``\tilde z^i = \sqrt{z^i n^i}``, so its
sedimentation characteristic is the mean of the ``Z``- and ``N``-weighted particle
speeds rather than ``V_z^i`` alone. The purely reflectivity-weighted ``V_z^i`` is
still tabulated and kept as a diagnostic. One velocity is not exact for
independently size-sorted ``Z`` and ``N`` profiles; a coupled two-flux form
awaits a host tracer interface that can assemble one tendency from two moment
fluxes.

The Fortran adaptive `dt_left` Courant substepping is *not* part of P3
in Breeze; the host transport scheme is responsible for stability, and no fall
speed feeds back into a Courant constraint inside P3.

## Sixth-moment (``Z_i``) update

Breeze follows Fortran v5.5.0's active hybrid path. ``μ_i`` is held at its
pre-process (Table-3) value for the whole step. The "group 1" tendencies
(deposition / sublimation, melting, riming, refreezing, shedding, ``q^{wi}``
condensation and evaporation, aggregation, and the number limiters — that is,
every per-field tendency *minus* its group-2 sources) are integrated over
``τ_\text{sink}`` to get ``(q^i, q^{wi}, n^i, q^f, b^f)_\text{new}``. ``M_3`` is
then re-estimated from that state,

```math
M_{3,\text{new}} = \frac{6\, q^i_\text{total,new}}{π\, \bar{ρ}_i(μ_i)},\qquad
Z_\text{new} = G(μ_i)\,\frac{M_{3,\text{new}}^2}{n^i_\text{new}},\qquad
G(μ) = \frac{(6+μ)(5+μ)(4+μ)}{(3+μ)(2+μ)(1+μ)},
```

and contributed as ``(Z_\text{new} - Z_i)/τ_\text{sink}``. The bulk density
``\bar{ρ}_i(μ_i)`` here comes from **Table 1** read at the fixed ``μ_i``
(`ice_mean_density_at_fixed_shape`, Fortran
`proc_from_LUT_main3mom(12, …)`), not from Table 3: Table 3's density is a
function of ``z^i/q^i``, so using it inside a reconstruction of ``z^i`` would
close a loop on the value being replaced. Fortran states the same reason at
`microphy_p3.f90:4453-4457`.

For "group 2" initiation processes (deposition nucleation, immersion freezing
of cloud / rain, both splintering branches, homogeneous freezing of cloud /
rain), an explicit rate increment is added:

```math
\dot{M}_3 = \frac{6\,\dot{q}_\text{src}}{π\, ρ_i},\qquad
\dot{Z}_i = G(μ_\text{src})\,\frac{\dot{M}_3^2}{\dot{N}_\text{src}},
\qquad ρ_i = 900\;\text{kg/m}^3,
```

where ``μ_\text{src} = μ_r = 0`` for every source *except* homogeneous freezing
of cloud water, which uses the cloud shape ``μ_c`` diagnosed from the residual
cloud reservoir just before that process fires. This matches the Fortran
`update_zi_proc2` block once Fortran's `mu_r_constant = 0` runtime is taken into
account. The increment is zero unless both the mass and the number source are
positive.

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
| CCN activation | ``q^{cl}, n^{cl}, n^a`` | Köhler equilibrium, pool-capped, ``τ_\text{act}`` | [MorrisonGrabowski2007](@cite) |
| Autoconversion | ``q^{cl} \to q^r`` | KK2000 (default) / SB2001 / Kogan2013 | [KhairoutdinovKogan2000](@cite), [SeifertBeheng2001](@cite), [Kogan2013](@cite) |
| Accretion | ``q^{cl} \to q^r`` | KK2000 (default) / SB2001 / Kogan2013 | [KhairoutdinovKogan2000](@cite), [SeifertBeheng2001](@cite), [Kogan2013](@cite) |
| Rain self-collection / breakup | ``n^r`` | Verlinde–Cotton + KK2000/SB2001/Kogan2013 | [Morrison2015parameterization](@cite) |
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
| Shedding | ``q^{wi} \to q^r`` | Tabulated PSD integral, ``D \ge 9`` mm (Fortran `f1pr28`) | [MilbrandtEtAl2025liquidfraction](@cite) |
| Refreezing | ``q^{wi} \to q^f`` | Wet-growth form, ``T < T_0`` | [MilbrandtEtAl2025liquidfraction](@cite) |
| Whole-particle clipping | all ice fields ``\to q^r`` | ``F^l > 0.99``, warm fully-liquid, tiny warm ice | [MilbrandtEtAl2025liquidfraction](@cite) |
| Sedimentation | Cloud, rain, all ice fields | Tabulated; delegated to Oceananigans | [MilbrandtYau2005](@cite) |

## References for This Section

### Core P3 Process References
- [Morrison2015parameterization](@cite): Primary process formulations (Section 2).
- [Morrison2015part2](@cite): Process validation against observations.
- [MilbrandtEtAl2021](@cite): ``Z``-tendencies for three-moment ice.
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid-fraction processes (shedding, refreezing).
- [Morrison2025complete3moment](@cite): Complete three-moment process rates.

### Related References
- [KhairoutdinovKogan2000](@cite): Warm rain autoconversion (Breeze default).
- [SeifertBeheng2001](@cite): Alternative warm-rain (`autoAccr_param = 1`).
- [Kogan2013](@cite): Alternative warm-rain (`autoAccr_param = 3`).
- [MilbrandtYau2005](@cite): Multimoment sedimentation.
- [pruppacher2010microphysics](@cite): Cloud physics fundamentals.
- [rogers1989short](@cite): Cloud physics textbook.
