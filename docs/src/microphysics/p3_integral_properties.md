# [Integral Properties](@id p3_integral_properties)

Bulk microphysical rates require population-averaged quantities computed by integrating
over the particle size distribution. P3 defines numerous integral properties organized
by physical concept.

Most ice-side integrals are pre-computed offline and stored in the Fortran
ASCII lookup tables (see `create_p3_lookupTable_1.f90` and `create_p3_lookupTable_3.f90`
in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics));
Breeze loads the same files. The 1D rain integrals (mass- and number-weighted
fall speeds, evaporation ventilation) are tabulated at startup inside Breeze
from Chebyshev–Gauss quadrature evaluators in `rain_quadrature.jl`. The
integral formulations are from:
- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Fall speed, ventilation, collection
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Sixth moment integrals for 3-moment ice
- [Morrison et al. (2025)](@cite Morrison2025complete3moment): Complete 3-moment lookup tables

## General Form

All integral properties have the form:

```math
\langle X \rangle = \frac{\int_0^∞ X(D) N'(D)\, dD}{\int_0^∞ W(D) N'(D)\, dD}
```

where ``X(D)`` is the quantity of interest and ``W(D)`` is a weighting function
(often unity or particle mass).

## Fall Speed Integrals

Terminal velocity determines sedimentation rates. P3 computes three weighted fall speeds,
corresponding to `uns`, `ums`, `uzs` in the Fortran lookup tables
(see [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b for
the underlying ``V(D)`` formulation; the integrated fall speeds are stored in
`p3_lookupTable_1.dat-v*` and described in
[Morrison et al. (2025)](@cite Morrison2025complete3moment)).

### Terminal Velocity Formulation

Individual particle fall speed follows the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best number formulation, which relates fall speed to particle mass, projected area, and air properties.
The formulation accounts for the transition from Stokes to turbulent flow regimes and includes
surface roughness effects. A density correction factor ``(ρ₀/ρ)^{0.54}`` is applied following
[Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007).

For mixed-phase particles (with liquid fraction ``F^l``), the fall speed is linearly interpolated
between the ice fall speed and the rain fall speed:

```math
V(D) = F^l V_{rain}(D) + (1 - F^l) V_{ice}(D)
```

The fall speed depends on the mass-diameter and area-diameter relationships, which vary
across the four particle regimes (see [Particle Properties](@ref p3_particle_properties)).

### Number-Weighted Fall Speed

```math
V_n = \frac{\int_0^∞ V(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}
```

This represents the average fall speed of particles and governs number flux:

```math
F_N = -V_n \cdot N
```

### Mass-Weighted Fall Speed

```math
V_m = \frac{\int_0^∞ V(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

This governs mass flux:

```math
F_L = -V_m \cdot L
```

### Reflectivity-Weighted Fall Speed

For 3-moment ice, the 6th moment flux uses:

```math
V_z = \frac{\int_0^∞ V(D) D^6 N'(D)\, dD}{\int_0^∞ D^6 N'(D)\, dD}
```

## Deposition/Sublimation Integrals

Vapor diffusion to/from ice particles is enhanced by air flow around falling particles.

### Ventilation Factor

The ventilation factor ``f_v`` accounts for enhanced mass transfer:

```math
f_v = a_v + b_v \text{Re}^{1/2} \text{Sc}^{1/3}
```

where:
- ``\text{Re} = V D / ν`` is the Reynolds number
- ``\text{Sc} = ν / D_v`` is the Schmidt number
- ``a_v, b_v`` are empirical coefficients from [HallPruppacher1976](@cite)

### Ventilation Integrals

`IceDeposition` (`ice_deposition.jl`) holds two wet-PSD ventilation components
for deposition / sublimation and four dry-PSD components for liquid-fraction
melting:

| Field of `p3.ice.deposition` | Description | Integration / Routing | Fortran |
|------------------------------|-------------|-----------------------|---------|
| `small_ice_ventilation_constant` | Constant melting component | ``D \le D_\text{crit}``; meltwater goes to rain | `f1pr24` |
| `small_ice_ventilation_reynolds` | Re-dependent melting component | ``D \le D_\text{crit}``; meltwater goes to rain | `f1pr25` |
| `large_ice_ventilation_constant` | Constant melting component | ``D > D_\text{crit}``; meltwater stays on ice | `f1pr26` |
| `large_ice_ventilation_reynolds` | Re-dependent melting component | ``D > D_\text{crit}``; meltwater stays on ice | `f1pr27` |
| `ventilation` | Constant deposition / sublimation component | Wet PSD, all sizes | `f1pr05` |
| `ventilation_enhanced` | Re-dependent deposition / sublimation component | Wet PSD, ``D \ge 100`` μm | `f1pr14` |

The ``D_\text{crit}`` split controls where meltwater is routed. It is distinct
from the 100 μm Hall-Pruppacher ventilation transition: below 100 μm only the
constant coefficient contributes, while larger particles also contribute to the
Re-dependent component. Breeze's melting rate reads `f1pr24`–`f1pr27`; dry-ice
deposition / sublimation reads the wet-PSD `f1pr05` / `f1pr14` pair (see
[Size Distribution](@ref p3_size_distribution)).

## Bulk Property Integrals

Population-averaged properties for radiation, radar, and diagnostics.

### Effective Radius

Important for radiation parameterizations. Following the
Francis et al. (1994) / Fu (1996, Eq. 3.11 in *J. Climate*) definition:

```math
r_\text{eff} = \frac{3}{4\, ρ_i^*}
               \frac{\int_0^∞ m(D)\, N'(D)\, dD}{\int_0^∞ A(D)\, N'(D)\, dD},
```

with ``ρ_i^* = 916.7`` kg/m³. With liquid fraction active the integrands
include the ``F^l``-blended mass and projected area (i.e.
``m = (1-F^l) m_\text{ice} + F^l\, (π/6)\, ρ_w D^3`` and
``A = (1-F^l) A_\text{ice} + F^l\, (π/4) D^2``).

### Mean Diameter

Mass-weighted mean particle size:

```math
D_m = \frac{\int_0^∞ D \cdot m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

### Mean Density

Mass-weighted particle density:

```math
ρ_m = \frac{\int_0^∞ ρ(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

### Reflectivity

Radar reflectivity factor. The pure ``D^6`` closed form

```math
Z_\text{mono} = \int_0^∞ D^6 N'(D)\, dD = N₀ \frac{Γ(μ + 7)}{λ^{μ+7}}
```

applies only to a single power-law mass regime. In P3 the tabulated
reflectivity column integrates the equal-volume ``D_\text{eq}^6`` over the
full piecewise ``m(D)`` (i.e. ``(6/(π\, ρ_i^*))^2 m(D)^2`` per particle,
with ``ρ_i^* = 917`` kg/m³); for partially melted particles it switches to
a Rayleigh–Mie wet-ice mixing rule. The
runtime ``Z_i`` is recomputed via the active hybrid path
``Z_i = G(μ_i)\, M_3^2 / N_i`` rather than from this monomial closed form.

## Collection Integrals

Collection processes (aggregation, riming) require integrals over collision kernels.

### Aggregation

The collection kernel for ice-ice aggregation is:

```math
K_{agg}(D_1, D_2) = E_{agg} \frac{π}{4} (D_1 + D_2)^2 |V(D_1) - V(D_2)|
```

The aggregation rate integral:

```math
I_{agg} = \int_0^∞ \int_0^∞ K_{agg}(D_1, D_2) N'(D_1) N'(D_2)\, dD_1 dD_2
```

### Ice-Cloud Collection (Riming)

```math
\frac{dq^f}{dt} = E_{ic} q^{cl} \int_0^∞ A(D) V(D) N'(D)\, dD
```

### Ice-Rain Collection

Unlike cloud collection, this depends on the rain PSD as well, so it is a double
integral over both distributions and needs the rain slope parameter as an extra
table coordinate:

```math
I_{ir} = \int_0^∞ \!\! \int_0^∞ \frac{π}{4} (D_i + D_r)^2\, |V(D_i) - V(D_r)|\,
         N_i'(D_i)\, N_r'(D_r)\, dD_r\, dD_i .
```

The mass and number forms (Fortran `f1pr08`, `f1pr07`) are stored as ``\log_{10}``
values and exponentiated at runtime; the sixth-moment form (`m6collr`) is not.
They live in the 6-D rain-ice block of Lookup Table 1 rather than the 5-D
ice-only block, and all three share the same
``(\log \bar{m}, \log λ_r, F^f, F^l, ρ^f, μ)`` axes, so the interpolation indices
are computed once per lookup.

## Sixth Moment Integrals

For 3-moment ice ([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024)),
P3 tracks the 6th moment ``Z`` which requires additional integrals
for each process affecting reflectivity. These are documented in
[Morrison et al. (2025)](@cite Morrison2025complete3moment) and stored
in the 3-moment lookup table (`p3_lookupTable_1.dat-v*_3momI`).

| Process | Integral(s) | Physical Meaning |
|---------|-------------|------------------|
| Riming | `rime` | Z change from rime accretion |
| Deposition | `deposition`, `deposition1` | Z change from vapor growth: a constant ventilation component over all sizes and a Re-dependent component for ``D \ge 100`` μm (Fortran `m6dep` / `m6dep1`) |
| Melting | `melt1`, `melt2` | Melt-to-rain piece (``D \le D_\text{th}``) and ``q^{wi}``-routed piece for the liquid-fraction path |
| Aggregation | `aggregation` | Z change from aggregation |
| Shedding | `shedding` | Z change from liquid shedding |
| Sublimation | `sublimation`, `sublimation1` | Same constant / Re-dependent decomposition as deposition, with sign and a ``\dot{N}_0`` contribution |

`IceSixthMoment` also carries `melt_all1` / `melt_all2`, the all-``D`` melting
integrands for the non-liquid-fraction path. The Fortran file has no such
columns — the all-``D`` distinction only arises in Breeze's own quadrature — so
reading a Fortran table sets them equal to `melt1` / `melt2`.

## Lambda Limiter Integrals

To prevent unphysical size distributions, P3 limits the slope parameter ``λ``
based on physical constraints. `IceLambdaLimiter` (`ice_lambda_limiter.jl`) holds
the two tabulated bounds:

| Field of `p3.ice.lambda_limiter` | Purpose | Fortran |
|----------------------------------|---------|---------|
| `small_q` | Upper bound on λ (prevents unrealistically small particles) | `f1pr09` |
| `large_q` | Lower bound on λ (prevents unrealistically large particles) | `f1pr10` |

Fortran clamps `nitot` against these bounds in place. Breeze instead diagnoses
the bounded number and feeds the difference back as the ``\text{N-CORR}_i``
relaxation tendency described in [Prognostic Equations](@ref p3_prognostics).

## Tabulation

For efficiency in simulations, integrals are organized into three table families.
Only two files are read: `p3_lookupTable_1.dat-v6.9-3momI` (or `-2momI`), which
holds the first two families, and `p3_lookupTable_3.dat-v1.4`.

- **Table 1** — the 5-D ice-only block: fall speed, ventilation, bulk,
  cloud-collection, aggregation, sixth-moment, and lambda-limiter integrals, on
  ``(\log \bar{m}, F^f, F^l, ρ^f, μ)`` axes (``μ`` is a singleton in 2-moment mode).
- **Table 2** — the 6-D ice–rain collection block embedded later in the same
  file, which adds ``\log λ_r`` as a coordinate.
- **Table 3** — the three-moment diagnostic lookup for ``μ^i`` and a companion
  mean-density column. (The slope ``λ^i`` is *not* in Table 3 — it is
  recovered at runtime from Table 1 using the diagnosed ``μ^i``.) Requesting
  three-moment ice without this file is an error, matching Fortran's hard stop:
  a silent fallback to the two-moment ``μ`` closure would give different results.

The `MultiIceCategory` scaffolding has no tabulated inter-category collection
family; that kernel is an unimplemented placeholder.

```@example p3_integrals
using Breeze
using Logging: NullLogger, with_logger

# The default constructor reads the Fortran ASCII lookup tables
# (downloaded automatically on first use).
p3 = with_logger(NullLogger()) do
    PredictedParticlePropertiesMicrophysics()
end

fs = p3.ice.fall_speed
println("Tabulated fall speed integrals from Fortran tables:")
println("  Number-weighted: $(typeof(fs.number_weighted))")
println("  Mass-weighted:   $(typeof(fs.mass_weighted))")
```

## Summary

P3 organises its integral properties by concept; the actual column counts in
the Fortran tables are 21 in the 2-moment ice file (`p3_lookupTable_1.dat-v*`)
and 31 in the 3-moment ice file (`*_3momI`). The ten extra 3-moment columns are
the reflectivity-weighted fall speed and the nine sixth-moment integrals
(`m6rime, m6dep, m6dep1, m6mlt1, m6mlt2, m6agg, m6shd, m6sub, m6sub1`). The
ice–rain collection family (`qrcol`/`nrcol`, plus `m6collr` in 3-moment mode)
sits in the separate 6-D block of the same file.

At runtime each ice-side integral is read from the corresponding Fortran
ASCII lookup table; the rain 1D tables are tabulated at startup inside
Breeze using Chebyshev–Gauss quadrature in `rain_quadrature.jl`. The
quadrature evaluators in `quadrature.jl::chebyshev_gauss_nodes_weights`
provide the nodes and weights; integrals are evaluated as compensated
sums of the integrand on those nodes.

## References for This Section

- [Morrison2015parameterization](@cite): Fall speed, ventilation, collection integrals (Section 2b and Appendix C)
- [HallPruppacher1976](@cite): Ventilation factor coefficients
- [MilbrandtEtAl2021](@cite): Sixth moment integrals for three-moment ice (Table 1)
- [MilbrandtEtAl2024](@cite): Updated three-moment formulation
- [Morrison2025complete3moment](@cite): Complete three-moment lookup table (29 quantities)
