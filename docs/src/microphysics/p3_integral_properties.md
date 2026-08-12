# [Integral Properties](@id p3_integral_properties)

Bulk microphysical rates require population-averaged quantities computed by integrating
over the particle size distribution. P3 defines numerous integral properties organized
by physical concept.

Most ice-side integrals are pre-computed offline and stored in the Fortran
ASCII lookup table (see `create_p3_lookupTable_1.f90`
in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics));
Breeze loads the same file. The 1D rain integrals (mass- and number-weighted
fall speeds, evaporation ventilation) are tabulated at startup inside Breeze
from Chebyshev–Gauss quadrature evaluators in `rain_quadrature.jl`. The
integral formulations are from:
- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Fall speed, ventilation, collection

## General Form

All integral properties have the form:

```math
\langle X \rangle = \frac{\int_0^∞ X(D) N'(D)\, dD}{\int_0^∞ W(D) N'(D)\, dD}
```

where ``X(D)`` is the quantity of interest and ``W(D)`` is a weighting function
(often unity or particle mass).

## Fall Speed Integrals

Terminal velocity determines sedimentation rates. P3 computes two weighted fall speeds,
corresponding to `uns` and `ums` in the Fortran lookup table
(see [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b for
the underlying ``V(D)`` formulation; the integrated fall speeds are stored in
`p3_lookupTable_1.dat-v*`).

### Terminal Velocity Formulation

Individual particle fall speed follows the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best number formulation, which relates fall speed to particle mass, projected area, and air properties.
The formulation accounts for the transition from Stokes to turbulent flow regimes and includes
surface roughness effects. A density correction factor ``(ρ₀/ρ)^{0.54}`` is applied following
[Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007).

For mixed-phase particles (with liquid fraction ``F^l``), the fall speed is linearly interpolated
between the ice fall speed and the rain fall speed:

```math
V(D) = F^l V^r(D) + (1 - F^l) V^i(D)
```

The fall speed depends on the mass-diameter and area-diameter relationships, which vary
across the four particle regimes (see [Particle Properties](@ref p3_particle_properties)).

### Number-Weighted Fall Speed

```math
V_n = \frac{\int_0^∞ V(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}
```

This represents the average fall speed of particles and governs number flux:

```math
\mathcal{F}_{ρn^i} = -V_n\, ρn^i
```

### Mass-Weighted Fall Speed

```math
V_m = \frac{\int_0^∞ V(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

This governs mass flux:

```math
\mathcal{F}_{ρq^i} = -V_m\, ρq^i
```

## Deposition/Sublimation Integrals

Vapor diffusion to/from ice particles is enhanced by air flow around falling particles.

### Ventilation Factor

The ventilation factor ``f_v`` accounts for enhanced mass transfer:

```math
f_v = \mathbb{C}^\text{vent}_1 + \mathbb{C}^\text{vent}_2 \text{Re}^{1/2} \text{Sc}^{1/3}
```

where:
- ``\text{Re} = V D / ν`` is the Reynolds number
- ``\text{Sc} = ν / D^v`` is the Schmidt number
- ``\mathbb{C}^\text{vent}`` are the empirical ventilation coefficients from [HallPruppacher1976](@cite)

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
``Z^i = G(μ^i)\, M_3^2 / n^i`` rather than from this monomial closed form.

## Collection Integrals

Collection processes (aggregation, riming) require integrals over collision kernels.

### Aggregation

The collection kernel for ice-ice aggregation is:

```math
\mathcal{K}(D_1, D_2) = E^{ii} \frac{π}{4} (D_1 + D_2)^2 |V(D_1) - V(D_2)|
```

The aggregation rate integral:

```math
\mathcal{K}_\text{agg} = \int_0^∞ \int_0^∞ \mathcal{K}(D_1, D_2) N'(D_1) N'(D_2)\, dD_1 dD_2
```

### Ice-Cloud Collection (Riming)

```math
\dot{q}^{cl}_\text{rim} = E^{ci} q^{cl} \int_0^∞ A(D) V(D) N'(D)\, dD
```

### Ice-Rain Collection

Unlike cloud collection, this depends on the rain PSD as well, so it is a double
integral over both distributions and needs the rain slope parameter as an extra
table coordinate:

```math
\mathcal{K}^{ri} = \int_0^∞ \!\! \int_0^∞ \frac{π}{4} (D^i + D^r)^2\, |V(D^i) - V(D^r)|\,
                   N^{i\prime}(D^i)\, N^{r\prime}(D^r)\, dD^r\, dD^i .
```

The mass and number forms (Fortran `f1pr08`, `f1pr07`) are stored as ``\log_{10}``
values and exponentiated at runtime.
They live in the 6-D rain-ice block of Lookup Table 1 rather than the 5-D
ice-only block, and both share the same
``(\log \bar{m}, \log λ^r, F^f, F^l, ρ^f, μ^i)`` axes, so the interpolation indices
are computed once per lookup.

## Lambda Limiter Integrals

To prevent unphysical size distributions, P3 limits the slope parameter ``λ``
based on physical constraints. `IceLambdaLimiter` (`ice_lambda_limiter.jl`) holds
the two tabulated bounds:

| Field of `p3.ice.lambda_limiter` | Purpose | Fortran |
|----------------------------------|---------|---------|
| `small_q` | Upper bound on λ (prevents unrealistically small particles) | `f1pr09` |
| `large_q` | Lower bound on λ (prevents unrealistically large particles) | `f1pr10` |

Fortran clamps `nitot` against these bounds in place. Breeze instead diagnoses
the bounded number and feeds the difference back as the ``\dot{n}^{i}_\text{corr}``
relaxation tendency described in [Prognostic Equations](@ref p3_prognostics).

## Tabulation

For efficiency in simulations, integrals are organized into two table families,
both held in `p3_lookupTable_1.dat-v6.9-2momI`.

- **Table 1** — the 5-D ice-only block: fall speed, ventilation, bulk,
  cloud-collection, aggregation, and lambda-limiter integrals, on
  ``(\log \bar{m}, F^f, F^l, ρ^f, μ)`` axes (``μ`` is a singleton axis).
- **Table 2** — the 6-D ice–rain collection block embedded later in the same
  file, which adds ``\log λ^r`` as a coordinate.

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

P3 organises its integral properties by concept; the actual column count in
the Fortran 2-moment ice file (`p3_lookupTable_1.dat-v6.9-2momI`) is 21. The
ice–rain collection family (`qrcol`/`nrcol`) sits in the separate 6-D block of
the same file.

At runtime each ice-side integral is read from the corresponding Fortran
ASCII lookup table; the rain 1D tables are tabulated at startup inside
Breeze using Chebyshev–Gauss quadrature in `rain_quadrature.jl`. The
quadrature evaluators in `quadrature.jl::chebyshev_gauss_nodes_weights`
provide the nodes and weights; integrals are evaluated as compensated
sums of the integrand on those nodes.

## References for This Section

- [Morrison2015parameterization](@cite): Fall speed, ventilation, collection integrals (Section 2b and Appendix C)
- [HallPruppacher1976](@cite): Ventilation factor coefficients
