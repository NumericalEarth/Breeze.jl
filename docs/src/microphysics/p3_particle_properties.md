# [Particle Properties](@id p3_particle_properties)

Ice particles in P3 span a continuum from small pristine crystals to large rimed graupel.
The mass-diameter and area-diameter relationships vary across this spectrum, depending on
particle size and riming state.

The foundational particle property relationships are from
[Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization), Section 2.

## Mass-Diameter Relationship

The particle mass ``m(D)`` follows a piecewise power law that depends on maximum dimension ``D``,
rime fraction ``F^f``, and rime density ``ρ^f``. This formulation is given in
[Morrison2015parameterization](@citet) Eqs. 6, 7, 12, and 13.

### The Four Regimes

P3 defines four diameter regimes with distinct mass-diameter relationships:

**Regime 1: Small Spherical Ice** (``D < D^{th}``)

Small ice particles are assumed spherical with bulk ice density
([Morrison2015parameterization](@citet) Eq. 6):

```math
m(D) = \frac{π}{6} ρ^i D³
```

where ``ρ^i = 900`` kg/m³ is the bulk ice density, matching the Fortran P3
runtime convention. The slightly higher pure-ice value ``ρ^i_\text{pure} = 917``
kg/m³ (`pure_ice_density`) is reserved for the radar reflectivity diagnostic
and the melt densification of rime.

**Regime 2: Vapor-Grown Aggregates** (``D^{th} ≤ D < D^{gr}`` or unrimed)

Larger particles follow an empirical power law based on aircraft observations
of ice crystals and aggregates ([Morrison2015parameterization](@citet) Eq. 7):

```math
m(D) = α D^β
```

where ``α = 0.0121`` kg/m^β and ``β = 1.9`` are based on observations compiled in the
supplementary material of [Morrison2015parameterization](@citet).
This relationship captures the fractal nature of aggregated crystals.

**Regime 3: Graupel** (``D^{gr} ≤ D < D^{cr}``)

When particles acquire sufficient rime, they become compact graupel
with density ``ρ^{gr}`` ([Morrison2015parameterization](@cite) Eq. 13):

```math
m(D) = \frac{π}{6} ρ^{gr} D³
```

The graupel density ``ρ^{gr}`` depends on the rime fraction and rime density
([Morrison2015parameterization](@citet) Eq. 16):

```math
ρ^{gr} = F^f ρ^f + (1 - F^f) ρ^d
```

where ``ρ^d`` is the density of the deposited (vapor-grown) ice component.

**Regime 4: Partially Rimed** (``D ≥ D^{cr}``)

The largest particles have a rimed core with unrimed aggregate extensions
([Morrison2015parameterization](@citet) Eq. 12):

```math
m(D) = \frac{α}{1 - F^f} D^β
```

### Threshold Diameters

The transitions between regimes occur at critical diameters determined by
equating masses ([Morrison2015parameterization](@citet) Eqs. 8, 14, and 15):

**Spherical-Aggregate Threshold** ``D^{th}`` (Eq. 8):

The diameter where spherical mass equals aggregate mass:

```math
D^{th} = \left( \frac{6α}{π ρ^i} \right)^{1/(3-β)}
```

**Aggregate-Graupel Threshold** ``D^{gr}`` (Eq. 15):

The diameter where aggregate mass equals graupel mass:

```math
D^{gr} = \left( \frac{6α}{π ρ^{gr}} \right)^{1/(3-β)}
```

**Graupel-Partial Threshold** ``D^{cr}`` (Eq. 14):

The diameter where graupel mass equals partially rimed mass:

```math
D^{cr} = \left( \frac{6α}{π ρ^{gr} (1 - F^f)} \right)^{1/(3-β)}
```

### Deposited Ice Density

The density of the vapor-deposited (unrimed) component ``ρ^d`` is derived from
the constraint that total mass equals rime mass plus deposited mass. The form
below is algebraically equivalent to [Morrison2015parameterization](@citet)
Eq. 17 (which expresses ``ρ^d`` directly in terms of the threshold diameters
``D^{cr}`` and ``D^{gr}``), rewritten here as a closed-form expression in
``F^f`` and ``ρ^f``:

```math
ρ^d = \frac{F^f ρ^f}{(β - 2) \frac{k - 1}{(1 - F^f)k - 1} - (1 - F^f)}
```

where ``k = (1 - F^f)^{-1/(3-β)}``.

The relations above are what the official P3 generator integrates when it builds
the lookup tables; Breeze itself never evaluates ``m(D)`` or the thresholds at
runtime, it interpolates the resulting bulk quantities. See
[P3 Examples and Visualization](@ref p3_examples) for the tabulated mean
diameter and bulk density plotted against mean particle mass.

## Area-Diameter Relationship

The projected cross-sectional area ``A(D)`` determines collection rates and fall speed.
These relationships are described in [Morrison2015parameterization](@citet) Section 2b
(area-diameter forms are not numbered as equations in the paper).

**Small Spherical Ice** (``D < D^{th}``):

```math
A(D) = \frac{π}{4} D²
```

**Nonspherical Ice** (aggregates):

```math
A(D) = \mathbb{C}^A_1 D^{\mathbb{C}^A_2}
```

with the exponent ``\mathbb{C}^A_2 = 1.88`` and the coefficient
``\mathbb{C}^A_1 ≈ 0.1318`` m``^{0.12}``. Both are the empirical values of
[Mitchell1996powerlaws](@citet) for aggregates of side planes, bullets,
and columns and assemblages of planar polycrystals, as adopted by
[Morrison2015parameterization](@citet). Mitchell (1996) quotes the coefficient in
cgs as ``0.2285`` cm``^{0.12}``; the Fortran reference and Breeze both convert in
place by multiplying with ``100^{\mathbb{C}^A_2-2}``.

**Graupel**:

Reverts to spherical:

```math
A(D) = \frac{π}{4} D²
```

**Partially Rimed**:

Per official P3 code, the projected area is interpolated by particle mass between
the unrimed and graupel relationships, rather than a simple Fᶠ weighting:

```math
A(D) = A^{ur} + \frac{m^{pr} - m^{ur}}{m^{gr} - m^{ur}} \left(A^{gr} - A^{ur}\right)
```

with ``A^{ur} = \mathbb{C}^A_1 D^{\mathbb{C}^A_2}``, ``A^{gr} = \frac{π}{4} D^2``,
``m^{ur} = α D^β``, ``m^{gr} = \frac{π}{6} ρ^{gr} D^3``, and
``m^{pr} = α D^β / (1 - F^f)`` from the partially rimed mass law.

## Terminal Velocity

The official P3 code computes terminal velocity using the
[Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005) Best-number drag formulation with the
regime-dependent ``m(D)`` and ``A(D)`` relationships. The resulting fall speeds
are stored in lookup tables and include the air-density correction
``(ρ₀/ρ)^{0.54}`` following [Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007).

Breeze implements this full Best-number formulation directly in the quadrature routines,
ensuring consistency with the lookup tables. For mixed-phase particles, the velocity
interpolates between the ice and rain fall speeds based on liquid fraction.

## Particle Density

The effective density ``ρ(D)`` is defined as mass divided by the volume
of a sphere with diameter ``D``:

```math
ρ(D) = \frac{m(D)}{(π/6) D³} = \frac{6 m(D)}{π D³}
```

This definition is convenient for comparing particles of different types
and connects directly to the mass-diameter relationship.

Table 1 carries the PSD-integrated version of this quantity in its mean-density
column, plotted against mean particle mass and rime fraction in
[P3 Examples and Visualization](@ref p3_examples).

## Effect of Riming

Riming dramatically affects particle properties. This is the key insight of P3 that enables
continuous evolution without discrete category conversions
([Morrison2015parameterization](@citet) Section 2b):

| Property | Unrimed Aggregate | Heavily Rimed Graupel |
|----------|-------------------|----------------------|
| Mass | ``α D^β`` | ``(π/6) ρ^{gr} D³`` |
| Density | Low (~100 kg/m³) | High (~500 kg/m³) |
| Fall speed | Slow | Fast |
| Collection efficiency | Low | High |

## Rime Density Parameterization

The rime density ``ρ^f`` depends on the collection conditions during riming. The
parameterization follows [Cober and List (1993)](@cite CoberList1993) as implemented in
[Morrison2015parameterization](@citet). The rime density is computed as a function of
the impact parameter ``R_\text{imp}``, which depends on droplet size, impact velocity, and temperature:

```math
ρ^f = \begin{cases}
(0.051 + 0.114 R_\text{imp} - 0.0055 R_\text{imp}^2) \times 1000 & R_\text{imp} \le 8 \\
611 + 72.25 (R_\text{imp} - 8) & R_\text{imp} > 8
\end{cases}
```

Wherever the particle carries rime volume, the diagnosed rime density is bounded
into `[minimum_rime_density, maximum_rime_density]`:
- ``ρ^f_\text{min} = 50`` kg/m³ is the minimum rime density
- ``ρ^f_\text{max} = 900`` kg/m³ is the maximum rime density

`consistent_rime_state` applies those bounds only when ``ρb^f`` is non-negligible;
an unrimed particle keeps the canonical ``ρ^f = 0`` rather than being pushed up to
``ρ^f_\text{min}``, and the table lookup clamps that 0 onto the first rime-density
coordinate on the way in (see [Size Distribution](@ref p3_size_distribution)).

The rime density affects the graupel density ``ρ^{gr}`` and thus the regime thresholds.
As particles rime more heavily, they become denser and more spherical.

!!! note "Official P3 implementation details"
    The Fortran scheme clamps ``R_\text{imp}`` to [1, 12] before applying the Cober–List fit;
    the linear branch for ``R_\text{imp} > 8`` is extended to ``R_\text{imp} = 12`` so that
    ``ρ^f = 900`` kg/m³. The lookup tables discretize ``ρ^f`` on an uneven grid
    (50, 250, 450, 650, 900 kg/m³) and interpolate between bins; `rime_density_index`
    maps a physical ``ρ^f`` onto that grid.

## Summary

The P3 mass-diameter relationship captures the full spectrum of ice particle types:

1. **Small crystals**: Dense, spherical approximation
2. **Aggregates**: Fractal structure, low density, follows ``m ∝ D^{1.9}``
3. **Graupel**: Compact, dense from riming
4. **Partially rimed**: Large aggregates with rimed cores

The transitions occur naturally through the regime thresholds, which depend only on the
predicted rime fraction and rime density—no arbitrary conversion terms required.

## References for This Section

- [Morrison2015parameterization](@cite): Primary source for m(D), A(D), V(D) relationships
- [Morrison2015part2](@cite): Validation of particle property parameterizations
- [pruppacher2010microphysics](@cite): Background on ice particle physics
